from metaflow import (
    FlowSpec,
    Parameter,  # pylint: disable=no-name-in-module
    step,
)
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from sklearn.decomposition import PCA

from vectorgeo.models import initialize_triplet, triplet_loss
from vectorgeo.raster import unpack_array, extend_negatives
from vectorgeo import transfer
from vectorgeo import constants as c

import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

class HDF5IterableDataset(IterableDataset):
    def __init__(self, hdf5_filepath, batch_size):
        self.hdf5_filepath = hdf5_filepath
        self.batch_size = batch_size

    def __iter__(self):
        with h5py.File(self.hdf5_filepath, 'r') as f:
            n_samples = len(f['all_data'])
            for i in range(0, n_samples, self.batch_size):
                yield torch.tensor(f['all_data'][i:i+self.batch_size], dtype=torch.float32)


class TrainLandCoverTripletFlow(FlowSpec):
    """
    Flow for producing training dataset of paired anchor/neighbor land cover images
    from Copernicus LULC data. The end result is a sequence of .npy files on the targeted
    S3 bucket, each of which contains matched anchor/neighbor pairs of land use / land cover
    data encoded as integers and with the resolution specified by the `patch_size` parameter.
    """

    epochs = Parameter("epochs", help="Number of epochs to train for", default=20)

    batch_size = Parameter("batch_size", help="Batch size for training", default=512)

    embed_dim = Parameter(
        "embed_dim", help="Dimension of the embedding space", default=16
    )

    num_filters = Parameter(
        "num_filters", help="Number of filters in each convolutional layer", default=64
    )

    n_linear = Parameter(
        "n_linear", help="Number of units in the final dense layers", default=64
    )

    n_conv_blocks = Parameter(
        "n_conv_blocks",
        help="Number of non-downsample convolutional ResNet blocks per downsample in the embedding network",
        default=2,
    )

    n_train = Parameter(
        "n_train",
        help="Number of training examples to use for training",
        default=500_000,
    )

    model_filename = Parameter(
        "model_filename",
        help="Filename to save the model to",
        required=True,
    )

    device = Parameter(
        "device", help="Device to use for PyTorch operations", default="cuda"
    )
    
    image_size = Parameter(
        "image_size", help="Dimensions of input image data",
        default=32)

    @step
    def start(self):
        """
        Loads the training files from S3 and starts the training process.
        """

        # Get list of files in the S3 bucket
        keys = transfer.ls_s3("train/")
        keys = list(filter(lambda x: x.endswith(".npy"), keys))
        print("Found training data files {} files".format(len(keys)))

        self.image_shape = (3, 24, self.image_size, self.image_size,)
                
        n_loaded = 0       
        self.hdf5_filepath = os.path.join(c.TMP_DIR, 'train.h5') 
        
        if os.path.exists(self.hdf5_filepath): os.remove(self.hdf5_filepath)
            
        with h5py.File(self.hdf5_filepath, 'w') as f:
            # Create an expandable dataset in the HDF5 file
            dset = f.create_dataset("all_data", shape=(0,) + self.image_shape, maxshape=(None,) + self.image_shape, dtype='f4')

            for key in keys:
                print(f"Inserting data from file {key} into HDF5 store")
                local_filepath = os.path.join(c.TMP_DIR, os.path.basename(key))
                transfer.download_file(key, local_filepath)
                arr = np.load(local_filepath)

                # Your preprocessing logic here
                xs_one_hot = unpack_array(arr[:, [0]])
                xs_dem = arr[:, 1]
                xs_dem = extend_negatives(xs_dem)
                xs_dem = np.transpose(xs_dem, (0, 3, 1, 2))[:, :, np.newaxis]
                xs_combined = np.concatenate([xs_one_hot, xs_dem], axis=2)

                # Incrementally save to HDF5
                new_shape = (dset.shape[0] + xs_combined.shape[0],) + self.image_shape
                dset.resize(new_shape)
                dset[-xs_combined.shape[0]:] = xs_combined
                
                n_loaded += len(xs_combined)

                if n_loaded > self.n_train:
                    print(f"Loaded {n_loaded} samples; stopping")
                    break
                    
        # Save these off for later analyses
        # We pick off the first N examples from the anchor pool
        self.test_batch = xs_combined[0:1024, 0]

        print(f"Loading phase finished with {n_loaded} data samples loaded!")
        self.input_shape = self.image_shape[1:]
        
        print("Initializing triplet model...")
        self.embedding_network, optimizer = initialize_triplet(
            self.input_shape,
            self.n_conv_blocks,
            self.embed_dim,
            self.num_filters,
            self.n_linear,
        )

        self.embedding_network.to(self.device)  # Move model to GPU
        
        dataset = HDF5IterableDataset(self.hdf5_filepath, self.batch_size)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

        # Training loop
        self.history = {"loss": []}
        print("Beginning training loop...")
        for epoch in range(self.epochs):
            print(f"---Beginning epoch {epoch}...")
            for batch in dataloader:
                optimizer.zero_grad()
                
                anchor_embedding   = self.embedding_network(batch[:,0].to(self.device))
                positive_embedding = self.embedding_network(batch[:,1].to(self.device))
                negative_embedding = self.embedding_network(batch[:,2].to(self.device))
                merged_vector = torch.cat(
                    [anchor_embedding, positive_embedding, negative_embedding], dim=1
                )
                loss = triplet_loss(merged_vector)
                loss.backward()
                optimizer.step()
                self.history["loss"].append(loss.item())

        self.next(self.end)

    @step
    def end(self):
        """
        Generate statistics on the number of active dimensions after PCA & upload to S3.
        """

        # Save the model to a temporary directory
        model_path = os.path.join(c.TMP_DIR, self.model_filename)
        torch.save(self.embedding_network, model_path)

        # Upload the model to S3
        transfer.upload_file(f"models/{self.model_filename}", model_path)

        # Convert test_batch to PyTorch tensor and run through the model
        test_batch_tensor = torch.tensor(self.test_batch, dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            zs = self.embedding_network(test_batch_tensor).cpu().numpy()

        # Transform with PCA
        pca = PCA(n_components=self.embed_dim)
        zs_pca = pca.fit_transform(zs)

        # Print out the number of eigenvalues
        # needed to reach threshold of variance explained
        variance_threshold = 0.99
        variance_explained = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(variance_explained > variance_threshold)[0][0]
        print(
            f"Number of components needed to reach {variance_threshold} variance explained: {n_components}"
        )

        # Generate a plot showing the activity of each dimension before/after PCA
        # Create a figure
        plt.figure(figsize=(10, 6))

        # Create a GridSpec layout
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Create the subplots
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[1, :])

        # Plot of embeddings without PCA
        ax0.imshow(zs[0:64])
        ax0.set_ylabel("Different vectors")
        ax0.set_xlabel("Vector dimensions")

        # Plot of embeddings with PCA
        ax1.imshow(zs_pca[0:64])
        ax1.set_ylabel("Different vectors")
        ax1.set_xlabel("Vector dimensions (PCA)")

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.savefig("figures/embeddings.png")


if __name__ == "__main__":
    TrainLandCoverTripletFlow()

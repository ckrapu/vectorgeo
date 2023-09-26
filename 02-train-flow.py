from metaflow import (
    FlowSpec,
    Parameter,  # pylint: disable=no-name-in-module
    step,
)

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
from torch.utils.data import DataLoader, TensorDataset


class TrainLandCoverTripletFlow(FlowSpec):
    """
    Flow for producing training dataset of paired anchor/neighbor land cover images
    from Copernicus LULC data. The end result is a sequence of .npy files on the targeted
    S3 bucket, each of which contains matched anchor/neighbor pairs of land use / land cover
    data encoded as integers and with the resolution specified by the `patch_size` parameter.
    """

    epochs = Parameter("epochs", help="Number of epochs to train for", default=20)

    batch_size = Parameter("batch_size", help="Batch size for training", default=64)

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

    @step
    def start(self):
        """
        Loads the training files from S3 and starts the training process.
        """

        # Get list of files in the S3 bucket
        keys = transfer.ls_s3("train/")
        keys = list(filter(lambda x: x.endswith(".npy"), keys))
        print("Found training data files {} files".format(len(keys)))

        arrays = []
        n_loaded = 0
        for key in keys:
            local_filepath = os.path.join(c.TMP_DIR, os.path.basename(key))
            transfer.download_file(key, local_filepath)
            # Read each file in the list and append it to the arrays list
            arr = np.load(local_filepath)
            print(f"....Read file at {key} with {np.sum(np.isnan(arr))} NaNs in array")

            arrays += [arr]
            n_loaded += len(arr)

            if n_loaded > self.n_train:
                print(f"Loaded {n_loaded} samples; stopping")
                break

        # Convert from integer to one-hot encoding
        # We don't preprocess as one-hot because the storage is way larger.
        print(f"Unpacking {len(arrays)} arrays...")

        # The shape of the arrays is (N, 3, K, H, W) where the 3 is for 
        # anchor, positive, negative and the K is for the different classes.
        xs_one_hot = np.concatenate([unpack_array(xs[:, [0]]) for xs in arrays], axis=0)
        print(f"Shape of land cover trainin component: {xs_one_hot.shape}")

        xs_dem = np.concatenate([xs[:, 1] for xs in arrays], axis=0)
        xs_dem = extend_negatives(xs_dem)

        # Move last axis to be second and add a singleton dimension
        # for concatenation with the one-hot encoded data
        xs_dem = np.transpose(xs_dem, (0, 3, 1, 2))[:, :, np.newaxis]

        print(f"Shape of DEM training component: {xs_dem.shape}")

        xs_combined = np.concatenate([xs_one_hot, xs_dem], axis=2)
        print(f"Shape of combined training component: {xs_combined.shape}")

        self.input_shape = xs_combined.shape[2:]
        anchors, positives, negatives = (
            xs_combined[:, 0],
            xs_combined[:, 1],
            xs_combined[:, 2],
        )
        labels = np.zeros((len(anchors), 1))
        print(
            f"Loaded {len(arrays)} files; resulting stacked array has shape {xs_combined.shape}"
        )

        # Save off a select group of images
        # to use for downstream information content
        # assessment with PCA
        self.test_batch = xs_combined[0:1024, 0]
        print("Initializing triplet model...")
        self.embedding_network, optimizer = initialize_triplet(
            self.input_shape,
            self.n_conv_blocks,
            self.embed_dim,
            self.num_filters,
            self.n_linear,
        )

        self.embedding_network.to(self.device)  # Move model to GPU

        # Convert Numpy arrays to PyTorch tensors and move them to GPU
        anchors = torch.tensor(anchors, dtype=torch.float32).to(self.device)
        positives = torch.tensor(positives, dtype=torch.float32).to(self.device)
        negatives = torch.tensor(negatives, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

        # Create a DataLoader
        dataset = TensorDataset(anchors, positives, negatives, labels)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )

        # Training loop
        self.history = {"loss": []}
        print("Beginning training loop...")
        for epoch in range(self.epochs):
            print(f"Beginning epoch {epoch}...")
            for batch_anchors, batch_positives, batch_negatives, _ in dataloader:
                optimizer.zero_grad()
                anchor_embedding = self.embedding_network(batch_anchors)
                positive_embedding = self.embedding_network(batch_positives)
                negative_embedding = self.embedding_network(batch_negatives)
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

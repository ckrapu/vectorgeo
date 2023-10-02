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

from vectorgeo.models import initialize_triplet, triplet_loss, early_stopping
from vectorgeo.raster import unpack_array, extend_negatives
from vectorgeo import transfer
from vectorgeo import constants as c

import torch
from torch.utils.data import DataLoader, IterableDataset

class HDF5Dataset(IterableDataset):
    def __init__(self, hdf5_filepath, batch_size):
        self.hdf5_filepath = hdf5_filepath
        self.batch_size = batch_size

    def __iter__(self):
        with h5py.File(self.hdf5_filepath, 'r') as f:
            n_samples = len(f['train_data'])
            for i in range(0, n_samples, self.batch_size):
                yield torch.tensor(f['train_data'][i:i+self.batch_size], dtype=torch.float32)

    def __len__(self):
        with h5py.File(self.hdf5_filepath, 'r') as f:
            return len(f['train_data'])
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as f:
            return torch.tensor(f['train_data'][idx], dtype=torch.float32)


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
        "num_filters", help="Number of filters in each convolutional layer", default=128
    )

    num_dense = Parameter(
        "num_dense", help="Number of units in the final dense layers", default=64
    )

    use_dem = Parameter(
        "use_dem", help="Whether to use the DEM data in the training", default=True
    )
    
    overwrite_hdf5 = Parameter(
        "overwrite_hdf5",
        help="Whether to overwrite the HDF5 file if it already exists",
        default=True,
    )

    n_conv_blocks = Parameter(
        "n_conv_blocks",
        help="Number of non-downsample convolutional ResNet blocks per downsample in the embedding network",
        default=2,
    )

    verbose = Parameter(
        "verbose", help="Whether to print out verbose information", default=False
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

    test_fraction = Parameter(
        "test_fraction", help="Fraction of data to use for early stopping",
        default=0.01)
    
    checkpoint_interval = Parameter(
        "checkpoint_interval", help="Number of epochs between checkpoints",
        default=10)

    @step
    def start(self):
        """
        Loads the training files from S3 and starts the training process.
        """

        self.model_path = os.path.join(c.TMP_DIR, self.model_filename)

        # Get list of files in the S3 bucket
        keys = transfer.ls_s3("train/")
        keys = list(filter(lambda x: x.endswith(".npy"), keys))
        print("Found training data files {} files".format(len(keys)))

        if self.use_dem:
            self.image_shape = (3, 24, self.image_size, self.image_size,)
        else:
            self.image_shape = (3, 23, self.image_size, self.image_size,)
                
        n_loaded = 0       
        self.hdf5_filepath = os.path.join(c.TMP_DIR, 'train.h5') 

        test_arrays = []
        
        if os.path.exists(self.hdf5_filepath) and self.overwrite_hdf5:
            print(f"Removing existing HDF5 file {self.hdf5_filepath}")
            os.remove(self.hdf5_filepath)
            
            print(f"Loading data from {len(keys)} files into HDF5 store")
            with h5py.File(self.hdf5_filepath, 'w') as f:

                # Create an expandable dataset in the HDF5 file
                dset      = f.create_dataset("train_data", shape=(0,) + self.image_shape, maxshape=(None,) + self.image_shape, dtype='f4')
                dset_test = f.create_dataset("test_data", shape=(0,) + self.image_shape, maxshape=(None,) + self.image_shape, dtype='f4')

                for key in keys:
                    print(f"Inserting data from file {key} into HDF5 store")
                    local_filepath = os.path.join(c.TMP_DIR, os.path.basename(key))
                    transfer.download_file(key, local_filepath)
                    arr = np.load(local_filepath)

                    if self.verbose: print(f"Loaded array of shape {arr.shape} and dtype {arr.dtype} from {key}")

                    # Your preprocessing logic here
                    xs_combined = unpack_array(arr[:, [0]])

                    if self.use_dem:
                        xs_dem = arr[:, 1]
                        xs_dem = extend_negatives(xs_dem)
                        xs_dem = np.transpose(xs_dem, (0, 3, 1, 2))[:, :, np.newaxis]
                        xs_combined = np.concatenate([xs_combined, xs_dem], axis=2)

                    # Create random variable to determine whether to add to train data or keep as test data
                    is_test = np.random.uniform(size=xs_combined.shape[0]) < self.test_fraction

                    xs_combined_test = xs_combined[is_test]
                    xs_combined_train = xs_combined[~is_test]

                    test_arrays.append(xs_combined_test)

                    # Incrementally save to HDF5
                    new_shape = (dset.shape[0] + xs_combined_train.shape[0],) + self.image_shape
                    dset.resize(new_shape)
                    dset[-xs_combined_train.shape[0]:] = xs_combined_train

                    new_shape_test = (dset_test.shape[0] + xs_combined_test.shape[0],) + self.image_shape
                    dset_test.resize(new_shape_test)
                    dset_test[-xs_combined_test.shape[0]:] = xs_combined_test
                    
                    n_loaded += len(xs_combined_train)

                    if n_loaded > self.n_train:
                        print(f"Loaded {n_loaded} samples; stopping")
                        break
        
        else:
            # Load the xs_combined as the first <batch_size> samples
            # from the hdf5 file
            with h5py.File(self.hdf5_filepath, 'r') as f:
                xs_combined = f['train_data'][:self.batch_size]

                # Get n_loaded as length of dataset
                n_loaded = len(f['train_data'])

        # As a check, print out statistics on most recent batch
        # to make sure that the data is being loaded correctly
        train_max, train_min = xs_combined.max(), xs_combined.min()
        print(f"Loaded samples; test batch has max {train_max:.2f} and min {train_min:.2f}")
        
        print(f"Loading phase finished with {n_loaded} data samples loaded!")
        self.input_shape = self.image_shape[1:]
        
        print("Initializing triplet model...")
        self.model, optimizer = initialize_triplet(
            self.input_shape,
            self.n_conv_blocks,
            self.embed_dim,
            self.num_filters,
            self.num_dense,
        )

        # Apply initialization to all parameters
        print("Applying glorot initialization to all parameters...")
        for param in self.model.parameters():
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

        # Printout for total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")

        self.model.to(self.device) 
        
        dataset    = HDF5Dataset(self.hdf5_filepath, self.batch_size)
        dataloader = DataLoader(dataset, batch_size=None)

        # Load entire test dataset into memory
        with h5py.File(self.hdf5_filepath, 'r') as f:
            self.test_data = f['test_data'][:]

        test_loss_history = []

        # Training loop
        self.history = {"loss": []}
        print("Beginning training loop...")
        try:
            for epoch in range(self.epochs):

                if epoch % self.checkpoint_interval == 0:
                    print(f"Saving checkpoint at epoch {epoch}...")
                    torch.save(self.model, self.model_path)                

                # Evaluate the model loss on the test data
                with torch.no_grad():
                    test_data = torch.tensor(self.test_data, dtype=torch.float32)

                    # Split into batches and run through the network
                    test_losses = []
                    for tbatch in torch.split(test_data, self.batch_size):

                        tbatch = tbatch.to(self.device)

                        anchor_zs   = self.model(tbatch[:,0])
                        positive_zs = self.model(tbatch[:,1])
                        negative_zs = self.model(tbatch[:,2])

                        merged_vector = torch.cat([anchor_zs, positive_zs, negative_zs], dim=1)
                        test_loss, _, _ = triplet_loss(merged_vector)
                        test_losses.append(test_loss.item())
                    
                    test_loss_full = np.mean(test_losses)

                    del merged_vector
                    del anchor_zs
                    del positive_zs
                    del negative_zs
                    del tbatch

                    print(f"(Epoch {epoch}) - Test loss: {test_loss_full:.3f}")
                    test_loss_history += [test_loss_full]
                
                # Execute the actual training loop
                for i, batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    batch = batch.to(self.device)

                    anchor_zs   = self.model(batch[:,0])
                    positive_zs = self.model(batch[:,1])
                    negative_zs = self.model(batch[:,2])

                    merged_vector = torch.cat([anchor_zs, positive_zs, negative_zs], dim=1)
                    loss, pos_dist, neg_dist = triplet_loss(merged_vector)
                    
                    if self.verbose:
                        print(f"(Batch {i}, epoch {epoch}) Loss: {loss.item():.2f}, pos_dist: {pos_dist.mean().item():.2f}, neg_dist: {neg_dist.mean().item():.2f}")

                    # Clip the gradient values to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    loss.backward()
                    optimizer.step()
                    
                    self.history["loss"].append(loss.item())

                zs = merged_vector.cpu().detach().numpy()

                print(f"First 5 x 4 block of zs: {zs[:5, :4]}")
                print(f"Finished epoch {epoch}! Loss: {loss.item():.2f}, pos_dist: {pos_dist.mean().item():.2f}, neg_dist: {neg_dist.mean().item():.2f}; Average std. of first 3 dims: {(np.var(zs[:, :3], axis=0)**0.5).mean():.2f}")

                # Check for convergence in test loss
                is_stopped = False if epoch < 10 else early_stopping(test_loss_history, patience=3, tolerance=0.1)

                if is_stopped:
                    print(f"Stopping early at epoch {epoch}! Test loss average over last 5 epochs: {np.mean(test_loss_history[-5:]):.2f}")
                    break
                else:
                    print(f"Test loss average over last 5 epochs: {np.mean(test_loss_history[-5:]):.2f}")
        
        except KeyboardInterrupt:
            print("Keyboard interrupt detected; stopping training early")

        # Save these off for later analyses
        self.test_batch = anchor_zs.cpu().detach().numpy()

        # This number helps tell us if the model is producing degenerate outputs
        print(f"Average L2 distance: {np.linalg.norm(self.test_batch[1:] - self.test_batch[:-1], axis=1).mean():.2f}")
        self.next(self.end)

    @step
    def end(self):
        """
        Generate statistics on the number of active dimensions after PCA & upload to S3.
        """

        # Save the model to a temporary directory
        torch.save(self.model, self.model_path)

        # Upload the model to S3
        transfer.upload_file(f"models/{self.model_filename}", self.model_path)

        # Transform with PCA
        pca = PCA(n_components=self.embed_dim)
        zs_pca = pca.fit_transform(self.test_batch)

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
        ax0.imshow(self.test_batch[0:64])
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

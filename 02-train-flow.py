"""MetaFlow flow for identifying dominant geographical units for each user on the basis of their viewing history.

Tested resolutions include geo units and neighborhoods.
"""

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
from umap import UMAP

from vectorgeo.models import initialize_triplet
from vectorgeo.landcover import unpack_array
from vectorgeo import data_utils
from vectorgeo import constants as c

class TrainLandCoverTripletFlow(FlowSpec):
    """
    Flow for producing training dataset of paired anchor/neighbor land cover images 
    from Copernicus LULC data. The end result is a sequence of .npy files on the targeted
    S3 bucket, each of which contains matched anchor/neighbor pairs of land use / land cover
    data encoded as integers and with the resolution specified by the `patch_size` parameter.
    """

    epochs = Parameter(
        'epochs',
        help='Number of epochs to train for',
        default=10)
    
    batch_size = Parameter(
        'batch_size',
        help='Batch size for training',
        default=32)
    
    embed_dim = Parameter(
        'embed_dim',
        help='Dimension of the embedding space',
        default=16)
    
    num_filters = Parameter(
        'num_filters',
        help='Number of filters in each convolutional layer',
        default=32)
    
    n_linear = Parameter(
        'n_linear',
        help='Number of units in the final dense layers',
        default=64)
    
    n_conv_blocks = Parameter(
        'n_conv_blocks',
        help='Number of non-downsample convolutional ResNet blocks per downsample in the embedding network',
        default=2)
    
    n_train_files = Parameter(
        'n_train_files',
        help='Number of files to use for training',
        default=10)
    
    model_filename = Parameter(
        'model_filename',
        help='Filename to save the model to',
        default= "resnet-triplet-lc.keras"
    )

    @step
    def start(self):
        """
        Loads the training files from S3 and starts the training process.
        """

        # Get list of files in the S3 bucket
        keys = data_utils.ls_s3('landcover/')
        keys = list(filter(lambda x: x.endswith('.npy'), keys))
        print('Found {} files'.format(len(keys)))


        keys = keys[0:self.n_train_files]
        print(f"Preparing to read {len(keys)} files")

        arrays = []
        for key in keys:
            local_filepath = os.path.join(c.TMP_DIR, os.path.basename(key))
            data_utils.download_file(key, local_filepath)
            # Read each file in the list and append it to the arrays list
            print('....Reading {}'.format(key))
            arr = np.load(local_filepath)

            print(f"Found {np.sum(np.isnan(arr))} NaNs in array")
            arrays += [arr]

        # Convert from integer to one-hot encoding
        # We don't preprocess as one-hot because the storage is way larger.
        print(f"Unpacking {len(arrays)} arrays...")
        xs_one_hot = np.concatenate([unpack_array(xs) for xs in arrays], axis=0)

        self.input_shape = xs_one_hot.shape[2:]
        self.anchors, self.positives, self.negatives = xs_one_hot[:, 0], xs_one_hot[:, 1], xs_one_hot[:, 2]
        self.labels = np.zeros((len(self.anchors), 1))
        print(f"Loaded {len(arrays)} files; resulting stacked array has shape {xs_one_hot.shape}")

        # Save off a select group of images
        # to use for downstream information content
        # assessment with PCA
        self.test_batch = xs_one_hot[0:1024, 0]
        self.next(self.train_embedding_model)

    @step
    def train_embedding_model(self):
        """
        Run the sampler in parallel across different jobs to iteratively samples of neighboring
        land cover patches and store them as Numpy arrays on S3.
        """

        self.triplet_model, self.embedding_network = initialize_triplet(
            self.input_shape,
            self.n_conv_blocks,
            self.embed_dim,
            self.num_filters,
            self.n_linear
        )

        self.history = self.triplet_model.fit(
            [self.anchors, self.positives, self.negatives],
            self.labels, epochs=self.epochs, batch_size=self.batch_size)
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Generate statistics on the number of active dimensions after PCA & upload to S3.

        """

        # Save the model to S3
        model_path = os.path.join(c.TMP_DIR, self.model_filename)
        self.embedding_network.save(model_path) 
        data_utils.upload_file(f"models/{self.model_filename}", model_path)
        
        zs = self.embedding_network(self.test_batch).numpy()

        # transform with pca
        pca = PCA(n_components=self.embed_dim)

        zs_pca = pca.fit_transform(zs)

        # Print out the number of eigenvalues
        # needed to reach threshold of variance explained
        variance_threshold = 0.99
        variance_explained = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(variance_explained > variance_threshold)[0][0]
        print(f"Number of components needed to reach {variance_threshold} variance explained: {n_components}")

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

        # Plot of UMAP projection of embeddings
        reducer = UMAP()
        ws = reducer.fit_transform(zs)
        hb = ax2.hexbin(ws[:, 0], ws[:, 1], cmap='viridis', bins='log')
        ax2.set_title("Hexbin log density for\nUMAP projection of embeddings")
        plt.colorbar(hb, ax=ax2, orientation='vertical')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.savefig("figures/embeddings.png")

        

if __name__ == "__main__":
    TrainLandCoverTripletFlow()

"""MetaFlow flow for identifying dominant geographical units for each user on the basis of their viewing history.

Tested resolutions include geo units and neighborhoods.
"""

from metaflow import (
    FlowSpec,
    Parameter,  # pylint: disable=no-name-in-module
    step,
)

import os
import constants as c
import yaml
import numpy as np
import s3fs
import boto3
import matplotlib.pyplot as plt

from matplotlib import gridspec
from sklearn.decomposition import PCA
from models import initialize_triplet
from umap import UMAP
from tqdm import tqdm

from landcover import unpack_array




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

        with open('.secrets.yml', 'r') as f:
            secrets = yaml.safe_load(f)

        # Initialize s3fs using aws_aceess_key_id and aws_secret_access_key
        fs = s3fs.S3FileSystem(
            key=secrets['aws_access_key_id'],
            secret=secrets['aws_secret_access_key'],
            client_kwargs={'region_name': c.S3_REGION}
        )

        s3_client = boto3.client(
            's3',
            aws_access_key_id=secrets['aws_access_key_id'],
            aws_secret_access_key=secrets['aws_secret_access_key'],
            region_name=c.S3_REGION
        )


        # Read all files in the bucket c.S3_BUCKET and key 'landcover' with file extension .npy
        # and store them in a list
        files = fs.ls(os.path.join(c.S3_BUCKET, 'landcover'))
        files = [f for f in files if f.endswith('.npy')]
        print('Found {} files'.format(len(files)))

        arrays = []

        files_to_read = files[0:self.n_train_files]
        print(f"Preparing to read {len(files_to_read)} files")

        for f in files_to_read:
            # Read each file in the list and append it to the arrays list
            print('....Reading {}'.format(f))
            arrays.append(np.load(fs.open(f)))


        # Convert from integer to one-hot encoding
        # We don't preprocess as one-hot because the storage is way larger.
        xs_one_hot = np.concatenate([unpack_array(xs) for xs in tqdm(arrays)], axis=0)

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

        secrets = yaml.load(open(os.path.join(c.BASE_DIR, '.secrets.yml')), Loader=yaml.FullLoader)

        s3_client = boto3.client(
            's3',
            aws_access_key_id=secrets['aws_access_key_id'],
            aws_secret_access_key=secrets['aws_secret_access_key'],
            region_name=c.S3_REGION
        )
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

        # Save the model to S3
        model_path = os.path.join('temp', self.model_filename)
        self.embedding_network.save(model_path) 
        s3_client.upload_file(model_path, c.S3_BUCKET, f"models/{self.model_filename}")
        os.remove(model_path)

if __name__ == "__main__":
    TrainLandCoverTripletFlow()

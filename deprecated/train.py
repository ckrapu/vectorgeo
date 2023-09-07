# %%
import boto3
import yaml
import matplotlib.pyplot as plt
import constants as c
import numpy as np
import os
import json
import io
import keras
import tensorflow as tf
import fire

from models import ResnetCAE
from s3fs.core import S3FileSystem
from tensorflow.keras import layers, Model
from datetime import datetimeimport os
import io
import json
import yaml
import boto3
import numpy as np
import matplotlib.pyplot as plt
import fire
import keras
import tensorflow as tf

from datetime import datetime
from s3fs.core import S3FileSystem
from tensorflow.keras import layers, Model
from models import ResnetCAE
import constants as c

class Trainer:
    def __init__(self):
        with open('secrets.yml', 'r') as f:
            self.secrets = yaml.safe_load(f)
        self.s3_boto = boto3.client(
            's3', 
            aws_access_key_id=self.secrets['aws_access_key_id'], 
            aws_secret_access_key=self.secrets['aws_secret_access_key']
        )
        self.bucket = boto3.resource(
            's3', 
            aws_access_key_id=self.secrets['aws_access_key_id'], 
            aws_secret_access_key=self.secrets['aws_secret_access_key']
        ).Bucket(c.S3_BUCKET)
        self.s3fs = S3FileSystem(
            key=self.secrets['aws_access_key_id'], 
            secret=self.secrets['aws_secret_access_key']
        )

    def read_s3_np(self, key):
        return np.load(self.s3fs.open('{}/{}'.format(c.S3_BUCKET, key)))

    def classwise_tjur_r2(self, model, x_true):
        """Evaluate the Tjur R^2 for each unique outcome in x_true's last axis."""
        x_pred = model.predict(x_true)
        x_true_int = np.argmax(x_true, axis=-1)
        return {
            i: (x_true[x_true_int == i] * x_pred[x_true_int == i]).mean() - 
               ((1 - x_true[x_true_int == i]) * x_pred[x_true_int == i]).mean()
            for i in range(c.LC_N_CLASSES)
        }

    def plot_reconstruction(self, x_true, x_pred, model_id, n_plots=6):
        """Make side-by-side plots of original and reconstructed data."""
        fig, axes = plt.subplots(n_plots, 2, figsize=(10, 5*n_plots))
        for i in range(n_plots):
            axes[i, 0].imshow(np.argmax(x_true[i], axis=-1), cmap='tab20b')
            axes[i, 1].imshow(np.argmax(x_pred[i], axis=-1), cmap='tab20b')
            axes[i, 0].axis('off'), axes[i, 1].axis('off')

        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        self.bucket.put_object(
            Body=img_data, 
            ContentType='image/png', 
            Key=f'figures/reconstructions_{model_id}.png'
        )
        plt.close()

    def train(self, config_path=os.path.join(c.CONFIGS_DIR, 'resnet.json'), model_path=None):
        """Train the model and save it with metadata to S3."""
        with open(config_path, 'r') as f:
            config = json.load(f)

def train(config_path=os.path.join(c.CONFIGS_DIR, 'resnet.json'), model_path=None):
    '''
    Loads configuration file from disk and spins up a Keras model, 
    before training it and saving the model + metadata to S3.
    '''
    with open('secrets.yml', 'r') as f:
        secrets = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_shape     = config['input_shape']
    K               = config['K']
    z_dim           = config['z_dim']
    num_filters     = config['num_filters']
    n_linear        = config['n_linear']
    epochs          = config['epochs']
    batch_size      = config['batch_size']
    convert_one_hot = config['convert_one_hot']
    max_n_train     = config['max_n_train']

    s3_boto = boto3.client('s3', aws_access_key_id=secrets['aws_access_key_id'], aws_secret_access_key=secrets['aws_secret_access_key'])
    bucket  = boto3.resource('s3', aws_access_key_id=secrets['aws_access_key_id'], aws_secret_access_key=secrets['aws_secret_access_key']).Bucket(c.S3_BUCKET)
    s3fs    = S3FileSystem(key=secrets['aws_access_key_id'], secret=secrets['aws_secret_access_key'])

    read_s3_np = lambda key: np.load(s3fs.open('{}/{}'.format(c.S3_BUCKET, key)))
    
    training_keys = [d['Key'] for d in s3_boto.list_objects(Bucket=c.S3_BUCKET, Prefix=c.LC_VPATH)['Contents'] if d['Key'].endswith('.npy')]

    arrays = []
    n_rows_loaded = 0

    while n_rows_loaded < max_n_train:
        arrays.append(read_s3_np(training_keys.pop()))
        n_rows_loaded += arrays[-1].shape[0]

    train_data_int = np.concatenate(arrays)
    print(f"Loaded {len(arrays)} files; resulting stacked array has shape {train_data_int.shape}")

    # Convert to one-hot
    # We don't preprocess as one-hot because the storage is way larger. 
    if convert_one_hot:
        N, C, H, W = train_data_int.shape
        train_data = np.zeros((N, c.LC_N_CLASSES, H, W))

        for i in range(c.LC_N_CLASSES):
            train_data[:, i, :, :] = (train_data_int == i).squeeze().astype(int)

    # Swap second and fourth axis for Keras
    # compatibility
    if train_data.shape[-1] != c.LC_N_CLASSES:
        train_data = np.swapaxes(train_data, 1, 3)

    inputs      = layers.Input(shape=input_shape)
    autoencoder = ResnetCAE(input_shape, K, z_dim, num_filters, n_linear)
    outputs     = autoencoder(inputs)

    # If a model is saved either locally or on s3,
    # we can attempt to load it.
    if model_path:
        try:
            base_filename = os.path.basename(model_path)
            local_path    = os.path.join(c.BASE_DIR,"tmp", base_filename)
            s3_boto.download_file(c.S3_BUCKET, model_path, local_path)
        except:
            print(f"Could not find model at {model_path} on s3. Trying to load from local disk...")

        model = keras.models.load_model(model_path)
    else:
        model = Model(inputs, outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    timestamp_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_id            = f"{timestamp_formatted}-{model.count_params()}p"

    model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)

    model_filename = f"{model_id}.h5"
    model.save(model_filename)  

    s3_boto.upload_file(Filename=model_filename,
                    Bucket=c.S3_BUCKET,
                    Key=f"models/{model_id}.h5")
    
    os.remove(model_filename)

    x_true = train_data[0:200]

    metadata = config.copy()
    metadata.update(
        {
            "summary"       : model.summary(),
            "loss"          : "categorical_crossentropy",
            "model_id"      : model_id,
            "tjur_r2_class" : classwise_tjur_r2(model, x_true),
        }
    )

    metadata_key = f"models/metadata/metadata-{model_id}.json"
    bucket.put_object(Key=metadata_key, Body=json.dumps(metadata))

    plot_reconstruction(x_true, model(x_true), bucket, model_id)

if __name__ == '__main__':
  fire.Fire()
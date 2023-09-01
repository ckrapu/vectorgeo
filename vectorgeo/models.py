from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf

class ResnetCAE(Model):
    """
    Convolutional autoencoder with resnet blocks.
    """
    def __init__(self, input_shape, K, z_dim, num_filters, n_linear):
        super(ResnetCAE, self).__init__()
        
        self.encoder = self.build_encoder(input_shape, K, num_filters, z_dim, n_linear)
        self.decoder = self.build_decoder(K, num_filters, input_shape[-1], n_linear)

    def build_encoder(self, input_shape, K, num_filters, z_dim, n_linear):
        encoder_layers = []
        
        for _ in range(3):
            encoder_layers.append(layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same"))
            for _ in range(K):
                encoder_layers.extend(self.conv_res_block(num_filters, (3, 3)))
            encoder_layers.append(layers.MaxPooling2D((2, 2), padding="same"))
        
        encoder_layers.append(layers.Flatten())
        encoder_layers.append(layers.Dense(n_linear, activation="relu"))
        encoder_layers.append(layers.Dense(z_dim, activation="relu"))
        
        return Sequential(encoder_layers)

    def build_decoder(self, K, num_filters, output_channels, n_linear):
        decoder_layers = []
        assert n_linear % 16 == 0, "n_linear must be divisible by 16"

        decoder_layers.append(layers.Dense(n_linear, activation="relu"))
        decoder_layers.append(layers.Reshape((4, 4, n_linear // 16)))
        
        for _ in range(3):
            decoder_layers.append(layers.Conv2DTranspose(num_filters, (3, 3), strides=2, activation="relu", padding="same"))
            for _ in range(K):
                decoder_layers.extend(self.conv_res_block(num_filters, (3, 3)))
        
        decoder_layers.append(layers.Conv2D(output_channels, (3, 3), padding="same"))
        decoder_layers.append(layers.Softmax(axis=-1))
        
        return Sequential(decoder_layers)

    def conv_res_block(self, filters, kernel_size, **kwargs):
        # Define a residual block with Batch Normalization and an identity addition
        
        identity = layers.Input(shape=(None, None, filters)) # Add a dummy input for the identity connection
        
        x = layers.Conv2D(filters, kernel_size, **kwargs, activation="relu", padding="same")(identity)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, **kwargs, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([identity, x])
        
        model = Model(inputs=identity, outputs=x)  # Wrap the block in a model
        
        return [model]

    def call(self, inputs):
        z = self.encoder(inputs)
        x = self.decoder(z)
        return x
    

def dense_res_block(units, **kwargs):
    identity = layers.Input(shape=(units,))
    x = layers.Dense(units, **kwargs, activation="relu")(identity)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([identity, x])
    x = layers.Activation('relu')(x)
    
    model = Model(inputs=identity, outputs=x)
    return [model]


class ResnetTripletEmbedding(Model):
    """
    Embedding network with resnet blocks for triplet loss.
    """
    def __init__(self, input_shape, K, z_dim, num_filters, n_linear, num_dense_blocks=2):

        self.input_layer = layers.Input(shape=input_shape)
        x = self.input_layer
        
        # Extract # of downsamples based on input shape
        n_downsamples = 0
        while input_shape[0] > 4:
            input_shape = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2])
            n_downsamples += 1

        # Encoder structure similar to autoencoder
        for _ in range(n_downsamples):
            x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
            for _ in range(K):
                x = self.conv_res_block(x, num_filters, (3, 3))
            x = layers.MaxPooling2D((2, 2), padding="same")(x)

        x = layers.Flatten()(x)

        # Add single dense layer to help with mismatched shapes
        x = layers.Dense(n_linear, activation="relu")(x)
        
        # Adding dense residual blocks
        for _ in range(num_dense_blocks):
            x = self.dense_res_block(x, n_linear)
        
        self.output_layer = layers.Dense(z_dim)(x)

        super(ResnetTripletEmbedding, self).__init__(inputs=self.input_layer, outputs=self.output_layer)

    def conv_res_block(self, input_tensor, filters, kernel_size, **kwargs):
        identity = input_tensor
        x = layers.Conv2D(filters, kernel_size, **kwargs, activation="relu", padding="same")(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, **kwargs, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([identity, x])
        return x

    def dense_res_block(self, input_tensor, units, **kwargs):
        identity = input_tensor
        x = layers.Dense(units, **kwargs, activation="relu")(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units, **kwargs)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([identity, x])
        x = layers.Activation('relu')(x)
        return x

def triplet_loss(y_true, y_pred, alpha=0.4, eta=0.1):
    """
    Compute triplet loss.

    y_pred shape: (batch_size, embeddings*3)
    Contains anchor, positive, negative embeddings for each batch
    """
    total_length = y_pred.shape.as_list()[-1]
    
    anchor, positive, negative = y_pred[:, :total_length//3], y_pred[:, total_length//3:2*total_length//3], y_pred[:, 2*total_length//3:]

    # Compute pairwise distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)

    # Compute pairwise distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Add L2 regularization with strength eta
    l2_reg = tf.reduce_sum(tf.square(anchor), axis=1)
    l2_reg += tf.reduce_sum(tf.square(positive), axis=1)
    l2_reg += tf.reduce_sum(tf.square(negative), axis=1)

    # Compute triplet loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0)) + eta * tf.reduce_sum(l2_reg)

    return loss

def initialize_triplet(input_shape, n_conv_blocks, embed_dim, num_filters, n_linear):
    # Create the embedding model
    embedding_network = ResnetTripletEmbedding(
        input_shape, K=n_conv_blocks,
        z_dim=embed_dim,
        num_filters=num_filters,
        n_linear=n_linear)

    # Define triplet inputs
    anchor_input   = layers.Input(shape=input_shape, name="anchor_input")
    positive_input = layers.Input(shape=input_shape, name="positive_input")
    negative_input = layers.Input(shape=input_shape, name="negative_input")

    # Process the triplets through the embedding network
    anchor_embedding   = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)

    # Merge the triplet embeddings into one vector
    merged_vector = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)

    # Define the triplet model
    triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)

    return triplet_model, embedding_network


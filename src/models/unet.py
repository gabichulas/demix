import tensorflow as tf
from tensorflow.keras import layers

class UNet(tf.keras.Model):
    def __init__(self, filters=[16, 32, 64, 128]):
        super(UNet, self).__init__()
        self.filters = filters
        
        # Encoder
        self.enc1 = self.conv_block(filters[0], 'enc1')
        self.enc2 = self.conv_block(filters[1], 'enc2')
        self.enc3 = self.conv_block(filters[2], 'enc3')
        self.enc4 = self.conv_block(filters[3], 'enc4')
        
        self.pool = layers.MaxPooling2D(pool_size=2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(filters[3], 'bottleneck')
        
        # Decoder
        self.upconv4 = layers.Conv2DTranspose(filters[2], kernel_size=2, strides=2)
        self.dec4 = self.conv_block(filters[2], 'dec4')

        self.upconv3 = layers.Conv2DTranspose(filters[1], kernel_size=2, strides=2)
        self.dec3 = self.conv_block(filters[1], 'dec3')

        self.upconv2 = layers.Conv2DTranspose(filters[0], kernel_size=2, strides=2)
        self.dec2 = self.conv_block(filters[0], 'dec2')

        self.upconv1 = layers.Conv2DTranspose(filters[0], kernel_size=2, strides=2)
        self.dec1 = self.conv_block(filters[0], 'dec1')

        # Output
        self.output_conv = layers.Conv2D(4, kernel_size=1, activation='sigmoid')
        
        
    def conv_block(self, filters, name):
        return tf.keras.Sequential([
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization()
        ], name=name)
        
    def call(self, inputs, training=False):
        enc1 = self.enc1(inputs, training=training)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x, training=training)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x, training=training)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x, training=training)
        x = self.pool(enc4)
        
        x = self.bottleneck(x, training=training)
        
        x = self.upconv4(x)
        pad_h = tf.shape(enc4)[1] - tf.shape(x)[1]
        pad_w = tf.shape(enc4)[2] - tf.shape(x)[2]
        x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc4])
        x = self.dec4(x, training=training)
        
        x = self.upconv3(x)
        pad_h = tf.shape(enc3)[1] - tf.shape(x)[1]
        pad_w = tf.shape(enc3)[2] - tf.shape(x)[2]
        x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc3])
        x = self.dec3(x, training=training)
        
        x = self.upconv2(x)
        pad_h = tf.shape(enc2)[1] - tf.shape(x)[1]
        pad_w = tf.shape(enc2)[2] - tf.shape(x)[2]
        x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc2])
        x = self.dec2(x, training=training)
        
        x = self.upconv1(x)
        pad_h = tf.shape(enc1)[1] - tf.shape(x)[1]
        pad_w = tf.shape(enc1)[2] - tf.shape(x)[2]
        x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc1])
        x = self.dec1(x, training=training)
        
        output = self.output_conv(x)
        
        return output
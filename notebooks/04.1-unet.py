import tensorflow as tf
from tensorflow.keras import layers

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


class UNet(tf.keras.Model):
    def __init__(self, filters=[16, 32, 64, 128], output_channels=4):
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
        enc1 = self.enc1(inputs)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        x = self.bottleneck(x)
        
        # Decoder con padding para manejar dimensiones impares
        x = self.upconv4(x)
        if x.shape[1] != enc4.shape[1]:
            pad_h = enc4.shape[1] - x.shape[1]
            x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        if x.shape[2] != enc4.shape[2]:
            pad_w = enc4.shape[2] - x.shape[2]
            x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc4])
        x = self.dec4(x)
        
        x = self.upconv3(x)
        if x.shape[1] != enc3.shape[1]:
            pad_h = enc3.shape[1] - x.shape[1]
            x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        if x.shape[2] != enc3.shape[2]:
            pad_w = enc3.shape[2] - x.shape[2]
            x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc3])
        x = self.dec3(x)
        
        x = self.upconv2(x)
        if x.shape[1] != enc2.shape[1]:
            pad_h = enc2.shape[1] - x.shape[1]
            x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        if x.shape[2] != enc2.shape[2]:
            pad_w = enc2.shape[2] - x.shape[2]
            x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc2])
        x = self.dec2(x)
        
        x = self.upconv1(x)
        if x.shape[1] != enc1.shape[1]:
            pad_h = enc1.shape[1] - x.shape[1]
            x = tf.pad(x, [[0,0], [0, pad_h], [0,0], [0,0]])
        if x.shape[2] != enc1.shape[2]:
            pad_w = enc1.shape[2] - x.shape[2]
            x = tf.pad(x, [[0,0], [0,0], [0, pad_w], [0,0]])
        x = layers.Concatenate()([x, enc1])
        x = self.dec1(x)
        
        output = self.output_conv(x)
        
        return output

model = UNet(filters=[8, 16, 32, 64])

dummy = tf.random.normal((1, 1025, 128, 1))

output = model(dummy)

print("Model is working!")
print(f"Input shape: {dummy.shape}")
print(f"Output shape: {output.shape}")

model.summary()

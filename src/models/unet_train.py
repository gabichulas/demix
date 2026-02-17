from unet import UNet
import tensorflow as tf
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import create_dataset

model = UNet(filters=[8, 16, 32, 64])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

print("// Configuring model...")

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print("// Creating datasets...")

train_dataset = create_dataset(Path('data/processed/train'), chunk_size=128, batch_size=4)
test_dataset = create_dataset(Path('data/processed/test'), chunk_size=128, batch_size=4)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'models/checkpoints/unet_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )
]

print("// Starting training...")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=callbacks
)

print("// Training finished")
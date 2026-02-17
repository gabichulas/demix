import tensorflow as tf
import h5py
import numpy as np
from pathlib import Path

def load_track(file_path, chunk_size=128):
    with h5py.File(file_path, 'r') as f:
        mix = f['mix'][:]
        drums = f['drums'][:]
        bass = f['bass'][:]
        other = f['other'][:]
        vocals = f['vocals'][:]
    
    n_frames = mix.shape[1]
    
    chunks_X = []
    chunks_y = []
    
    for start in range(0, n_frames, chunk_size):
        end = start + chunk_size
        if end > n_frames:
            break
        
        chunk_mix = mix[:, start:end]
        chunk_drums = drums[:, start:end]
        chunk_bass = bass[:, start:end]
        chunk_other = other[:, start:end]
        chunk_vocals = vocals[:, start:end]
        
        
        chunk_stems = np.stack([chunk_drums, chunk_bass, chunk_other, chunk_vocals], axis=-1)
        chunk_mix_expanded = np.expand_dims(chunk_mix, axis=-1)
        epsilon = 1e-10
        masks = chunk_stems / (chunk_mix_expanded + epsilon)
        masks = np.clip(masks, 0.0, 1.0)
        
        chunks_X.append(chunk_mix_expanded)
        chunks_y.append(masks)
        
    
    return chunks_X, chunks_y

def create_dataset(data_path: Path, chunk_size=128, batch_size=8):
    files = list(Path(data_path).glob('*.h5'))
    
    def generator():
        for file in files:
            chunks_X, chunks_y = load_track(file, chunk_size)
            for X, y in zip(chunks_X, chunks_y):
                yield X, y
                
    def normalize_example(x, y): 
        x_max = tf.maximum(tf.reduce_max(tf.abs(x)), 1e-10)
        x_norm = x / x_max
        return x_norm, y
    
    dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=(
        tf.TensorSpec(shape=(None, chunk_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, chunk_size, 4), dtype=tf.float32)
    ))
    
    dataset = dataset.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(normalize_example)
    dataset = dataset.shuffle(buffer_size=50)

    return dataset
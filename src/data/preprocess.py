import stempeg
import librosa
import numpy as np
import h5py
from pathlib import Path

def generate_stft(audio):
    spectrograms = []
    for i in range(5):
        mono = audio[i, :, :].mean(axis=1)
        D = librosa.stft(mono)
        magnitude = np.abs(D)
        spectrograms.append(magnitude)
    return spectrograms

def preprocess_dataset(path: Path, output_path: Path):
    tracks = sorted(list(path.glob('*.stem.mp4')))
    
    print("Generating spectrograms for every stem...")
    for track in tracks:
        track_name = track.stem.replace('.stem', '')
        h5_file = output_path / f"{track_name}.h5"
        
        if h5_file.exists(): 
            print(f"Track {track_name} already processed. Skipping.")
            continue
        
        audio, _ = stempeg.read_stems(str(track))
        spectrograms = generate_stft(audio)
        
        # Save all stems in one HDF5 file with compression
        stem_names = ['mix', 'drums', 'bass', 'other', 'vocals']
        with h5py.File(h5_file, 'w') as f:
            for i, name in enumerate(stem_names):
                f.create_dataset(name, data=spectrograms[i], compression='gzip', compression_opts=4)
        
        print(f"Processed: {track_name}")
    
    print(f"{len(tracks)} tracks processed successfully.")
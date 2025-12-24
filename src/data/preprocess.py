import stempeg
import librosa
import numpy as np
from pathlib import Path

def generate_stft(audio):
    spectrograms = []
    for i in range(5):
        mono = audio[i, :, :].mean(axis=1)
        D = librosa.stft(mono)
        magnitude = np.abs(D)
        spectrograms.append(magnitude)
    return np.array(spectrograms)

def preprocess_dataset(path: Path, output_path: Path = ""):
    tracks = sorted(list(path.glob('*.stem.mp4')))
    print("Generating spectrograms for every stem...")
    for track in tracks:
        track_name = track.stem.replace('.stem', '')
        track_dir = output_path / track_name
        if track_dir.exists(): 
            print(f"Track {track_name} already processed. Skipping.")
            continue
        
        audio, _ = stempeg.read_stems(str(track))
        spectrograms = generate_stft(audio)
        track_dir.mkdir(exist_ok=True)
        
        stem_names = ['mix', 'drums', 'bass', 'other', 'vocals']
        for i, name in enumerate(stem_names):
            np.save(track_dir / f"{name}.npy", spectrograms[i])
    
    print(f"{len(tracks)} tracks processed succesfully.")
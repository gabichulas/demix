import musdb

print("Downloading MUSDB18...")
mus = musdb.DB(root='data/raw/musdb18', download=True)
print(f"Dataset successfully downloaded to: data/raw/musdb18")

print("\nLoading tracks...")
train_tracks = mus.load_mus_tracks(subsets=['train'])
test_tracks = mus.load_mus_tracks(subsets=['test'])

print(f"\nTraining tracks: {len(train_tracks)}")
print(f"Test tracks: {len(test_tracks)}")
print(f"Total tracks: {len(train_tracks) + len(test_tracks)}")
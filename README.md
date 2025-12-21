# Demix

Comparing U-Net and Vision Transformers for audio source separation.

![Logo del proyecto](img/logo.png)

---

## About

U-Nets have been the go-to architecture for audio source separation (ASS) for years. They're good at it because the task is similar to their original purpose: image segmentation. In both cases, you're generating masks to isolate specific elements—either organs in medical images or instruments in spectrograms.

This project questions whether that dominance is justified. **Vision Transformers (ViT)** have taken over computer vision by using self-attention to capture global context. The hypothesis here is simple: if music has long-range patterns (like a drum loop that repeats throughout a song), Transformers should handle them better than CNNs with their limited receptive fields.

## The Goal

Train both architectures on the same data and compare their performance using Signal-to-Distortion Ratio (SDR). 

- **Input**: Time-frequency representation of a song (spectrogram)
- **Output**: 4 separation masks, one for each source (vocals, drums, bass, other)
- **Dataset**: MUSDB18

Both models will generate masks that segment the spectrogram into isolated sources. The masks are then used to reconstruct the individual audio signals.

## Why it matters

If ViT outperforms U-Net, it suggests that global attention is more important than local convolutions for understanding musical structure. That opens doors to better separation models and potentially transfer learning from pretrained vision models.

If U-Net wins, we confirm that convolutional inductive biases still have value and aren't going away anytime soon.

---

## Structure

```
demix/
├── data/
│   ├── raw/              # MUSDB18 dataset
│   ├── processed/        # preprocessed spectrograms
│   └── temp/             # temp files for API
│
├── src/
│   ├── models/           # U-Net and ViT implementations
│   ├── data/             # data loading and preprocessing
│   └── api/              # FastAPI server for inference
│
├── notebooks/            # experimentation
├── docker/               # containerization
├── checkpoints/          # saved model weights
└── results/              # metrics and plots
```

---

## Tech

- TensorFlow 2.10
- librosa for audio
- FastAPI for serving
- Docker for deployment

---

**Building with curiosity and a GTX 1050 Ti** 


import torch
import librosa
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32000, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 32000),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def get_gan_model():
    model = Generator()
    # Load pretrained weights here if needed
    return model.to('cpu')

def enhance_audio_with_gan(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    model = get_gan_model()
    model.eval()

    segments = librosa.util.frame(y, frame_length=sr*2, hop_length=sr*2).T
    enhanced_segments = []

    with torch.no_grad():
        for segment in segments:
            noisy = torch.tensor(segment + 0.05 * np.random.randn(*segment.shape), dtype=torch.float32).view(1, -1)
            enhanced = model(noisy).view(-1).numpy()
            enhanced_segments.append(enhanced)

    return np.concatenate(enhanced_segments)

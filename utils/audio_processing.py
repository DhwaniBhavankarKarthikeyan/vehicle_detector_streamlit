import librosa
import numpy as np
import moviepy.editor as mp
import soundfile as sf

def extract_audio(video_path, output_path="audio.wav", sr=16000):
    clip = mp.VideoFileClip(video_path)
    audio_path = output_path
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def compute_energy_envelope(audio, sr=16000):
    frames = librosa.util.frame(audio, frame_length=1024, hop_length=512)
    energy = np.sum(frames ** 2, axis=0)
    threshold = np.percentile(energy, 95)
    suspicious_frames = np.where(energy > threshold)[0]
    suspicious_times = suspicious_frames * 512 / sr
    return energy, suspicious_times

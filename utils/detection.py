def detect_vehicle_times(energy, sr=16000):
    threshold = np.percentile(energy, 95)
    suspicious_frames = np.where(energy > threshold)[0]
    return suspicious_frames * 512 / sr

import librosa
import soundfile as sf

# Load a test audio file
y, sr = librosa.load('audios/muthai_tharu.wav')

# Original duration
original_duration = librosa.get_duration(y=y, sr=sr)
print(f"Original Duration: {original_duration:.2f} seconds")

# Apply time-stretching
stretch_factor = 2.0
rate = 1 / stretch_factor  # 0.5 to slow down by half
y_stretched = librosa.effects.time_stretch(y=y, rate=rate)

# Stretched duration
stretched_duration = librosa.get_duration(y=y_stretched, sr=sr)
print(f"Stretched Duration: {stretched_duration:.2f} seconds")

# Save stretched audio for manual verification
sf.write('stretched_audio.wav', y_stretched, sr)

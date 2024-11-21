# %%
from utils import Utils
import scipy
import librosa.display
from scipy.stats import mode
import tensorflow as tf
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import sounddevice as sd

# %%
# Expanded Kattai Frequencies across multiple octaves
kattai_freqs = {
    "0_75": 246.94,
    "1": 261.63,
    "1_5": 277.18,
    '2': 293.66,
    '2_5': 311.13,
    '3': 329.63,
    '4': 349.23,
    '4_5': 369.99,
    '5': 392.0,
    '5_5': 415.3,
    '6': 440.0,
    '6_5': 466.16,
    '7': 493.88
}

# Generate frequencies across multiple octaves
OCTAVES = [0.5, 1.0, 2.0]  # Lower, Middle, Upper

SWARA_CENTS = {
    'S': 0,
    'R2': 112,    # Chathushruthi Rishabha
    'G2': 294,    # Sadharana Gandhara
    'M2': 408,    # Prati Madhyama
    'P': 498,     # Panchama
    'D1': 588,    # Shuddha Dhaivata
    'N2': 792,    # Kaisiki Nishada
    'S_': 1200    # High Sa
}

# %%
def load_audio(file_path, sr=22050):
    """
    Load an audio file.

    Parameters:
    - file_path: Path to the audio file.
    - sr: Sampling rate.

    Returns:
    - y: Audio time series.
    - sr: Sampling rate.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def extract_pitch_histogram(y, sr, bins=128, hist_range=(0, 128)):
    """
    Extract a pitch histogram using HPSS and YIN algorithm.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.
    - bins: Number of histogram bins.
    - hist_range: The lower and upper range of the bins.

    Returns:
    - histogram: Normalized pitch histogram.
    """
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Use YIN for fundamental frequency estimation
    f0 = librosa.yin(y_harmonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
                    frame_length=2048, hop_length=512)
    f0 = f0[f0 > 0]  # Remove unvoiced frames

    if len(f0) == 0:
        return np.zeros(bins)

    # Convert frequencies to MIDI notes
    midi_notes = librosa.hz_to_midi(f0)
    histogram, _ = np.histogram(midi_notes, bins=bins, range=hist_range)
    histogram = histogram.astype(float)
    histogram /= np.sum(histogram)  # Normalize

    return histogram

def load_shruthis(shruthis_dir):
    """
    Load and process tanpura (Shruti) reference audios across all octaves.

    Parameters:
    - shruthis_dir: Directory containing shruti subdirectories.

    Returns:
    - shruti_features: Dictionary mapping frequency to pitch histogram.
    """
    shruti_features = {}
    for kattai_str, freq in kattai_freqs.items():
        
        kattai_dir = os.path.join(shruthis_dir, f'kattai{kattai_str}')
        audio_file = os.path.join(kattai_dir, f'kattai{kattai_str}_audio_1.wav')

        if not os.path.exists(audio_file):
            print(f"Warning: {audio_file} does not exist. Skipping.")
            continue

        y, sr = load_audio(audio_file)
        histogram = extract_pitch_histogram(y, sr)
        shruti_features[freq] = histogram

        print(f"Loaded Shruti for Kattai {kattai_str} ({freq:.2f} Hz)")

    return shruti_features

def process_test_audio(test_audio_path, shruti_features):
    """
    Process the test audio and compute similarity with shruti references.

    Parameters:
    - test_audio_path: Path to the test audio file.
    - shruti_features: Dictionary mapping frequency to pitch histogram.

    Returns:
    - similarity_scores: Dictionary mapping frequency to similarity score.
    """
    y, sr = load_audio(test_audio_path)
    test_histogram = extract_pitch_histogram(y, sr)

    similarity_scores = {}
    for freq, shruti_hist in shruti_features.items():
        distance = euclidean(test_histogram, shruti_hist)
        similarity_scores[freq] = distance

        print(f"Distance with {freq:.2f} Hz Shruti: {distance:.4f}")

    return similarity_scores

def predict_tonic(similarity_scores):
    """
    Predict the tonic frequency based on similarity scores.

    Parameters:
    - similarity_scores: Dictionary mapping frequency to similarity score.

    Returns:
    - predicted_freq: Predicted tonic frequency.
    - sorted_scores: List of frequencies sorted by similarity.
    """
    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1])
    predicted_freq = sorted_scores[0][0]
    return predicted_freq, sorted_scores

def plot_similarity(similarity_scores):
    """
    Plot similarity scores for visualization.
    """
    plt.close('all')  # Close any existing plots
    freqs = list(similarity_scores.keys())
    scores = list(similarity_scores.values())

    # Sort frequencies for better visualization
    sorted_indices = np.argsort(freqs)
    freqs_sorted = np.array(freqs)[sorted_indices]
    scores_sorted = np.array(scores)[sorted_indices]

    plt.figure(figsize=(14, 7))
    plt.bar(freqs_sorted, scores_sorted, color='skyblue')
    plt.xlabel('Tonic Frequency (Hz)')
    plt.ylabel('Euclidean Distance')
    plt.title('Distance Scores between Test Audio and Shruti References')
    plt.xticks(freqs_sorted, rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the plot after showing

def extract_carnatic_features(y, sr, tonic_freq, frame_length=2048, hop_length=512):
    """
    Extract features specific to Carnatic music analysis.
    """
    features = {}
    
    # Pitch tracking using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Convert to cents
    pitch_cents = 1200 * np.log2(f0 / tonic_freq)
    pitch_cents[~voiced_flag] = np.nan
    
    # Calculate times for the frames
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # Calculate onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    features['pitch_cents'] = pitch_cents
    features['times'] = times
    features['onset_env'] = onset_env
    features['voiced_flag'] = voiced_flag
    
    return features

# Define swara ratios relative to Sa
SWARA_RATIOS = {
    'S': 1.0,
    'R2': 9/8,     # Chathushruthi Rishabha
    'G2': 5/4,     # Sadharana Gandhara
    'M2': 45/32,   # Prati Madhyama
    'P': 3/2,      # Panchama
    'D1': 5/3,     # Shuddha Dhaivata
    'N2': 15/8,    # Kaisiki Nishada
    'S_': 2.0      # High Sa
}

def identify_swaras(features, window_size=4):  # Small window for fast notes
    """
    Identify swaras using cents values and include octave information.
    """
    pitch_cents = features['pitch_cents']
    times = features['times']
    last_swara = None
    detected_swaras = []
    
    # Parameters - all thresholds significantly lowered
    CENTS_TOLERANCE = 70  # Increased tolerance for matching
    STABILITY_THRESHOLD = 200  # Much more lenient for pitch variation
    MIN_VOICED_FRAMES = 1  # Detect even single-frame notes
    HOP_SIZE = 1  # Check every frame
    MIN_TIME_DIFF = 0.03  # Minimum time between notes (30ms)
    
    last_detection_time = -MIN_TIME_DIFF  # Initialize last detection time
    
    # Process the entire audio
    for i in range(0, len(pitch_cents) - window_size, HOP_SIZE):
        window = pitch_cents[i:i+window_size]
        
        # Skip if completely unvoiced
        if np.isnan(window).all():
            continue
        
        # Get valid pitches in window
        valid_pitches = window[~np.isnan(window)]
        if len(valid_pitches) < MIN_VOICED_FRAMES:
            continue
        
        current_time = times[i]
        
        # Skip if too close to last detection
        if current_time - last_detection_time < MIN_TIME_DIFF:
            continue
        
        # Calculate stability (but with very lenient threshold)
        pitch_std = np.nanstd(valid_pitches)
        if pitch_std > STABILITY_THRESHOLD:
            continue
        
        # Get current pitch in cents
        curr_cents = np.nanmedian(valid_pitches)
        
        # Determine octave (only marking high octave)
        octave = 'middle'
        if curr_cents >= 1200:
            octave = 'high'
            curr_cents_normalized = curr_cents % 1200
        else:
            curr_cents_normalized = curr_cents
            
        # Find closest swara
        closest_swara = min(SWARA_CENTS.items(), 
                          key=lambda x: abs(x[1] - curr_cents_normalized))
        
        # Accept if within tolerance
        if abs(closest_swara[1] - curr_cents_normalized) <= CENTS_TOLERANCE:
            # Calculate onset strength
            onset_strength = np.mean(features['onset_env'][i:i+window_size])
            
            # Detect if this is a fast note
            is_fast = (current_time - last_detection_time) < 0.1
            
            detected_swaras.append({
                'time': current_time,
                'swara': closest_swara[0],
                'pitch': curr_cents,
                'stability': pitch_std,
                'octave': octave,
                'onset_strength': onset_strength,
                'is_fast': is_fast
            })
            
            # Update last detection time
            last_detection_time = current_time
            
            # Print notation
            notation = closest_swara[0]
            if octave == 'high':
                notation = f"{notation}*"
            if is_fast:
                notation = notation.lower()
                
            if notation != last_swara:
                print(notation, end=' ', flush=True)
                last_swara = notation
    
    print()
    return detected_swaras

def generate_swara_notation(swaras, min_duration=0.03):  # Shorter minimum duration
    """
    Generate a readable transcription of the detected swaras with timing.
    """
    transcription = []
    current_line = []
    line_duration = 0
    last_time = 0

    for i, swara in enumerate(swaras):
        # Apply octave notation
        notation = swara['swara']
        if swara['octave'] == 'high':
            notation = f"{notation}*"

        # Use lowercase for fast sequences
        if swara['is_fast']:
            notation = notation.lower()
        
        # Start new line after 5 seconds
        if line_duration > 5.0:
            transcription.append(" ".join(current_line))
            current_line = []
            line_duration = 0

        current_line.append(notation)
        line_duration += min_duration

    # Add remaining notes
    if current_line:
        transcription.append(" ".join(current_line))

    # Print with timing information
    for i, line in enumerate(transcription):
        timestamp = f"{i*5:.1f}s"
        print(f"{timestamp}: {line}")

def plot_carnatic_analysis(y, sr, features, swaras):
    """
    Visualize the Carnatic music analysis.
    """
    plt.figure(figsize=(15, 10))

    # Plot 1: Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # Plot 2: Pitch track with identified swaras
    plt.subplot(3, 1, 2)
    plt.plot(features['times'], features['pitch_cents'], 'b-', alpha=0.5, label='Pitch')
    
    # Plot identified swaras
    swara_times = [s['time'] for s in swaras]
    swara_pitches = [s['pitch'] for s in swaras]
    plt.scatter(swara_times, swara_pitches, c='r', alpha=0.7, label='Swaras')
    
    plt.ylabel('Cents')
    plt.legend()
    plt.title('Pitch Track and Identified Swaras')

    # Plot 3: Onset strength
    plt.subplot(3, 1, 3)
    plt.plot(librosa.times_like(features['onset_env']), features['onset_env'])
    plt.ylabel('Onset Strength')
    plt.xlabel('Time (s)')
    plt.title('Rhythm Analysis')

    plt.tight_layout()
    plt.show()

def generate_swara_tone(freq, duration=0.2, sr=22050):  # Reduced duration to 0.2s
    """
    Generate a pure tone for a given swara frequency.
    """
    t = np.linspace(0, duration, int(sr * duration))
    # Generate sine wave
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    # Apply envelope to avoid clicks
    envelope = np.concatenate([
        np.linspace(0, 1, int(0.05 * sr)),  # Faster attack
        np.ones(int((duration - 0.1) * sr)),
        np.linspace(1, 0, int(0.05 * sr))    # Faster release
    ])
    return tone * envelope

def playback_swaras(swaras, tonic_freq):
    """
    Play back the detected swara sequence.
    """
    print("\nPlaying back detected swaras...")
    sr = 22050  # Sample rate
    
    for swara in swaras:
        # Get the swara's frequency relative to tonic
        if swara['octave'] == 'lower':
            octave_mult = 0.5
        elif swara['octave'] == 'high':
            octave_mult = 2.0
        else:
            octave_mult = 1.0
            
        # Get base frequency ratio for the swara
        base_ratio = SWARA_RATIOS[swara['swara'].replace('*', '').replace('.', '')]
        
        # Calculate actual frequency
        freq = tonic_freq * base_ratio * octave_mult
        
        # Adjust duration based on whether it's a fast note
        duration = 0.1 if swara.get('is_fast', False) else 0.2
        
        # Generate and play tone
        tone = generate_swara_tone(freq, duration=duration)
        sd.play(tone, sr)
        sd.wait()  # Wait until playback is finished
        
        # Print the swara being played
        notation = swara['swara']
        if swara['octave'] == 'high':
            notation = f"{notation}*"
        if swara.get('is_fast', False):
            notation = notation.lower()
        print(notation, end=' ', flush=True)
        
    print("\nPlayback complete")

def analyze_carnatic_performance(audio_path, shruti_features):
    """
    Complete analysis of a Carnatic music performance.
    """
    print("\nStarting analysis...")
    
    # Load and process audio
    y, sr = load_audio(audio_path)
    print("Audio loaded successfully")

    # Extract pitch histogram and predict tonic
    similarity_scores = process_test_audio(audio_path, shruti_features)
    predicted_freq, sorted_scores = predict_tonic(similarity_scores)
    print(f"\nPredicted Tonic Frequency: {predicted_freq:.2f} Hz")
    print(f"Predicted Tonic Note: {librosa.hz_to_note(predicted_freq, octave=True)}")

    # Plot similarity scores
    plot_similarity(similarity_scores)

    # Extract features
    print("\nExtracting features...")
    features = extract_carnatic_features(y, sr, predicted_freq)
    
    # Identify swaras
    print("\nIdentifying swaras...")
    print("Swaras detected: ", end='')
    swaras = identify_swaras(features)
    if swaras:
        print("\nSwaras identified successfully")
    else:
        print("\nNo swaras detected")
        return

    # Plot analysis
    print("Plotting analysis...")
    try:
        plot_carnatic_analysis(y, sr, features, swaras)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    # Print complete swara analysis
    print("\nComplete Swara Analysis:")
    print("Time (s) | Swara | Stability")
    print("-" * 40)
    for swara in swaras:
        if swara['octave'] == 'lower':
            notation = f".{swara['swara']}"
        elif swara['octave'] == 'high':
            notation = f"{swara['swara']}*"
        else:
            notation = swara['swara']
        print(f"{swara['time']:7.2f} | {notation:5} | {swara['stability']:8.2f}")

    # Generate and print swara notation
    print("\nComplete Swara Sequence:")
    generate_swara_notation(swaras)

    print("\nWould you like to hear the detected swaras? (y/n)")
    if input().lower() == 'y':
        playback_swaras(swaras, predicted_freq)

    print("\nAnalysis complete")

# %%
# Load Shruti references
shruthis_dir = 'shruthis'  
test_audio_path = "./audios/muthai_tharu.wav"

if not os.path.exists(shruthis_dir):
    print(f"Error: {shruthis_dir} directory does not exist.")
    exit(1)

if not os.path.exists(test_audio_path):
    print(f"Error: {test_audio_path} does not exist.")
    exit(1)

shruti_features = load_shruthis(shruthis_dir)

if not shruti_features:
    print("Error: No Shruti references loaded. Exiting.")
    exit(1)

print("Starting Carnatic performance analysis...")
analyze_carnatic_performance(test_audio_path, shruti_features)

# %%
def extract_note_features(y, sr, frame_size=2048, hop_length=512):
    """
    Extract features for each frame of audio
    """
    features = {}
    
    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Pitch features
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=frame_size,
        hop_length=hop_length
    )
    
    # Pitch variations (for gamaka detection)
    pitch_diff = np.diff(f0, prepend=f0[0])
    pitch_diff_abs = np.abs(pitch_diff)
    
    # Create feature matrix
    feature_matrix = np.vstack([
        spec_centroid,
        spec_rolloff,
        mfcc,
        f0,
        pitch_diff,
        pitch_diff_abs,
        voiced_probs
    ])
    
    return feature_matrix.T  # Shape: (n_frames, n_features)

class SwaraClassifier:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(19,)),  # Number of features
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(SWARA_CENTS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def prepare_dataset(self, audio_files, labels):
        """
        Prepare dataset from audio files and their labels
        """
        X = []
        y = []
        
        for audio_file, label_file in zip(audio_files, labels):
            # Load audio
            audio, sr = librosa.load(audio_file, sr=22050)
            
            # Extract features
            features = extract_note_features(audio, sr)
            
            # Load labels (assuming CSV with time, swara)
            labels_df = pd.read_csv(label_file)
            
            # Align features with labels
            times = librosa.frames_to_time(
                np.arange(len(features)), 
                sr=sr, 
                hop_length=512
            )
            
            for i, time in enumerate(times):
                # Find the closest label within a small time window
                label_idx = (np.abs(labels_df['time'] - time)).idxmin()
                swara = labels_df.iloc[label_idx]['swara']
                
                X.append(features[i])
                y.append(list(SWARA_CENTS.keys()).index(swara))
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, validation_split=0.2, epochs=50):
        """
        Train the model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=validation_split, 
            random_state=42
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def predict(self, audio, sr):
        """
        Predict swaras for new audio
        """
        # Extract features
        features = extract_note_features(audio, sr)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        
        # Convert to swaras
        swaras = []
        times = librosa.frames_to_time(
            np.arange(len(features)), 
            sr=sr, 
            hop_length=512
        )
        
        for i, (time, pred) in enumerate(zip(times, predictions)):
            swara_idx = np.argmax(pred)
            swara = list(SWARA_CENTS.keys())[swara_idx]
            confidence = pred[swara_idx]
            
            swaras.append({
                'time': time,
                'swara': swara,
                'confidence': confidence
            })
            
        return swaras

# Usage example:
def train_swara_classifier(dataset_path):
    """
    Train the swara classifier with a labeled dataset
    """
    # Load dataset
    audio_files = []
    label_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
                label_files.append(os.path.join(root, file.replace('.wav', '_labels.csv')))
    
    # Create and train classifier
    classifier = SwaraClassifier()
    X, y = classifier.prepare_dataset(audio_files, label_files)
    history = classifier.train(X, y)
    
    return classifier

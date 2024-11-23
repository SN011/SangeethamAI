# %%
from utils import Utils
import scipy

# %%
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


#kattai_freqs

# %%

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


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
    f0 = librosa.yin(y_harmonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
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
    Load and process tanpura (Shruti) reference audios.

    Parameters:
    - shruthis_dir: Directory containing shruti subdirectories.

    Returns:
    - shruti_features: Dictionary mapping frequency to pitch histogram.
    """
    shruti_features = {}
    for kattai_str, freq in kattai_freqs.items():
        # Convert kattai key from float to string with underscore
        kattai_dir = os.path.join(shruthis_dir, f'kattai{kattai_str}')
        audio_file = os.path.join(kattai_dir, f'kattai{kattai_str}_audio_1.wav')

        if not os.path.exists(audio_file):
            print(f"Warning: {audio_file} does not exist. Skipping.")
            continue

        y, sr = load_audio(audio_file)
        histogram = extract_pitch_histogram(y, sr)
        shruti_features[freq] = histogram

        print(f"Loaded Shruti for Kattai {kattai_str} ({freq} Hz)")

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

        print(f"Distance with {freq} Hz Shruti: {distance:.4f}")

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

    Parameters:
    - similarity_scores: Dictionary mapping frequency to similarity score.
    """
    freqs = list(similarity_scores.keys())
    scores = list(similarity_scores.values())

    plt.figure(figsize=(12, 6))
    plt.bar(freqs, scores, color='skyblue')
    plt.xlabel('Tonic Frequency (Hz)')
    plt.ylabel('Euclidean Distance')
    plt.title('Distance Scores between Test Audio and Shruti References')
    plt.xticks(freqs, rotation=45)
    plt.tight_layout()
    plt.show()


def plot_similarity_with_swaras(similarity_scores, swaras_freq):
    """
    Plot similarity scores and swara frequencies.
    
    Parameters:
    - similarity_scores: Dictionary of frequencies and their similarity scores
    - swaras_freq: Dictionary of swaras and their frequencies
    """
    if not similarity_scores or not swaras_freq:
        print("No data to plot")
        return
        
    # Extract frequencies and scores from similarity_scores
    freqs = list(similarity_scores.keys())
    scores = list(similarity_scores.values())

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(freqs, scores, color='skyblue', label='Similarity Scores')

    # Plot Swara frequencies
    max_score = max(scores) if scores else 1.0
    for swara, freq in swaras_freq.items():
        plt.axvline(x=freq, color='red', linestyle='--', alpha=0.5)
        plt.text(freq, max_score * 0.9, swara, 
                rotation=90, verticalalignment='center', color='red')

    plt.xlabel('Tonic Frequency (Hz)')
    plt.ylabel('Distance')
    plt.title('Distance Scores between Test Audio and Shruti References with Swaras')
    plt.xticks(freqs, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
import time

start_time = time.time()
shruti_features = load_shruthis('shruthis')

# %%
shruthis_dir = 'shruthis'  
test_audio_path = 'audios/chakkani_thalliki.wav'  

if not os.path.exists(shruthis_dir):
    print(f"Error: {shruthis_dir} directory does not exist.")
    exit(1)

if not os.path.exists(test_audio_path):
    print(f"Error: {test_audio_path} does not exist.")
    exit(1)


if not shruti_features:
    print("Error: No Shruti references loaded. Exiting.")
    exit(1)

similarity_scores = process_test_audio(test_audio_path, shruti_features)


predicted_freq, sorted_scores = predict_tonic(similarity_scores)
print(f"\nPredicted Tonic Frequency: {predicted_freq} Hz")


plot_similarity(similarity_scores)

note = librosa.hz_to_note(predicted_freq, octave=False)
print(f"Predicted Tonic Note: {note}")
print("Shruthi operations took", time.time() - start_time, "seconds")

# %%
# Define Swaras and their frequency ratios relative to Adhara Shadja (Sa)
swaras_ratios = {
    'Sa': 1.0,            # Adhara Shadja
    'Ri1': 16/15,         # Shuddha Rishabha
    'Ri2/Ga1': 9/8,       # Chatushruti Rishabha / Shuddha Gandhara
    'Ri3/Ga2': 6/5,       # Shatshruti Rishabha / Sadharana Gandhara
    'Ga3': 5/4,           # Anthara Gandhara
    'Ma1': 4/3,           # Shuddha Madhyama
    'Ma2': 7/5,           # Prati Madhyama
    'Pa': 3/2,            # Panchama
    'Da1': 8/5,           # Shuddha Dhaivata
    'Da2/Ni1': 27/16,     # Chatushruti Dhaivata / Shuddha Nishada
    'Da3/Ni2': 9/5,       # Shatshruti Dhaivata / Kaishiki Nishada
    'Ni3': 15/8,          # Kakali Nishada
    'SA': 2.0             # Tara Shadja (octave)
}


# %%
def calculate_swara_frequencies(sa_freq):
    """
    Calculate the absolute frequencies of all Swaras in three octaves.
    
    Parameters:
    - sa_freq: Frequency of middle Sa in Hz
    
    Returns:
    - Dictionary mapping each Swara (with octave) to its frequency
    """
    swaras_freq = {}
    
    # Define octaves
    octaves = ['_low', '', '_high']  # Lower octave, middle octave, higher octave
    octave_multipliers = [0.5, 1, 2]  # Half frequency, normal, double frequency
    
    # Calculate frequencies for each octave
    for octave, multiplier in zip(octaves, octave_multipliers):
        base_freq = sa_freq * multiplier
        for swara, ratio in swaras_ratios.items():
            swara_name = f"{swara}{octave}" if octave else swara
            swaras_freq[swara_name] = base_freq * ratio
    
    return swaras_freq


# %%
def map_frequency_to_swara(freq, swaras_freq, tolerance_cents=50):
    """
    Map a frequency to the nearest Swara across all octaves.
    """
    nearest_swara = None
    swara_freq = 0
    min_diff = float('inf')
    
    # Convert tolerance_cents to frequency ratio
    tolerance_ratio = 2 ** (tolerance_cents/1200)
    
    for swara, s_freq in swaras_freq.items():
        # Avoid division by zero and ensure valid frequencies
        if freq <= 0 or s_freq <= 0:
            continue
            
        try:
            cents_diff = abs(1200 * np.log2(freq/s_freq))
            if cents_diff < min_diff and cents_diff < tolerance_cents:
                min_diff = cents_diff
                nearest_swara = swara
                swara_freq = s_freq
        except (ZeroDivisionError, RuntimeWarning):
            continue
    
    return nearest_swara, swara_freq, min_diff


# %%
# Existing code continuation...
start_time = time.time()
similarity_scores = process_test_audio(test_audio_path, shruti_features)

predicted_freq, sorted_scores = predict_tonic(similarity_scores)
print(f"\nPredicted Tonic Frequency: {predicted_freq} Hz")

plot_similarity(similarity_scores)

# Existing note conversion
note = librosa.hz_to_note(predicted_freq, octave=False)
print(f"Predicted Tonic Note: {note}")

# --- Added Code Starts Here ---

# Step 1: Calculate Swara Frequencies based on predicted Sa frequency
swaras_freq = calculate_swara_frequencies(predicted_freq)

# Step 2: Map the predicted frequency to the nearest Swara
nearest_swara, swara_freq, difference = map_frequency_to_swara(predicted_freq, swaras_freq)

print(f"Mapped Swara: {nearest_swara}")
print(f"Swara Frequency: {swara_freq:.2f} Hz")
print(f"Difference: {difference:.2f} Hz")

# Display all Swaras with their frequencies
print("\nSwaras Frequencies:")
for swara, freq in swaras_freq.items():
    print(f"{swara}: {freq:.2f} Hz")

# Plot with the new function
plot_similarity_with_swaras(similarity_scores, swaras_freq)
print("Swara operations 1 took", time.time() - start_time, "seconds")
# --- Added Code Ends Here ---

def isolate_melody(y, sr):
    """
    Isolate the main melody from audio by combining multiple techniques.
    """
    # 1. Harmonic-Percussive Source Separation
    y_harmonic, _ = librosa.effects.hpss(
        y,
        kernel_size=31,
        power=2.0,
        margin=3.0
    )
    
    # 2. Apply median filtering to reduce noise
    y_filtered = scipy.signal.medfilt(y_harmonic, kernel_size=3)
    
    # 3. Apply bandpass filter to focus on melody frequency range
    nyquist = sr / 2
    cutoff_low = 100  # Hz - below typical Carnatic music range
    cutoff_high = 2000  # Hz - above typical Carnatic music range
    b, a = scipy.signal.butter(3, 
                             [cutoff_low/nyquist, cutoff_high/nyquist], 
                             btype='band')
    y_bandpassed = scipy.signal.filtfilt(b, a, y_filtered)
    
    # 4. Spectral gating for noise reduction
    S = librosa.stft(y_bandpassed)
    S_db = librosa.amplitude_to_db(np.abs(S))
    
    # Calculate noise threshold
    noise_threshold = np.median(S_db) + 20  # 20 dB above median
    mask = S_db > noise_threshold
    
    # Apply soft mask
    S_clean = S * mask
    y_clean = librosa.istft(S_clean)
    
    # 5. Normalize
    y_clean = librosa.util.normalize(y_clean)
    
    return y_clean

def transcribe_audio(audio_path, sa_freq, tolerance_cents=40):
    """
    Transcribe audio using melodic inertia to prevent octave jumps.
    """
    # Load and isolate melody
    y, sr = load_audio(audio_path)
    y_clean = isolate_melody(y, sr)
    
    # Time stretch the cleaned audio
    y_slow = librosa.effects.time_stretch(y_clean, rate=0.25)
    
    # Calculate frequencies once
    swaras_freq = calculate_swara_frequencies(sa_freq)
    
    # Do pYIN on cleaned audio
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_slow,
        fmin=swaras_freq['Da1_low'],
        fmax=swaras_freq['SA_high'],
        sr=sr,
        hop_length=512,
        center=True,
        pad_mode='reflect'
    )
    
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    
    # Initialize transcription variables
    transcription = []
    min_duration = 0.1
    current_swara = None
    current_freq = None
    start_time = 0
    
    # Keep track of melodic context
    recent_freqs = []  # Store recent frequencies
    max_context = 10   # Number of recent frequencies to consider
    
    # Add these new variables
    last_stable_octave = None
    stable_count = 0
    required_stable_frames = 5  # Number of frames needed to confirm octave change
    
    for time, freq in zip(times, f0):
        if freq is not None and freq > 0:
            # Update melodic context
            recent_freqs.append(freq)
            if len(recent_freqs) > max_context:
                recent_freqs.pop(0)
            
            # Get median frequency of recent notes
            if recent_freqs:
                median_freq = np.median(recent_freqs)
                
                # Special handling for Sa and SA (octave)
                if len(recent_freqs) > 3:
                    octave_ratio = freq / median_freq
                    
                    # Check if we're near Sa or SA frequencies
                    sa_ratio = freq / sa_freq
                    is_near_sa = abs(1200 * np.log2(sa_ratio)) < tolerance_cents
                    is_near_SA = abs(1200 * np.log2(sa_ratio/2)) < tolerance_cents
                    
                    if is_near_sa or is_near_SA:
                        # If we're near either Sa or SA, use the last stable octave
                        if last_stable_octave is not None:
                            if last_stable_octave == "high" and is_near_sa:
                                freq = freq * 2
                            elif last_stable_octave == "low" and is_near_SA:
                                freq = freq / 2
                        
                        # Update octave stability
                        current_octave = "high" if is_near_SA else "low"
                        if current_octave == last_stable_octave:
                            stable_count += 1
                        else:
                            stable_count = 1
                            
                        # Only change stable octave after enough consistent frames
                        if stable_count >= required_stable_frames:
                            last_stable_octave = current_octave
                    
                    # For other notes, use normal octave correction
                    elif octave_ratio > 1.8:  # More than ~octave up
                        freq = freq / 2
                    elif octave_ratio < 0.6:  # More than ~octave down
                        freq = freq * 2
            
            # Find possible swara matches
            possible_matches = []
            for swara, swara_freq in swaras_freq.items():
                cents_diff = abs(1200 * np.log2(freq/swara_freq))
                if cents_diff < tolerance_cents:
                    possible_matches.append((swara, swara_freq, cents_diff))
            
            if possible_matches:
                possible_matches.sort(key=lambda x: x[2])
                
                # Determine best match using melodic context
                best_match = None
                
                if current_swara:
                    current_octave = '_low' if '_low' in current_swara else '_high' if '_high' in current_swara else ''
                    base_swara = current_swara.replace('_low', '').replace('_high', '')
                    
                    # Strong preference for same octave
                    same_octave_matches = [m for m in possible_matches 
                                         if (current_octave in m[0] or 
                                             (current_octave == '' and '_' not in m[0]))]
                    
                    # Also consider adjacent octaves only if moving stepwise
                    if not same_octave_matches:
                        for match in possible_matches:
                            match_base = match[0].replace('_low', '').replace('_high', '')
                            if abs(list(swaras_ratios.keys()).index(match_base) - 
                                 list(swaras_ratios.keys()).index(base_swara)) <= 2:
                                best_match = match[0]
                                break
                    else:
                        best_match = same_octave_matches[0][0]
                
                if not best_match:
                    best_match = possible_matches[0][0]
                
                if current_swara != best_match:
                    if current_swara and (time - start_time) >= min_duration:
                        transcription.append((start_time, time, current_swara))
                    current_swara = best_match
                    current_freq = freq
                    start_time = time
        
        else:
            if current_swara and (time - start_time) >= min_duration:
                transcription.append((start_time, time, current_swara))
            current_swara = None
            current_freq = None
            start_time = time
            recent_freqs = []  # Reset melodic context on silence
    
    return transcription

def print_transcription(transcription):
    """
    Print the transcription in a readable format.
    """
    if not transcription:
        print("No transcription available")
        return
        
    print("\nTranscribed Swaras:")
    print("Start(s)  End(s)    Swara         Duration(s)")
    print("-" * 50)
    
    for start, end, swara in transcription:
        duration = end - start
        # Only print reasonable durations
        if 0 <= start <= end and duration <= 10:  # Added more validation
            print(f"{start:7.2f}  {end:7.2f}  {swara:12s}  {duration:6.2f}")

def plot_transcription(transcription, audio_path, sa_freq):
    """
    Plot the transcription with time-stretched analysis.
    """
    y, sr = load_audio(audio_path)
    
    # Extract pitch
    hop_length = 512
    frame_length = 2048
    
    plt.style.use('default')
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C8'),
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length
    )
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    ax.set_facecolor('white')
    
    times = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)
    
    # Ensure arrays have same length
    min_length = min(len(times), len(f0))
    times = times[:min_length]
    f0 = f0[:min_length]
    
    # Calculate swara frequencies for all octaves
    swaras_freq = calculate_swara_frequencies(sa_freq)
    
    # Plot pitch track
    voiced_indices = ~np.isnan(f0)
    plt.plot(times[voiced_indices], f0[voiced_indices], 
             color='lightgray', alpha=0.5, linewidth=1, 
             label='Pitch Track')
    
    # Define colors for each base swara
    base_swaras = list(swaras_ratios.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(base_swaras)))
    color_dict = dict(zip(base_swaras, colors))
    
    # Plot swara lines
    for swara, freq in swaras_freq.items():
        base_swara = swara.split('_')[0]
        color = color_dict[base_swara]
        plt.axhline(y=freq, color=color, linestyle='--', alpha=0.3, linewidth=0.5)
        plt.text(-0.5, freq, swara, ha='right', va='center', 
                fontsize=8, color=color)
    
    # Plot transcribed swaras
    used_labels = set()
    for start, end, swara in transcription:
        base_swara = swara.split('_')[0]
        freq = swaras_freq[swara]
        color = color_dict[base_swara]
        
        label = base_swara if base_swara not in used_labels else ""
        plt.plot([start, end], [freq, freq], 
                color=color, linewidth=2, 
                label=label if label else None)
        
        if end - start > 0.3:
            plt.text((start + end)/2, freq, swara, 
                    ha='center', va='bottom', fontsize=8,
                    color=color)
        used_labels.add(base_swara)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.title('Swara Transcription (Multiple Octaves)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.2, linestyle=':')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=10)
    
    y_min = min(swaras_freq.values()) * 0.9
    y_max = max(swaras_freq.values()) * 1.1
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig, ax

def synthesize_transcription(transcription, sr=22050):
    """
    Synthesize audio from transcription using sine waves.
    
    Parameters:
    - transcription: List of (start_time, end_time, swara) tuples
    - sr: Sampling rate (default: 22050 Hz)
    
    Returns:
    - synthesized: Audio signal array
    """
    # First, find the total duration needed (divide by 4 for the time stretch)
    total_duration = max(end for _, end, _ in transcription) / 4
    total_samples = int(total_duration * sr)
    synthesized = np.zeros(total_samples)
    
    # For each note in the transcription
    for start, end, swara in transcription:
        # Adjust duration (divide by 4 as per the time stretch)
        adjusted_start = start / 4
        adjusted_end = end / 4
        
        # Convert time to sample indices
        start_idx = int(adjusted_start * sr)
        end_idx = int(adjusted_end * sr)
        
        # Get the frequency for this swara
        if swara in swaras_freq:
            freq = swaras_freq[swara]
            
            # Generate sine wave for this note
            duration = end_idx - start_idx
            note_t = np.arange(duration) / sr
            
            # Apply envelope to avoid clicks
            envelope = np.ones(duration)
            attack_samples = int(0.01 * sr)  # 10ms attack
            release_samples = int(0.01 * sr)  # 10ms release
            
            # Apply attack and release only if the note is long enough
            if duration > (attack_samples + release_samples):
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                envelope[-release_samples:] = np.linspace(1, 0, release_samples)
            
            note = np.sin(2 * np.pi * freq * note_t) * envelope
            
            # Add to main array
            if end_idx <= total_samples:
                synthesized[start_idx:end_idx] += note
    
    # Normalize
    synthesized = synthesized / np.max(np.abs(synthesized))
    
    return synthesized

def compare_melodic_similarity(original_path, synthesized_path, sr=22050):
    """
    Compare just the melodic content between original and synthesized audio.
    """
    # Load both files
    y_orig, _ = librosa.load(original_path, sr=sr)
    y_synth, _ = librosa.load(synthesized_path, sr=sr)
    
    # Extract fundamental frequency contours
    f0_orig, voiced_flag_orig, _ = librosa.pyin(
        y_orig, 
        fmin=50,
        fmax=2000,
        sr=sr,
        hop_length=512
    )
    
    f0_synth, voiced_flag_synth, _ = librosa.pyin(
        y_synth,
        fmin=50,
        fmax=2000,
        sr=sr,
        hop_length=512
    )
    
    # Convert frequencies to cents (relative pitch)
    cents_orig = 1200 * np.log2(f0_orig[voiced_flag_orig]/f0_orig[voiced_flag_orig][0])
    cents_synth = 1200 * np.log2(f0_synth[voiced_flag_synth]/f0_synth[voiced_flag_synth][0])
    
    # Compare melodic contours
    min_len = min(len(cents_orig), len(cents_synth))
    melodic_similarity = 1 - np.mean(np.abs(cents_orig[:min_len] - cents_synth[:min_len])) / 1200
    
    # Compare note timing
    note_changes_orig = np.diff(cents_orig) != 0
    note_changes_synth = np.diff(cents_synth) != 0
    timing_similarity = np.mean(note_changes_orig[:min_len-1] == note_changes_synth[:min_len-1])
    
    print("\nMelodic Similarity Analysis:")
    print("-" * 50)
    print(f"Pitch Contour Accuracy: {melodic_similarity:.2%}")
    print(f"Note Timing Accuracy: {timing_similarity:.2%}")
    
    overall = (melodic_similarity * 0.6 + timing_similarity * 0.4)
    print(f"\nOverall Melodic Similarity: {overall:.2%}")
    
    return {
        'melodic_similarity': melodic_similarity,
        'timing_similarity': timing_similarity,
        'overall': overall
    }

# Main execution code:
start_time = time.time()
try:
    transcription = transcribe_audio(test_audio_path, predicted_freq, tolerance_cents=200)
    if not transcription:
        print("No notes were detected. Adjusting thresholds might be needed.")
    else:
        print_transcription(transcription)
        print("\nSynthesizing audio from transcription...")
        synthesized_audio = synthesize_transcription(transcription)
        output_path = 'synthesized_transcription.wav'
        scipy.io.wavfile.write(output_path, 22050, synthesized_audio.astype(np.float32))
        print(f"Saved synthesized audio to {output_path}")
except Exception as e:
    print(f"Error during transcription or synthesis: {str(e)}")
    import traceback
    traceback.print_exc()
print("Swara operations (transcribe and synthesize) took", time.time() - start_time, "seconds")

# # After synthesizing...
# try:
#     # Your existing synthesis code...
#     print("\nComparing synthesized audio with original...")
#     similarity_scores = compare_melodic_similarity(test_audio_path, output_path)
# except Exception as e:
#     print(f"Error during audio comparison: {str(e)}")


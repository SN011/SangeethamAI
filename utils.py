
from yt_dlp import YoutubeDL
import os
import ffmpeg


def print_directory_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# List of YouTube URLs
urls = [
    "https://www.youtube.com/watch?v=JeiYQpMdlqU&list=PLE71A90EC89E8494E&index=1",  # 0.75
    "https://www.youtube.com/watch?v=z6y2IST7lbk&list=PLE71A90EC89E8494E&index=2",  # 1
    "https://www.youtube.com/watch?v=U8MtLItCHSw&list=PLE71A90EC89E8494E&index=3",  # 1.5
    "https://www.youtube.com/watch?v=xbW94-sHLF0&list=PLE71A90EC89E8494E&index=4",  # 2
    "https://www.youtube.com/watch?v=IXGUEOeZmyQ&list=PLE71A90EC89E8494E&index=5",  # 2.5
    "https://www.youtube.com/watch?v=0wAKomfxsl0",  # 3
    "https://www.youtube.com/watch?v=UMzpL9UaPRE&list=PLE71A90EC89E8494E&index=6",  # 4
    "https://www.youtube.com/watch?v=4SnRCz0KX3Q&list=PLE71A90EC89E8494E&index=7",  # 4.5
    "https://www.youtube.com/watch?v=fCtMNklEQKM&list=PLE71A90EC89E8494E&index=8",  # 5
    "https://www.youtube.com/watch?v=X2QVrOBGjHw&list=PLE71A90EC89E8494E&index=9",  # 5.5
    "https://www.youtube.com/watch?v=5ddMfvsmTkQ&list=PLE71A90EC89E8494E&index=10",  # 6
    "https://www.youtube.com/watch?v=k0Aa0z_sRdk&t=562s",  # 6.5
    "https://www.youtube.com/watch?v=3valEgYPsGw",  # 7
]

# Corresponding 'kattai' values
kattai = ['0_75', '1', '1_5', '2', '2_5', '3', '4', '4_5', '5', '5_5', '6', '6_5', '7']

# Ensure the base directory exists
base_dir = 'shruthis'
os.makedirs(base_dir, exist_ok=True)

# Download each video
for i, url in enumerate(urls):
    output_dir = os.path.join(base_dir, f'kattai{kattai[i]}')
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,  # Ensure only the single video is downloaded
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])




# Directory containing all subdirectories with downloaded mp4 files
base_dir = 'shruthis'

# Loop through each 'kattai' folder to find mp4 files and convert them to wav
for kattai_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, kattai_folder)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue

    # Process each .mp4 file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4'):
            mp4_path = os.path.join(folder_path, file_name)
            wav_path = os.path.join(folder_path, file_name.replace('.mp4', '.wav'))
            
            # Use ffmpeg to convert mp4 to wav
            try:
                ffmpeg.input(mp4_path).output(wav_path).run(overwrite_output=True)
                print(f"Converted {mp4_path} to {wav_path}")
            except ffmpeg.Error as e:
                print(f"Error converting {mp4_path}: {e}")



# Base directory containing 'kattai' subdirectories
base_dir = 'shruthis'

# Iterate over each subdirectory in the base directory
for kattai_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, kattai_folder)
    
    # Ensure the path is a directory
    if not os.path.isdir(folder_path):
        continue

    # Initialize a counter for naming
    counter = 1

    # Iterate over each file in the subdirectory
    for file_name in os.listdir(folder_path):
        # Process only .wav files
        if file_name.endswith('.wav'):
            old_file_path = os.path.join(folder_path, file_name)
            # Define the new file name pattern
            new_file_name = f"{kattai_folder}_audio_{counter}.wav"
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")
            
            # Increment the counter
            counter += 1





# Base directory containing 'kattai' subdirectories
base_dir = 'shruthis'
# Maximum allowed duration in seconds (10 minutes)
max_duration = 60  # seconds

# Iterate over each 'kattai' folder to find .wav files and shorten them if necessary
for kattai_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, kattai_folder)
    
    # Ensure the path is a directory
    if not os.path.isdir(folder_path):
        continue

    # Process each .wav file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            # Define output path for shortened audio
            output_path = os.path.join(folder_path, f"shortened_{file_name}")
            
            # Use ffmpeg to shorten the audio if it's longer than max_duration
            try:
                # Get the duration of the audio
                probe = ffmpeg.probe(file_path)
                duration = float(probe['format']['duration'])

                # Only shorten if duration exceeds max_duration
                if duration > max_duration:
                    ffmpeg.input(file_path).output(output_path, t=max_duration).run(overwrite_output=True)
                    print(f"Shortened {file_path} to 10 minutes.")
                else:
                    # Copy file if it's already within the limit
                    os.rename(file_path, output_path)
                    print(f"File {file_path} already within limit, copied without modification.")

            except ffmpeg.Error as e:
                print(f"Error processing {file_path}: {e}")



# Base directory containing 'kattai' subdirectories
base_dir = 'shruthis'

# Iterate over each 'kattai' folder
for kattai_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, kattai_folder)
    
    # Ensure the path is a directory
    if not os.path.isdir(folder_path):
        continue

    # Process each file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Delete original .wav files
        if file_name.endswith('.wav') and not file_name.startswith('shortened_'):
            os.remove(file_path)
            print(f"Deleted original file: {file_path}")
        
        # Rename shortened files by removing 'shortened_' prefix
        elif file_name.startswith('shortened_') and file_name.endswith('.wav'):
            new_file_name = file_name.replace('shortened_', '', 1)
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"Renamed '{file_path}' to '{new_file_path}'")

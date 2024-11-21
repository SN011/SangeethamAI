import os
from yt_dlp import YoutubeDL
import ffmpeg

class Utils:
    def __init__(self, base_dir='shruthis', urls=None, kattai=None):
        self.base_dir = base_dir
        self.urls = urls or []
        self.kattai = kattai or []
        os.makedirs(self.base_dir, exist_ok=True)

    def print_directory_tree(self):
        """Print the directory structure starting from base_dir."""
        for root, dirs, files in os.walk(self.base_dir):
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{subindent}{f}')

    def download_videos(self):
        """Download videos from YouTube based on the URLs and kattai mapping."""
        for i, url in enumerate(self.urls):
            output_dir = os.path.join(self.base_dir, f'kattai{self.kattai[i]}')
            os.makedirs(output_dir, exist_ok=True)
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'noplaylist': True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    def convert_videos_to_audio(self):
        """Convert downloaded MP4 files to WAV format."""
        for kattai_folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, kattai_folder)
            if not os.path.isdir(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.mp4'):
                    mp4_path = os.path.join(folder_path, file_name)
                    wav_path = os.path.join(folder_path, file_name.replace('.mp4', '.wav'))
                    try:
                        ffmpeg.input(mp4_path).output(wav_path).run(overwrite_output=True)
                        print(f"Converted {mp4_path} to {wav_path}")
                    except ffmpeg.Error as e:
                        print(f"Error converting {mp4_path}: {e}")

    def rename_audio_files(self):
        """Rename WAV files to a consistent format."""
        for kattai_folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, kattai_folder)
            if not os.path.isdir(folder_path):
                continue

            counter = 1
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    old_file_path = os.path.join(folder_path, file_name)
                    new_file_name = f"{kattai_folder}_audio_{counter}.wav"
                    new_file_path = os.path.join(folder_path, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                    counter += 1

    def shorten_audio_files(self, max_duration=60):
        """Shorten WAV files to a maximum duration."""
        for kattai_folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, kattai_folder)
            if not os.path.isdir(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    output_path = os.path.join(folder_path, f"shortened_{file_name}")
                    try:
                        probe = ffmpeg.probe(file_path)
                        duration = float(probe['format']['duration'])

                        if duration > max_duration:
                            ffmpeg.input(file_path).output(output_path, t=max_duration).run(overwrite_output=True)
                            print(f"Shortened {file_path} to {max_duration} seconds.")
                        else:
                            os.rename(file_path, output_path)
                            print(f"File {file_path} already within limit, copied without modification.")
                    except ffmpeg.Error as e:
                        print(f"Error processing {file_path}: {e}")

    def clean_up_files(self):
        """Clean up original WAV files and rename shortened files."""
        for kattai_folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, kattai_folder)
            if not os.path.isdir(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith('.wav') and not file_name.startswith('shortened_'):
                    os.remove(file_path)
                    print(f"Deleted original file: {file_path}")
                elif file_name.startswith('shortened_') and file_name.endswith('.wav'):
                    new_file_name = file_name.replace('shortened_', '', 1)
                    new_file_path = os.path.join(folder_path, new_file_name)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed '{file_path}' to '{new_file_path}'")

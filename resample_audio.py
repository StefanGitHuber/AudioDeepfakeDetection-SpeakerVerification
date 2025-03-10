# coding: utf-8

import librosa
import soundfile as sf

def resample_audio(input_file, output_file, target_sr=16000):
    """
    Resample audio file to specified target sample rate (default 16 kHz)

    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save resampled audio file
        target_sr (int, optional): Target sample rate in Hz. Defaults 16k
    """
    try:
        y, sr = librosa.load(input_file) # Load audio file
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr) # Resample audio
        sf.write(output_file, y_resampled, target_sr) # Save resampled audio
        print(f"Audio resampled and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

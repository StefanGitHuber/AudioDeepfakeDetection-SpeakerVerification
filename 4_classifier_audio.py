#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examine deepfake audio detection using WavLM fine-tuned for sequence classification https://huggingface.co/microsoft/wavlm-base-sv
DeepfakeDetector class loads model + feature extractor, performs deepfake detection on set of original, impostor + fake audio files
Instead of Speaker Verification with 100s/1000s of labels, put only two for the binary decision Real vs Fake
=> Super fails with worst accuracies fluctuating around 50 % like throwing a coin
Analysis: Sure, model made for SV not spoofing detection concentrates on "pointers assuring speaker identity" (formants= vocal tract, prosodic features, pitch contour etc)
"""

import glob
import librosa
import numpy as np
import torch
import torchaudio
from pathlib import Path
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from device_handling import device_detect
DEVICE, _ = device_detect()

import warnings
warnings.filterwarnings('ignore')

MODEL = "microsoft/wavlm-base-plus-sv"
DIR_BASIS = Path("/your/path/to/")
DIR_ORIGS = DIR_BASIS / "data_orig" / "*.wav"
DIR_IMPOS = DIR_BASIS / "data_impostors" / "*.wav"
DIR_FAKES = DIR_BASIS / "data_fakes" / "*.wav"

class DeepfakeDetector:
    def __init__(self):
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # Initialize Deepfake Detection model
        print("Loading audio classifier {MODEL}")
        self.detector_name = MODEL
        self.detector = AutoModelForAudioClassification.from_pretrained(
            self.detector_name,
            num_labels=2,  # Real vs Fake
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.detector_name)

    def load_audio(self, file_path, target_sr=16000):
        """Load and preprocess audio file"""
        wave_form, sample_rate = librosa.load(file_path, sr=None)
        if sample_rate != target_sr:
            wave_form = librosa.resample(wave_form, orig_sr=sample_rate, target_sr=target_sr)
        return wave_form, target_sr

    def detect_deepfake(self, audio_path):
        """Detect if audio is real or fake using WavLM"""
        try:
            # Load and preprocess audio
            wave_form, sample_rate = self.load_audio(audio_path)
            
            # Extract features
            inputs = self.feature_extractor(
                wave_form,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.detector(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            fake_prob = predictions[0][1].item()
            return {
                'is_fake': fake_prob > 0.5,
                'fake_probability': fake_prob,
                'real_probability': 1 - fake_prob
            }

        except Exception as e:
            print(f"Deepfake detection error: {str(e)}")
            return None

def main():
    detector = DeepfakeDetector()
    
    audio_files = []
    audio_files.extend(glob.glob(str(DIR_ORIGS)))
    audio_files.extend(glob.glob(str(DIR_IMPOS)))
    audio_files.extend(glob.glob(str(DIR_FAKES)))

    print("\nAnalyzing audio files for deepfakes:")
    for audio_file in audio_files:
        if Path(audio_file).exists():
            result = detector.detect_deepfake(audio_file)
            if result:
                print(f"\nFile: {Path(audio_file).name}")
                print(f"Fake Probability: {result['fake_probability']:.3f}")
                print(f"Real Probability: {result['real_probability']:.3f}")
                print(f"Classification: {'FAKE' if result['is_fake'] else 'REAL'}")

if __name__ == "__main__":
    main()
#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapt WavLM fine-tuned for Speaker Verification https://huggingface.co/microsoft/wavlm-base-sv
to detect audio deepfakes original + impostor real recordings vs fake synthesis audio.
Fine-tuned model concentrates on everything else then speaker relevant info (formants, prosody) but
subtle gaps + artefacts, especially phase discontinuities (unnatural phase shifts), oversmoothing (a mumbled but not clear audio room effect),
unusual background noise patterns (most likely found in higher frequency regions) and other statistical irregularities (speaking rate + style).
Create labels 0 for real recordings vs 1 for fake (TTS, VC).
"""

import glob
import librosa
import numpy as np
import torch
import torchaudio

from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from typing import Optional, Tuple, Union

import warnings
warnings.filterwarnings('ignore')

from device_handling import device_detect
DEVICE, _ = device_detect() # Handle various GPU types, fallback CPU

MODEL = "microsoft/wavlm-base-plus-sv"
DIR_BASIS = Path("/your/path/to/")
DIR_ORIGS = DIR_BASIS / "data_orig" / "*.wav"
DIR_IMPOS = DIR_BASIS / "data_impostors" / "*.wav"
DIR_FAKES = DIR_BASIS / "data_fakes" / "*.wav"
DIR_MODL = DIR_BASIS / "models_finetuned"

class DeepfakeDetector:
    def __init__(self):
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # Initialize Deepfake Detection model
        print("Loading Deepfake Detection model...")
        self.detector_name = "microsoft/wavlm-base-plus-sv"
        self.detector = AutoModelForAudioClassification.from_pretrained(
            self.detector_name,
            num_labels=2,  # Real vs Fake
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.detector_name)

    def load_audio(self, file_path, target_sr=16000):
        """Load and preprocess audio file"""
        wav, sr = librosa.load(file_path, sr=None)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav, target_sr

    def detect_deepfake(self, audio_path):
        """Detect if audio is real or fake using WavLM"""
        try:
            # Load and preprocess audio
            wav, sr = self.load_audio(audio_path)
            
            # Extract features
            inputs = self.feature_extractor(
                wav,
                sampling_rate=sr,
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

    def fine_tune(self, train_paths, train_labels, validation_paths=None, validation_labels=None):
        """Fine-tune the model on custom dataset"""
        print("Preparing datasets for fine-tuning...")

        data_collator = AudioDataCollatorWithPadding(
            padding="max_length",  # Pad to maximum length
            max_length=200,         # Set maximum length
            pad_to_multiple_of=8,  # Pad to multiple of xyz
        )

        # Split data if validation set not provided
        if validation_paths is None or validation_labels is None:
            train_paths, validation_paths, train_labels, validation_labels = train_test_split(
                train_paths, train_labels, test_size=0.2, random_state=42
            )

        # Create datasets
        train_dataset = AudioDataset(train_paths, train_labels, self.feature_extractor)
        validation_dataset = AudioDataset(validation_paths, validation_labels, self.feature_extractor)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./deepfake_detector_results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps, lowered mem
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.detector,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator
        )

        print("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        self.detector.save_pretrained("./deepfake_detector_finetuned")
        self.feature_extractor.save_pretrained("./deepfake_detector_finetuned")
        print("Fine-tuning completed and model saved!")

@dataclass
class AudioDataCollatorWithPadding:
    """ Data collator dynamically pad audio inputs """
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input values
        padded_input_values = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )

        # Create labels tensor
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_values": padded_input_values, "labels": labels}


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, feature_extractor, sampling_rate=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Load and preprocess audio
        wav, sr = librosa.load(audio_path, sr=None)
        if sr != self.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sampling_rate)

        # Extract features
        inputs = self.feature_extractor(
            wav,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )

        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return {
            **inputs,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def main():
    detector = DeepfakeDetector()
    
    orig_files = []
    orig_files.extend(glob.glob(str(DIR_ORIGS)))
    orig_files.extend(glob.glob(str(DIR_IMPOS)))
    fake_files = glob.glob(str(DIR_FAKES))
    
    # Create labels (0 for real, 1 for fake)
    orig_labels = [0] * len(orig_files)
    fake_labels = [1] * len(fake_files)
    
    all_files = orig_files + fake_files
    all_labels = orig_labels + fake_labels

    # Fine-tune the model
    print("\nStarting fine-tuning process...")
    detector.fine_tune(all_files, all_labels)

    # Test the fine-tuned model
    print("\nTesting fine-tuned model:")
    for audio_file in all_files:
        if Path(audio_file).exists():
            result = detector.detect_deepfake(audio_file)
            if result:
                print(f"\nFile: {Path(audio_file).name}")
                print(f"Fake Probability: {result['fake_probability']:.3f}")
                print(f"Real Probability: {result['real_probability']:.3f}")
                print(f"Classification: {'FAKE' if result['is_fake'] else 'REAL'}")

if __name__ == "__main__":
    main()
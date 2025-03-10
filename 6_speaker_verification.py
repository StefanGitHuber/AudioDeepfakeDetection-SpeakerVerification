#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performing deepfake audio detection is always closest related to speaker verification (if its a speech recording for a know person, e.g. in KYC).
Specific head added to pre-trained WavLM lets WavLMForXVector generate x-vectors:
Statistical pooling layer aggregates frame-level representations from WavLM into single utterance-level representation.
Fully connected layers processes pooled representation to produce x-vectors.
Compare x-vectors of two utterance to determine if same or different speaker.

2D plot of high-dimensional speaker embeddings (x-vectors) for visual analysis of speaker similarity.
PCA (Principal Component Analysis) as linear dimensionality reduction technique.
Writes a CSV file storing all cosine distance between evaluated speaker pairs (its embeddings compare).

Good old Kaldi: i-vector => d-vector => x-vector (i = identity, d=DNN, x=augmented)
One of my favourite papers all time: 190930 Probing the Information Encoded in X-vectors, https://arxiv.org/abs/1909.06351
Basically in wonderful frank manners admitting that x-vectors do NOT serve perfectly its purpose (remove any redundant influence and only identify a speakers identity)
but are highly correlated to many other variation contained in a speech recording !!
"""

import csv
import glob
import librosa
import numpy as np
import os
import torch    
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

import umap
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any

from device_handling import device_detect
DEVICE_STR, _ = device_detect() # Handle various GPU types, fallback CPU

MODLNAM: str = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE: int = 16000
SIMI_THRESH: float = 0.86  # Similarity threshold above which "same speaker" is determined
DIR_BASIS: Path = Path("/your/path/to/")
DIR_ORIGS: Path = DIR_BASIS / "data_orig" / "*.wav"
B_FAKE2ORIGIN: bool = True
if B_FAKE2ORIGIN:
    DIR_FAKES: Path = DIR_BASIS / "data_fakes_fake2orig"
    svcase: str = "SV-fake2orig"
else:
    DIR_FAKES: Path = DIR_BASIS / "data_fakes_orig2fake"
    svcase: str = "SV-orig2fake"
os.makedirs(str(DIR_FAKES), exist_ok=True)
DIR_FAKES: Path = DIR_FAKES / "*.wav"


class SpeakerVerification:
    def __init__(self, threshold: float = SIMI_THRESH, sample_rate: int = SAMPLE_RATE, model_name: str = MODLNAM):
        print("Initializing WavLM speaker verification model...")
        self.feature_extractor: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model: WavLMForXVector = WavLMForXVector.from_pretrained(model_name)
        self.threshold: float = threshold
        self.sample_rate: int = sample_rate

        # Move model to GPU if available
        self.device: str = DEVICE_STR
        print(f"Using device: {self.device}")
        self.model: WavLMForXVector = self.model.to(self.device)

    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file"""
        try:
            wav: np.ndarray
            sr: int
            wav, sr = librosa.load(file_path, sr=self.sample_rate)
            return wav
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None

    def get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """Extract speaker embedding from audio"""
        inputs: Dict[str, torch.Tensor] = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings.cpu()

    def verify_speaker(self, orig_path: str, fake_path: str) -> Optional[Dict[str, Any]]:
        """Verify if two audio files are from the same speaker"""

        orig_audio: Optional[np.ndarray] = self.load_audio(orig_path)
        fake_audio: Optional[np.ndarray] = self.load_audio(fake_path)
        if orig_audio is None or fake_audio is None:
            return None

        # Generate embeddings
        orig_embedding: torch.Tensor = self.get_embedding(orig_audio)
        fake_embedding: torch.Tensor = self.get_embedding(fake_audio)

        # Generate visualization
        self.visualize_embeddings(
            orig_embedding[0],
            fake_embedding[0],
            orig_path,
            fake_path
        )

        # Calculate similarity
        cosine_sim: torch.nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=-1)
        similarity: float = cosine_sim(orig_embedding[0], fake_embedding[0]).item()

        return {
            'similarity': similarity,
            'is_same_speaker': similarity >= self.threshold,
            'orig_file': Path(orig_path).name,
            'fake_file': Path(fake_path).name,
        }

    def visualize_all_embeddings(self, embedding_pairs: List[Tuple[torch.Tensor, torch.Tensor, str, str]], filename: str) -> None:
        """
        Visualize all speaker embeddings in a single plot
        embedding_pairs: List of tuples containing (orig_embedding, fake_embedding, orig_file, fake_file)
        """
        import matplotlib.pyplot as plt
        from datetime import datetime
        import numpy as np
        from sklearn.decomposition import PCA

        # Prepare data for PCA
        all_embeddings: List[np.ndarray] = []
        for orig_emb, fake_emb, _, _ in embedding_pairs:
            all_embeddings.append(orig_emb.numpy())
            all_embeddings.append(fake_emb.numpy())
        all_embeddings = np.vstack(all_embeddings)

        # Apply PCA
        pca: PCA = PCA(n_components=2)
        embeddings_2d: np.ndarray = pca.fit_transform(all_embeddings)

        # Create plot
        plt.figure(figsize=(15, 10))

        # Plot each pair with connecting arrows
        for idx in range(0, len(embeddings_2d), 2):
            orig_point: np.ndarray = embeddings_2d[idx]
            fake_point: np.ndarray = embeddings_2d[idx + 1]

            # Get file names for legend
            orig_file: str = Path(embedding_pairs[idx // 2][2]).stem
            fake_file: str = Path(embedding_pairs[idx // 2][3]).stem

            # Plot points
            plt.scatter(orig_point[0], orig_point[1], c='blue', marker='o', s=100, label=f'Original ({orig_file})')
            plt.scatter(fake_point[0], fake_point[1], c='red', marker='x', s=100, label=f'Fake ({fake_file})')

            # Add arrow connecting the pairs
            plt.arrow(orig_point[0], orig_point[1], fake_point[0] - orig_point[0], fake_point[1] - orig_point[1],
                      color='gray', alpha=0.5, head_width=0.05, head_length=0.1, length_includes_head=True)

            # Add similarity score next to the arrow
            midpoint: np.ndarray = (orig_point + fake_point) / 2
            cosine_sim: float = np.dot(all_embeddings[idx].flatten(), all_embeddings[idx + 1].flatten()) / \
                               (np.linalg.norm(all_embeddings[idx].flatten()) *
                                np.linalg.norm(all_embeddings[idx + 1].flatten()))
            plt.annotate(f'{cosine_sim:.2f}', xy=(midpoint[0], midpoint[1]), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)
        # Customize plot
        plt.title('All Speaker Embeddings Comparison\n' f'Total Pairs: {len(embedding_pairs)}')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add explained variance ratio information
        var_ratio: np.ndarray = pca.explained_variance_ratio_
        plt.figtext(0.02, 0.02,
                    f'Explained variance ratio: {var_ratio[0]:.3f}, {var_ratio[1]:.3f}',
                    fontsize=8, ha='left')

        # Adjust legend to show only unique entries
        handles: List[Any]
        labels: List[str]
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label: Dict[str, Any] = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(1, 0.5))

        # Save plot
        plot_filename: str = f'{filename}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"Combined plot saved as: {plot_filename}")


def main() -> None:
    # Files: Orig vs fake
    orig_files: List[str] = []
    orig_files.extend(glob.glob(str(DIR_ORIGS)))
    fake_files: List[str] = glob.glob(str(DIR_FAKES))
    if not orig_files or not fake_files:
        print(f"Error: No audio files found in {DIR_ORIGS} nor {DIR_FAKES}")
        return

    # Speaker Verification: Init
    verifier: SpeakerVerification = SpeakerVerification(threshold=SIMI_THRESH)

    # Setup CSV output
    timestamp: str = datetime.utcnow().strftime('%Y%m%d_%Hh-%Mm')
    filename: str = f"results_{svcase}_{timestamp}"
    csv_filename: str = filename + ".csv"
    csv_headers: List[str] = [
        'original_file',
        'fake_file',
        'similarity_score',
        'speaker',
    ]

    # Collect all embeddings
    embedding_pairs: List[Tuple[torch.Tensor, torch.Tensor, str, str]] = []

    # Open CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        print(f"\nSave speaker verification results to: {csv_filename}")
        print("-" * 80)

        for orig_file in orig_files:
            for fake_file in fake_files:
                # Load and process audio files
                orig_audio: Optional[np.ndarray] = verifier.load_audio(orig_file)
                fake_audio: Optional[np.ndarray] = verifier.load_audio(fake_file)

                if orig_audio is not None and fake_audio is not None:
                    # Get embeddings
                    orig_embedding: torch.Tensor = verifier.get_embedding(orig_audio)
                    fake_embedding: torch.Tensor = verifier.get_embedding(fake_audio)

                    # Store embeddings for visualization
                    embedding_pairs.append((
                        orig_embedding[0],
                        fake_embedding[0],
                        orig_file,
                        fake_file
                    ))

                    # Calculate cosine similarity
                    cosine_sim: torch.nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=-1)
                    similarity: float = cosine_sim(orig_embedding[0], fake_embedding[0]).item()

                    # Prepare row for CSV
                    csv_row: Dict[str, str] = {
                        'original_file': Path(orig_file).name,
                        'fake_file': Path(fake_file).name,
                        'similarity_score': f"{similarity:.3f}",
                        'speaker': 'Same Speaker' if similarity >= SIMI_THRESH else 'Different Speaker'
                    }

                    # Write to CSV
                    writer.writerow(csv_row)

                    # Print results
                    print(f"\nOrig: {Path(orig_file).name}")
                    print(f"Fake: {Path(fake_file).name}")
                    print(f"Score: {similarity:.3f}")
                    print(f"Speaker: {'Same Speaker' if similarity >= SIMI_THRESH else 'Different Speaker'}")

    # Create single visualization with all embeddings
    verifier.visualize_all_embeddings(embedding_pairs, filename)
    print(f"\nResults saved to {csv_filename}")


if __name__ == "__main__":
    main()        
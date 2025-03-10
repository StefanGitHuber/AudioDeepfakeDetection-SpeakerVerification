#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script compares two speech recordings (cosine distance) using WavLM: Base tuned for Speaker Verification https://huggingface.co/microsoft/wavlm-base-sv
Gradio interface visualizes embeddings and logits using t-SNE to reduce dimensionality for 2D scatter plots.
HF transformer pipelines:
Wav2Vec feature extractor encodes raw audio (resample, normalize, pad/truncate, attention mask)
WavLM computes embeddings + logits for classification
"""

import base64
import io
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchaudio
from pathlib import Path
from sklearn.manifold import TSNE
from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor

from measure_distance import cosine_distance

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

NUM_LABELS = 512
MODEL_NAME = "microsoft/wavlm-base-sv"
DIR_DATA = "/your/path/to/data_orig/"
FIL_WAV1 = os.path.join(DIR_DATA, "orig_ArcticBDL001.wav")
FIL_WAV2 = os.path.join(DIR_DATA, "orig_whatever.wav")
print(f"Using files:\n{FIL_WAV1}\n{FIL_WAV2}")

def process_audio(audio_file1, audio_file2):
    """ Process two audio files using WavLM, return embeddings + t-SNE plots """
    if not os.path.exists(audio_file1) or not os.path.exists(audio_file2):
        return f"Error: Audio files {audio_file1} not found.", f"Error: Audio files {audio_file2} not found.", None

    try:
        # Load audio files
        waveform1, sample_rate1 = torchaudio.load(audio_file1)
        waveform2, sample_rate2 = torchaudio.load(audio_file2)

        # Initialize WavLM model
        model = WavLMForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            classifier_proj_size=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
    
        # Wav2Vec feature extractor performs: zero mean + unit variance, pad/truncate + attention_mask, normalize
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv", ignore_mismatched_sizes=True)
        waveform1_normed = feature_extractor(waveform1.numpy(), sampling_rate=sample_rate1, return_tensors="pt")
        waveform2_normed = feature_extractor(waveform2.numpy(), sampling_rate=sample_rate2, return_tensors="pt")
        print(f"\nProcessing files: \n{audio_file1} \n{audio_file2}")
        print(f"\nFeatures size \n{waveform1_normed['input_values'].shape} \n{waveform2_normed['input_values'].shape}:")

        ##### Inference to classify input audio
        with torch.no_grad():
            # CNN feature encoder => transformer layers => hidden states
            # Increasingly abstract audio features: acoustic/phonetic => phonemic/linguistic => task-specific (speaker identity, phoneme/text)            
            outputs1 = model.wavlm(**waveform1_normed.to(model.device), output_hidden_states=True)
            outputs2 = model.wavlm(**waveform2_normed.to(model.device), output_hidden_states=True)

            # Mean pooled hidden states, dimensionality reduced vector, compact representation
            embeddings1 = outputs1.hidden_states[-1].squeeze().numpy()
            embeddings2 = outputs2.hidden_states[-1].squeeze().numpy()

            # Raw prediction scores passed to softmax/sigmoid final classification layer for speaker-specific propabilities per label size (speaker classes)
            logits1 = model(**waveform1_normed).logits
            logits2 = model(**waveform2_normed).logits

        result1 = f"{Path(audio_file1).name} embeddings of size {embeddings1.shape}: \n{embeddings1}"
        result2 = f"{Path(audio_file2).name} Logits of size {logits2.shape}: {logits2}"

        ##### t-SNE visualization: Embeddings
        all_embeddings = np.concatenate((embeddings1, embeddings2), axis=0)
        perplexity = min(30, len(all_embeddings) - 1)  # Adjust perplexity based on sample size
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Cosine distance: Embeddings
        dist_embeds = cosine_distance(embeddings1, embeddings2)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:len(embeddings1), 0], embeddings_2d[:len(embeddings1), 1], label=Path(audio_file1).name, s=100)
        plt.scatter(embeddings_2d[len(embeddings1):, 0], embeddings_2d[len(embeddings1):, 1], label=Path(audio_file2).name, s=100)
        plt.legend(fontsize=10)
        plt.title(f"t-SNE Embeddings: {dist_embeds:.2f} cosine distance", fontsize=14)
        
        # Convert plot to base64 for Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_embeds = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        ##### t-SNE visualization: logits
        logits_combined = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=0)
        perplexity_logits = min(30, len(logits_combined) - 1)
        tsne_logits = TSNE(n_components=2, random_state=0, perplexity=perplexity_logits)
        logits_2d = tsne_logits.fit_transform(logits_combined)
        
        # Cosine distance: Logits
        dist_logits = torch.nn.functional.cosine_similarity(logits1, logits2).item()

        plt.figure(figsize=(10, 8))
        plt.scatter(logits_2d[:len(logits1), 0], logits_2d[:len(logits1), 1], label=Path(audio_file1).name, s=100)
        plt.scatter(logits_2d[len(logits1):, 0], logits_2d[len(logits1):, 1], label=Path(audio_file2).name, s=100)
        plt.legend(fontsize=10)
        plt.title(f"t-SNE logits: {dist_logits:.2f} cosine distance", fontsize=14)
        
        # Convert plot to base64 for Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_logits = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Return four values: result1, result2, embeddings plot, logits_plot
        return result1, result2, f'<img src="data:image/png;base64,{plot_embeds}" alt="t-SNE Plots">', f'<img src="data:image/png;base64,{plot_logits}" alt="t-SNE Plots">'

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return (f"Error processing {Path(audio_file1).name}: {str(e)}", 
                f"Error processing {Path(audio_file2).name}: {str(e)}", 
                None, None)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Speaker Verification with WavLM")

    with gr.Row():
        audio1 = gr.Audio(value=FIL_WAV1, type="filepath", label="Audio File1", interactive=True)
        audio2 = gr.Audio(value=FIL_WAV2, type="filepath", label="Audio File2", interactive=True)

    with gr.Row():
        output1 = gr.Textbox(label="Example values: Embeddings (audio1)", lines=3)
        output2 = gr.Textbox(label="Example values: Logits (audio2)", lines=3)

    with gr.Row():
        out_plot_embeds = gr.HTML(label="t-SNE: Embeddings", elem_id="plot-embeds")
        out_plot_logits = gr.HTML(label="t-SNE: Logits", elem_id="plot-logits")

    # Add custom CSS to control plot size
    gr.Markdown("""
        <style>
        #plot-embeds img, #plot-logits img {
            max-width: 100%;
            height: auto;
            min-height: 400px;
            padding: 20px;
        }
        </style>
    """)

    process_btn = gr.Button("Process Audio Files")

    process_btn.click(
        fn=process_audio,
        inputs=[audio1, audio2],
        outputs=[output1, output2, out_plot_embeds, out_plot_logits]
    )

    demo.load(
        fn=process_audio,
        inputs=[audio1, audio2],
        outputs=[output1, output2, out_plot_embeds, out_plot_logits]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_api=True,
        show_error=True
    )
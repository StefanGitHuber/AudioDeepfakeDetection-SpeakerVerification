# AudioDeepfakeDetection-SpeakerVerification
Fine-tune WavLM for audio deepface detection and/or speaker verification

## 1_convert_wave_format.sh
Bash script to convert
from any speech recording (raw = whatever audio format it is)
to signed 16 Bit 16 kHz wave (since most speech models are trained on uncompressed waveforms).

Please install the awesome SoX (Sound eXchange) command line tool http://sox.sourceforge.net/
to perform simple sample rate conversion https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html


## 2_visualize_audio.py
Simple script to compare two speech recordings (cosine distance) using WavLM: 
- WavLM base tuned for Speaker Verification https://huggingface.co/microsoft/wavlm-base-sv
- Gradio interface visualizes embeddings and logits using t-SNE to reduce dimensionality for 2D scatter plots.
- HF transformer pipelines:
	Wav2Vec feature extractor encodes raw audio (resample, normalize, pad/truncate, attention mask);
	WavLM computes embeddings + logits for classification


## 3_voice_clone_coqui.py
Use Coqui TTS library's FreeVC24 model https://docs.coqui.ai/en/dev/ for voice conversion of original vs impostor audio files:
- Supports zero-shot VC, content preservation with speaker style transfer + natural prosody with fast inference.
- WavLM-based encoder removes speaker information while preserving content.
- ECAPA-TDNN extracts speaker embeddings.
- HiFiGAN decoder synthesizes content features, target speaker embedding and normalized pitch.
- SoX normalizes the converted speech waveform (adjusts magnitude).


## 4_classifier_audio.py
Examine deepfake audio detection using WavLM fine-tuned for sequence classification https://huggingface.co/microsoft/wavlm-base-sv:
- DeepfakeDetector class loads model + feature extractor, performs deepfake detection on set of original, impostor + fake audio files
-Instead of Speaker Verification with 100s/1000s of labels, put only two for the binary decision Real vs Fake

=> Super fails with worst accuracies fluctuating around 50 % like throwing a coin.
Analysis: Sure, model made for SV not spoofing detection concentrates on "pointers assuring speaker identity" (formants= vocal tract, prosodic features, pitch contour etc)


## 5_classifier_audio_finetuning.py
Adapt WavLM fine-tuned for Speaker Verification https://huggingface.co/microsoft/wavlm-base-sv:
- Detect audio deepfakes original + impostor real recordings vs fake synthesis audio.
- Fine-tuned model concentrates on everything else then speaker relevant info (formants, prosody) but subtle gaps + artefacts, especially phase discontinuities (unnatural phase shifts), oversmoothing (a mumbled but not clear audio room effect), unusual background noise patterns (most likely found in higher frequency regions) and other statistical irregularities (speaking rate + style).
- Create labels 0 for real recordings vs 1 for fake (TTS, VC).


## 6_speaker_verification.py
Performing deepfake audio detection is always closest related to speaker verification (if its a speech recording for a know person, e.g. in KYC). Specific head added to pre-trained WavLM lets WavLMForXVector generate x-vectors:
- Statistical pooling layer aggregates frame-level representations from WavLM into single utterance-level representation.
- Fully connected layers processes pooled representation to produce x-vectors.
- Compare x-vectors of two utterance to determine if same or different speaker.
- 2D plot of high-dimensional speaker embeddings (x-vectors) for visual analysis of speaker similarity.
- PCA (Principal Component Analysis) as linear dimensionality reduction technique.
 Writes a CSV file storing all cosine distance between evaluated speaker pairs (its embeddings compare).

Good old Kaldi: i-vector => d-vector => x-vector (i = identity, d=DNN, x=augmented).
One of my favourite papers of all time: 190930 Probing the Information Encoded in X-vectors, https://arxiv.org/abs/1909.06351
Basically in wonderful frank manners admitting that x-vectors do NOT serve perfectly its purpose (remove any redundant influence and only identify a speakers identity), but showing that x-vectors are highly correlated to (too) many other variation contained in a speech recording !!


## 7a_data_augmentation.py
Old script for simplest data augmentation to simulate audio signal detoriations/deviations:
- From an uncompressed (usually 16 Bit 16 kHz) waveform into phone call compression techniques (AMR narrow+wide-band, alaw 8 Bit 8 kHz) using ffmpeg
- Perturbations in speech rate using SoX (speed alters vocal tract filter, tempo maintains VTF)
- Add various levels in dB of artificial noise
- Simulate 3D room positions of microphone (sink) vs loudspeaker (source)


## 7b_finetune_triplet.py
A nowadays (year 2025) common contrastive loss trains a model to distinguish between similar + dissimilar speech segments:
- contrast different speakers saying same words/segments
- contrast same speaker saying different words/segments

In the context of deepfake detection and speaker similarity, a triplet loss is awesome:
- Anchor (A): Base data point.
- Positive (P): Data point similar to anchor.
- Negative (N): Data point dissimilar to anchor.
- L(A, P, N) = max(0, d(A, P) - d(A, N) + margin)

In this script I utilize data augmented speech signals as anchor (A),
in relation to their original recording before augmentation (P),
to distinct versus impostors approaches using TTS or VC (N).

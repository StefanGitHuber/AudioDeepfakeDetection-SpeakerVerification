#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use Coqui TTS library's FreeVC24 model https://docs.coqui.ai/en/dev/ for voice conversion of original vs impostor audio files.
Supports zero-shot VC, content preservation with speaker style transfer + natural prosody with fast inference.
WavLM-based encoder removes speaker information while preserving content.
ECAPA-TDNN extracts speaker embeddings.
HiFiGAN decoder synthesizes content features, target speaker embedding and normalized pitch.
SoX normalizes the converted speech waveform (adjusts magnitude).
"""

import glob
import os
import subprocess

from pathlib import Path
from TTS.api import TTS

# Coqui TTS: Free voice conversion model https://docs.coqui.ai/en/dev/
MODEL_NAME = "voice_conversion_models/multilingual/vctk/freevc24"
tts = TTS(model_name=MODEL_NAME, progress_bar=True)

# Path handling: Originals vs impostors fake each other
b_fake2origin = True
DIR_BASIS = Path("/your/path/to/")
DIR_ORIGS = DIR_BASIS / "data_orig" / "*.wav"
DIR_IMPOS = DIR_BASIS / "data_impostors" / "*.wav"
if b_fake2origin:
    DIR_FAKES = DIR_BASIS / "data_fakes_fake2orig" 
    os.makedirs(str(DIR_FAKES), exist_ok=True)
else:
    DIR_FAKES = DIR_BASIS / "data_fakes_orig2fake"
    os.makedirs(str(DIR_FAKES), exist_ok=True)

for wav_orig in glob.glob(str(DIR_ORIGS)):
    for wav_fake in glob.glob(str(DIR_IMPOS)):
        
        if b_fake2origin:
            # Fake impostors into Origins
            wav_conv = os.path.join(DIR_FAKES, os.path.basename(wav_fake)[:-4] + "_2_" + os.path.basename(wav_orig)[:-4] + "-VC.wav")
            tts.voice_conversion_to_file(source_wav=wav_fake, target_wav=wav_orig, file_path=wav_conv)
        else:
            # Origins fake into impostors
            wav_conv = os.path.join(DIR_FAKES, os.path.basename(wav_orig)[:-4] + "_2_" + os.path.basename(wav_fake)[:-4] + "-VC.wav")
            tts.voice_conversion_to_file(source_wav=wav_orig, target_wav=wav_fake, file_path=wav_conv)

        # Normalize magnitude
        wav_norm = os.path.join(DIR_FAKES, os.path.basename(wav_conv)[:-4] + "-norm.wav")
        cmd = "sox " + wav_conv + " --norm " + wav_norm
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        identify_info, unused_err_output = process.communicate()

        # Remove temp file VC, keep final VC + normalized waveform
        cmd = "rm " + wav_conv
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        identify_info, unused_err_output = process.communicate()
        
        print("Converted from %s to %s" % (wav_orig[:-4], wav_fake[:-4]))

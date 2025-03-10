#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Old script for simplest data augmentation to simulate audio signal detoriations/deviations:
- From an uncompressed (usually 16 Bit 16 kHz) waveform into phone call compression techniques (AMR narrow+wide-band, alaw 8 Bit 8 kHz) using ffmpeg
- Perturbations in speech rate using SoX (speed alters vocal tract filter, tempo maintains VTF)
- Add various levels in dB of artificial noise
- Simulate 3D room positions of microphone (sink) vs loudspeaker (source)
"""

import copy
import csv
import subprocess
import wave

bin_codecs  = True
bin_speed   = True
bin_artnois = True

speed_factors = ['0.67', '1.33']            # Speed & tempo factors
noise_levels  = ['0.001', '0.05', '0.2']    # Noise levels in dB

dir_basis = "/your/path/to/data_orig/"
dir_augm = "/your/path/to/data_augm/"
wave_orig = dir_basis + "my_orig.wav"
wave_augm = dir_augm + "my_augm.wav"

### Speech codecs
wave_8ksr = wave_augm + '_8ksr.wav'
g711_wave = wave_augm + '_g711.wav'
AMRWB_wav = file_file + '_AMR-WB-hi.wav'
AMRWB_amr = file_file + '_AMR-WB-hi.amr'
AMRNB_wav = file_file + '_AMR-NB-lo.wav'
AMRNB_amr = file_file + '_AMR-NB-lo.amr'

wavs_augm = [] # List of augmented speech phrases

wavs_augm.append(wave_8ksr)
wavs_augm.append(g711_wave)

###### Original wave downsampled to 8k sample rate ######
system_call = 'sox ' + wave_orig + ' -r 8000 ' + wave_8ksr
print('%s\n' % system_call)
subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

###### Simulate speech compression codecs ######

if bin_codecs:

    ### G711:
    # ffmpeg -i input.wav -c:a pcm_alaw input-alaw.wav
    system_call = 'ffmpeg -loglevel 1 -i ' + wave_orig + ' -c:a pcm_alaw ' + wave_augm + '-alaw.wav'
    print('%s\n' % system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    # sox input-alaw.wav -r 8000 input-alaw-8k.wav
    system_call = 'sox ' + wave_augm + '-alaw.wav' + ' -r 8000 ' + g711_wave
    print('%s\n' % system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    system_call = 'rm ' + wave_augm + '-alaw.wav'
    print('%s\n' % system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    ### g722.2 AMR-WB 16 kHz - Highest bitrate 23.85k:
    # ffmpeg -i $inp_wav -ar 16000 -ab 23.85k -acodec libvo_amrwbenc -loglevel 0 $inp_nam-hi-AMR-WB.amr
    system_call = 'ffmpeg -i ' + wave_orig + ' -ar 16000 -ab 23.85k -acodec libvo_amrwbenc -loglevel 2 ' + AMRWB_amr
    print('%s', system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    # sox $inp_nam-hi-AMR-WB.amr $inp_nam-hi-AMR-WB.wav
    system_call = 'sox ' + AMRWB_amr + ' ' + AMRWB_wav
    print('%s', system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    ### g722.2 AMR-NB 8 kHz - Highest bitrate 12.20k:
    # ffmpeg -i $inp_wav -ar 8000 -ab 12.20k -acodec libopencore_amrnb -loglevel 0 $inp_nam-hi-AMR-NB.amr
    system_call = 'ffmpeg -i ' + wave_orig + ' -ar 8000 -ab 12.20k -acodec libopencore_amrnb -loglevel 2 ' + AMRNB_amr
    print('%s', system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

    # sox $inp_nam-hi-AMR-NB.amr $inp_nam-hi-AMR-NB.wav
    system_call = 'sox ' + AMRNB_amr + ' ' + AMRNB_wav
    print('%s', system_call)
    subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

###### Simulate speed & tempo of speaking ######

# https://stackoverflow.com/questions/33957747/how-do-i-reduce-the-speed-of-a-voice-mp3-file-with-sox-to-75/33957968
# sox input.wav output.wav speed 1.33 (speed alters pitch)
# sox input.wav output.wav tempo 1.33 (tempo maintains pitch)
#
# https://groups.google.com/g/kaldi-help/c/8OOG7eE4sZ8?pli=1
# Kaldi uses speech not tempo cause "it's more helpful for performance"

### Simulate speaking rate, multiply speed & tempo factors
if bin_speed:
    temp_wavs = copy.deepcopy(wavs_augm)
    for wave_file in temp_wavs:
        for factor in speed_factors:
            # Construct wave names + add to augmented list
            speed_wave = wave_file[:-4] + '_speed' + factor + '.wav'
            tempo_wave = wave_file[:-4] + '_tempo' + factor + '.wav'
            wavs_augm.append(speed_wave)
            wavs_augm.append(tempo_wave)

            # sox $inp $out_speed speed 1.33
            system_call = 'sox ' + wave_file + ' ' + speed_wave + ' speed ' + factor
            print('%s\n' % system_call)
            subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

            # sox $inp $out_tempo tempo 1.33
            system_call = 'sox ' + wave_file + ' ' + tempo_wave + ' tempo ' + factor
            print('%s\n' % system_call)
            subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

###### Add artificially generated noise ######

# sox -n -b 16 whitenoise.wav synth 10.0 whitenoise, brownnoise, pinknoise
# sox $in_wav -p synth whitenoise vol 0.2 | sox -m $in_wav - 1688-142285-0000-addednoise.wav

temp_wavs = copy.deepcopy(wavs_augm)
for wave_file in temp_wavs:
    for level in noise_levels:
        # Construct wave names + add to augmented list
        white_wave = wave_file[:-4] + '_whitenoise' + level + '.wav'
        brown_wave = wave_file[:-4] + '_brownnoise' + level + '.wav'
        pinki_wave = wave_file[:-4] + '_pinknoise'  + level + '.wav'
        wavs_augm.append(white_wave)
        wavs_augm.append(brown_wave)
        wavs_augm.append(pinki_wave)

        if bin_artnois:
            # sox $in.wav -p synth whitenoise vol 0.2 | sox -m $in_wav - $out.wav
            system_call = 'sox ' + wave_file + ' -p synth whitenoise vol ' + level + ' | sox -m ' + wave_file + ' - ' + white_wave
            print('%s\n' % system_call)
            subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

            system_call = 'sox ' + wave_file + ' -p synth brownnoise vol ' + level + ' | sox -m ' + wave_file + ' - ' + brown_wave
            print('%s\n' % system_call)
            subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

            system_call = 'sox ' + wave_file + ' -p synth pinknoise vol '  + level + ' | sox -m ' + wave_file + ' - ' + pinki_wave
            print('%s\n' % system_call)
            subprocess.run(system_call, shell=True, stdout=subprocess.PIPE)

if 0:
    ###### Add far field speaker ######
    def add_filt_csv_entries(csv_file, speakers):
    # Open csv file, read line-wise 2nd column (row[1]), add new entry to list speakers (if not yet added)
        with open(csv_file) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if row[1] not in speakers:
                speakers.append(row[1])
        return speakers
    
    # Find in LibriSpeech files of similar length
    temp_wavs = copy.deepcopy(wavs_augm)
    for wave_file in temp_wavs:
    
        # Put speakers record from near -> far field: wave_inp -> wave_out
        far_field_room(wave_inp, wave_out)

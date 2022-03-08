import random
import os
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from pathlib import Path
from hparams import create_hparams


hparams = create_hparams() 

load_mel_from_disk = hparams.load_mel_from_disk
filter_mel =  hparams.filter_mel
mel_files = hparams.mel_files
text_cleaners = hparams.text_cleaners
max_wav_value = hparams.max_wav_value
sampling_rate = hparams.sampling_rate
#BL:added by me 
max_output_length = hparams.max_output_length
bl_counter = 0
# BL: end added
sampling_rate = hparams.sampling_rate
stft = layers.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)

    
def get_loop():    
    mypath = os.getcwd()
    full_input = os.path.join(mypath, "phrases/Gerard_fichiers/candidats_for_wavs_mel/")
    # Path(full_input).rglob('*.wav')
    mel_dir=os.path.join(os.getcwd(), "phrases/Gerard_fichiers/mels_candidats_for_wavs_mel")
    os.makedirs(mel_dir, exist_ok=True)
    for path in Path(full_input).rglob("*.wav"):
        ##ADD RELATIVE PATH
        # path_parent_relative_list = str(path.parent).strip().split("phrases/Gerard_fichiers/audio_output_gerard")
        path_parent_relative_list = str(path.name)
        print(path_parent_relative_list)
        #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
        #take the last part ES_LMP_NEB_02_0011_19
        basename = path_parent_relative_list
        audio_path = os.path.join(full_input, path_parent_relative_list)
        audio, sampling_rate = load_wav_to_torch(audio_path)
        if sampling_rate != 22050:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, stft.sampling_rate))
        audio_norm = audio / max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        mel_filename = 'mel-{}.npy'.format(basename)
        np.save(os.path.join(mel_dir, mel_filename), melspec, allow_pickle=False)
        print("mel file {} saved in {}".format(mel_filename, mel_dir))

 
def main(): 
    get_loop()
    
# Driver code   
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 
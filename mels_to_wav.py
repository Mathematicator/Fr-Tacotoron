import torch
import numpy as np
import sys
import os
from pathlib import Path
import argparse

sys.path.append('waveglow/')

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from waveglow.denoiser import Denoiser
# import stft 
import IPython.display as ipd
from scipy.io import wavfile



def waveglow_load(waveglow_path):
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser
    
    
def prepare_params(input, output, waveglow_path, mode):
    hparams = create_hparams()
    hparams.max_decoder_steps=2000
    hparams.sampling_rate = 22050
    
    if input is not None :
        input_dir = input
        
    if output is not None :
        output_dir= output
    
    #####
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
    stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)
    
    np_name = load_mels(input, mode, hparams, stft, stft_fn, output_dir,waveglow_path)
    
    
def load_mels(input, mode, hparams, stft, stft_fn, output_dir,waveglow_path):
    mypath = os.getcwd()
    full_input = os.path.join(mypath, input)
    
    # numpy_vars = {}
    # torch_vars = {}
    
    for path in Path(full_input).rglob('*.npy'):
    ##ADD RELATIVE PATH
        path_parent_relative_list = str(path.parent).strip().split(input)
        if path_parent_relative_list[1] is not None :
            path_parent_relative = path_parent_relative_list[1]
            output_dir_absolue=os.path.join(output_dir ,path_parent_relative)
            output_dir_final=os.path.join(os.getcwd() , output_dir_absolue)
        else :
            output_dir_final=os.path.join(os.getcwd() , output_dir)
        os.makedirs(output_dir_final, exist_ok=True)
        
        if mode == "g+w" or mode == "both" or mode == "waveglow" or mode == "w":
            waveglow, denoiser = waveglow_load(waveglow_path)
        
        if mode == "g" or mode == "griffin&lim":
            # print(path.name)
            # numpy_vars[path] = np.load(path)
            torch_vars = torch.from_numpy(np.load(path))
            griffin_lim_syn(hparams, stft, stft_fn, torch_vars, path.name, output_dir_final )
                
        if mode == "both" or mode == "g+w":
            # print(path.name)
            # numpy_vars[path] = np.load(path)
            torch_vars = torch.from_numpy(np.load(path))
            griffin_lim_syn(hparams, stft, stft_fn, torch_vars, path.name, output_dir_final ) 
            waveglow_infer(hparams, waveglow, denoiser, torch_vars, path.name, output_dir_final)    
                
        if mode == "w" or mode == "waveglow":
            # numpy_vars[path] = np.load(path)
            torch_vars = torch.from_numpy(np.load(path))
            waveglow_infer(hparams, waveglow, denoiser, torch_vars, path.name, str(path.parent), output_dir_final)

    return 0
        
    
    
def griffin_lim_syn(hparams, stft, stft_fn, torch_vars, path_name, output ):

    loaded_predict_mel_dest = torch_vars.unsqueeze(0)
    #print(mel_outputs_postnet.float().data.cpu().shape)

    inverse_transform = stft.mel_inv_spectrogram(loaded_predict_mel_dest)
    audio =griffin_lim(inverse_transform,stft_fn)

    ipd.Audio(audio.data.cpu().numpy(), rate=22050)
    #audio_grif = audio.data.cpu().numpy()


    save_wav(audio[0].data.cpu().numpy(), os.path.join(output, 'wav-griffin-{}-.wav'.format(path_name)), sr=hparams.sampling_rate)
    print('wav-griffin-{}-.wav saved in {}'.format(path_name, output))

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def waveglow_infer(hparams, waveglow, denoiser,  torch_vars, path_name, path_parent, output):
    #generate audio
    torch_vars = torch_vars.unsqueeze(0).to(torch.device('cuda'),torch.half, non_blocking =True, copy=False )
    with torch.no_grad():
        audio = waveglow.infer(torch_vars, sigma=0.6)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    #save audio
    audio_denoised = denoiser(audio, strength=0.05)[:, 0]
    ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
    
    
    #save audio
    save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(output, 'wav-waveglow-ljs-{}-.wav'.format(path_name)), sr=22050)  
    print('wav-waveglow-{}-.wav saved in {}'.format(path_name, output))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MELS_To_WAV')
    accepted_modes = {'g','griffin&lim', 'w','waveglow', 'g+w', 'both'}
    parser.add_argument('-i', '--input', type=str, default= 'mels-DTW',
                        help='full path to the csv file (pharases separated by | and new line)')
    parser.add_argument('-o', '--output',  default= 'waves_output',
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('-w', '--waveglow', type=str, default= 'waveglow_256channels_ljs_v2_new.pt',
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-m', '--mode', type=str, required= True,
                        help='mode of run: can be one of {}'.format(accepted_modes))
    args, _ = parser.parse_known_args()
    if args.mode not in accepted_modes :
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))
    prepare_params(args.input, args.output, args.waveglow, args.mode)
    
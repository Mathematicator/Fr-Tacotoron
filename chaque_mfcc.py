from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader

import multiprocessing
multiprocessing.set_start_method('spawn', True)
import os 
print(os.getcwd())
import sys
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

matplotlib.use("Agg")
matplotlib.rcParams['agg.path.chunksize'] = 10000000000000

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser
import random
import tensorflow as tf
import stft 

#for loss
from torch import nn
import torch.nn.functional as F

hparams = create_hparams()
hparams.max_decoder_steps=1000
hparams.sampling_rate = 22050
hparams.batch_size = 2
#hparams.training_files = 'FR-fr_Our/test_olivier_onf_150_973'
data_name = hparams.training_files.strip().split('/')
data_name =data_name[-1]


#load model
checkpoint_path = "./checkpoint_99000_idris"
checkpoint_name = checkpoint_path.strip().split('/')
checkpoint_name = checkpoint_name[-1]
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


# del load_mels:
#     with open(hparams.training_files, encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('|')
#             path = parts[0]        
#             parts = path.strip().split('/')
#             basename = parts[-1]
#             mel_filename = '{}mel-{}.npy'.format(self.mel_files,basename)
#             melspec_original = torch.from_numpy(np.load(mel_filename))


# #### Prepare TEXT split ###***CHANGE IN MODEL***###
def get_mel_frame(hparams):
    with open(hparams.training_files, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('|')
            text = parts[1]
            #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
            text_Npath =  parts[0].strip().split('/')
            #take the last part S_LMP_NEB_02_0011_19
            basename = text_Npath[-1]
            mel_filename = '{}mel-{}.npy'.format(hparams.mel_files, basename)
            melspec_original = torch.from_numpy(np.load(mel_filename))
            melspec_original = torch.squeeze(melspec_original, 0)  
            melspec_original = melspec_original.detach().cpu().numpy()
            sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
            #print(text) 
            sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence, hparams)
            #CB-BL: by me save the original duration
            ###***CHANGE Name***###
            
            _dir = os.path.join(os.getcwd(), 'bissectrice-eval_'+ checkpoint_name + data_name)
            ###***CHANGE Name***###
            mel_outputs_postnet = torch.squeeze(mel_outputs_postnet, 0)
            mel_outputs_postnet_np = mel_outputs_postnet.detach().cpu().numpy()
            for index in range (melspec_original.shape[0]):
                for j in range(min(len(melspec_original[0, :]), len(mel_outputs_postnet_np[0, :]))):
                    file_original = open('original_mel_frames_length.txt'+ str(index), "a+", encoding="utf-8")
                    file_original.write(str(melspec_original[index, :][j]) + "\n")
                    file = open('predicted_mel_frames_length.txt'+ str(index), "a+", encoding="utf-8")
                    file.write(str(mel_outputs_postnet_np[index, :][j]) + "\n")
                    print("Original me frame length : {}  Predicted mel frame length: {}".format(melspec_original[index, :][j], (mel_outputs_postnet_np[index, :][j])))
            #save mel spectrogram plot
            # plot_spectrogram(mel_outputs.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-{}.png'.format(basename)),
            #     title='{}'.format(text), split_title=True)  
            #save mel_outputs_postnet plot
            # plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}.png'.format(basename)),
            #     title='mel postnet \r \n {}'.format(text), split_title=True)  
            # print(mel_outputs_postnet.detach().cpu().numpy().shape[2])

            #save alignments
            # plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}.png'.format(basename)),
            #     title='{}'.format(text), split_title=True)
            #save gates 
            # idx = random.randint(0, alignments.size(0) - 1)
            # plot_gate_outputs_to_numpy(
            #     torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(basename)),
            #     title='{}'.format(text), split_title=True)
            #generate audio
            # with torch.no_grad():
            #     audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
            # ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
            # #save audio
            # audio_denoised = denoiser(audio, strength=0.05)[:, 0]
            # ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
            # #generate plot audio
            # plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}.png'.format(basename)),
            #     title='audio_denoised \r \n {}'.format(text), split_title=True)
            # #save audio
            # save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-waveglow-ljs-{}-.wav'.format(basename,split_title_line(text[-50:-1], max_words=1))), sr=22050)     
            # inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
            # audio =griffin_lim(inverse_transform,stft_fn)
            # save_wav(audio[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-griffin-{}-.wav'.format(basename,split_title_line(text[-50:-1], max_words=1))), sr=hparams.sampling_rate)
            #plot_data((mel_outputs.float().data.cpu().numpy()[0],
                #mel_outputs_postnet.float().data.cpu().numpy()[0],
                #alignments.float().data.cpu().numpy()[0].T))
    file.close()
    file_original.close()    
            #idx = random.randint(0, alignments.size(0) - 1)
            #plot_gate_outputs_to_numpy(
                #torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
    #return mel_outputs_postnet

def plot_mel_frames_length():
    color = iter(cm.rainbow(np.linspace(0, 1, 80)))
    print(matplotlib.rcParams['agg.path.chunksize'])
    
    for index in range(80):
        if index == 33:
            index += 1
        X, Y = [], []
        #for line in open('original_duration_{}_{}_{}.txt'.format(state,data,checkpoint_name)):
        for line in open('original_mel_frames_length.txt'+ str(index)):
            values = [float(s) for s in line.split()]
            X.append(values[0])


        for line in open('predicted_mel_frames_length.txt'+ str(index)):
            values = [float(s) for s in line.split()]
            Y.append(values[0])
        if len(Y) > 30000 :
            Y = Y[:30000]
            X = X[:30000]

        if len(Y) == len(X):
            print("The shape of X {} , the shape of Y {}".format(len(X), len(Y)))
            fig = plt.figure(figsize=(16, 10))
            plt.xlabel("Mel original frames length MFCC N:"+ str(index))
            plt.ylabel("Mel predicted frames length MFCC N:"+ str(index))
            plt.tight_layout()
            plt.title("Mel frames length originales vs predites MFCC N:"+ str(index))
            fig.canvas.draw()
            c = next(color)
            plt.scatter(X, Y, c=c)
            x = np.linspace(-10 ,1, 20)
            y1 = x
            plt.plot(x,y1)
            plt.savefig(os.path.join(os.getcwd(),'originales_vs_predites_mel_length_frames_MFCC_N'+ str(index)+'.eps'), format='eps')
            plt.close()
        else:
            print("ERROR SIZE #DIFFERENT The shape of X {} , the shape of Y {}".format(len(X), len(Y)))

if __name__ == '__main__':
    #get_mel_frame(hparams)
    plot_mel_frames_length()
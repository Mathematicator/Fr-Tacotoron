from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader

import multiprocessing
multiprocessing.set_start_method('spawn', True)
import os 
print(os.getcwd())
import sys
import matplotlib
import matplotlib.pylab as plt

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


def get_outputs(hparams):
    with open(hparams.training_files, encoding="utf-8") as f:
        mel_outputs_list, mel_outputs_postnet_list, gate_outputs_list, alignments_list, mel_original_list = [], [], [], [], []
        rigth=[]
        for line in f:
            parts = line.strip().split('|')
            #get the text part easy
            text_original = parts[1]
            #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
            text_Npath =  parts[0].strip().split('/')
            #take the last part S_LMP_NEB_02_0011_19
            basename = text_Npath[-1]
            mel_original = np.load('./mels/mel-{}.npy'.format(basename))
            print("inference of {}".format(text_original))
            sequence = np.array(text_to_sequence(text_original, ['transliteration_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence,hparams)
                mel_outputs_list.append(mel_outputs)
                mel_outputs_postnet_list.append(mel_outputs_postnet)
                mel_original_list.append(mel_original)
            #delete model in cache
            mel_outputs.to('cpu')
            del mel_outputs
            mel_outputs_postnet.to('cpu')
            del mel_outputs_postnet
            #torch.cuda.empty_cache()
    return mel_outputs_list, mel_outputs_postnet_list, mel_original_list

def loss(mel_outputs_list, mel_outputs_postnet_list, mel_original_list):
    for i in range(len(mel_outputs_list)):
        #gate_target = gate_target.view(-1, 1)

        # mel_out, mel_out_postnet, gate_out, _ = model_output
        #gate_out = gate_out.view(-1, 1)
        mel_original_torch = torch.from_numpy(mel_original_list[i]).cuda().half()
        mel_outputs_list[i] = torch.squeeze(mel_outputs_list[i], 0)
        mel_outputs_postnet_list[i] = torch.squeeze(mel_outputs_postnet_list[i], 0)

        if len(mel_outputs_list[i][0]) > len(mel_original_torch[0]):
            print("passed by IF ****************")
            mel_outputs_list[i] = mel_outputs_list[i][ :, :len(mel_original_torch[0])] 
            mel_outputs_postnet_list[i]= mel_outputs_postnet_list[i][:, :len(mel_original_torch[0])] 
            
        else:
            mel_original_torch = mel_original_torch[ :, :len(mel_outputs_list[i][0])] 
            print("passed by else ################")
        print("mel out : {} ,  mel postnet : {}, mel original  {} ".format(mel_outputs_list[i].shape, mel_outputs_postnet_list[i].shape, mel_original_torch.shape))
        print ("loss mel out : {} , loss mel postnet : {}  ".format(nn.MSELoss()(mel_outputs_list[i], mel_original_torch), nn.MSELoss()(mel_outputs_postnet_list[i], mel_original_torch)))
        mel_loss = nn.MSELoss()(mel_outputs_list[i], mel_original_torch) + \
            nn.MSELoss()(mel_outputs_postnet_list[i], mel_original_torch)
        file = open('mel_loss_inference.txt', "a+", encoding="utf-8")
        file.write(str(mel_loss.float().data.cpu().numpy())+"\n")
        #return mel_loss 

def plot_mel_loss():
    X, Y = [], []
    #for line in open('original_duration_{}_{}_{}.txt'.format(state,data,checkpoint_name)):
    for line in open('mel_loss_train.txt'):
      values = [float(s) for s in line.split()]
      X.append(values[0])


    for line in open('mel_loss_inference.txt'):
      values = [float(s) for s in line.split()]
      Y.append(values[0])

    Y=Y[:len(X)]
    print("The shape of X {} , the shape of Y {}".format(len(X), len(Y)))
    fig = plt.figure(figsize=(16, 10))
    plt.xlabel("mel loss train")
    plt.ylabel("Mel loss inférence ")
    plt.tight_layout()
    plt.title(" loss pour le système en mode (Train Vs inférence)")
    fig.canvas.draw()

    plt.plot(X, Y,'o')
    plt.savefig(os.path.join(os.getcwd(),'comparaison-mel_loss_mel_inference.png'), format='png')
    plt.close()


if __name__ == '__main__':
    # mel_outputs_list, mel_outputs_postnet_list, mel_original_list = get_outputs(hparams)
    # loss(mel_outputs_list, mel_outputs_postnet_list, mel_original_list)
    plot_mel_loss()





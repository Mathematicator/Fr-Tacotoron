# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  ## Compare mel-spectrogram
#  
# %% [markdown]
#  #### Import libraries and setup matplotlib

# %%
import matplotlib
import torch 
from torch.autograd import Variable
from torch import nn 
from torch.nn import functional as F
#matplotlib inline
import matplotlib.pylab as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
import math
import os 
os.chdir('/media/lebbat/lebbat/NVIDIA-official/tacotron2_gricad')


# %%
## Load data
#filename ='./mels-test/'
basename = 'ES_LMP_NEB_01_0010_55'

mel_original = np.load('./mels-test/mel-original_{}.npy'.format(basename))
mel_predicted_inference = np.load('./mels-test/mel-predicted_{}.npy'.format(basename))
mel_predicted_train = np.load('./Train_Inside_Visualize/mel_save_dir/mel-0.562820553779602-0-0-0.npy')
mel_predicted_valid = np.load('./Valid_Inside_Visualize/mel_save_dir/mel-0.7791593670845032-0-0-0.npy')


x = mel_original
y = mel_predicted_inference

Nx = mel_original.shape[1];
Ny = mel_predicted_inference.shape[1];
Nmel = mel_original.shape[0]

print("mel_original.shape:{} mel_predicted_inference:{} mel_predicted_train: {} mel_predicted_valid: {}".format(mel_original.shape, mel_predicted_inference.shape, mel_predicted_train.shape, mel_predicted_valid.shape))

# %%
def dist_eculidian_BL(x,y):
    d = math.sqrt(sum((x-y)**2))
    return d



# %%
def dist_MSE_BL(x,y):
    d =((x-y)**2).mean()
    return d

# %%
#Compute DTW 
# np.linalg.norm equal to the euclidian_dist default norm =2 ==> 

M_dtw = np.zeros((Nx,Ny))
M_dtw_old = np.zeros((Nx,Ny))


print(M_dtw.shape)
for i in range(Nx):
    for j in range(Ny):
        M_dtw[i,j] = dist_eculidian_BL(x[:,i],y[:,j])
        M_dtw_old[i,j] = np.linalg.norm(x[:,i]-y[:,j])

print(M_dtw)
print("***********")
print(M_dtw_old)


# %%
def minCost(cost, m, n): 
  
    R, C = cost.shape
    # Instead of following line, we can use int tc[m + 1][n + 1] or 
    # dynamically allocate memoery to save space. The following 
    # line is used to keep te program simple and make it working 
    # on all compilers. 
    tc = [[0 for x in range(C)] for x in range(R)] 
  
    tc[0][0] = cost[0][0] 
  
    # Initialize first column of total cost(tc) array 
    for i in range(1, m + 1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
  
    # Initialize first row of tc array 
    for j in range(1, n + 1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
  
    # Construct rest of the tc array 
    for i in range(1, m + 1): 
        for j in range(1, n + 1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], 
                            tc[i][j-1]) + cost[i][j] 
    return tc[m][n] 


# %%
R = Nx
C = Ny
minCost(M_dtw,Nx-1,Ny-1)


# %%
# Compute the path through the distance matrix
[p,q] = shortest_path(M_dtw);
Nl = length(p);


# %%
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                    interpolation='none')
# %% [markdown]


# %%
# #### Prepare TEXT split ###***CHANGE IN MODEL***###
# #### Prepare TEXT split ###***CHANGE IN MODEL***###
# #### Prepare TEXT split ###***CHANGE IN MODEL***###

import matplotlib
import warnings
warnings.filterwarnings('ignore')

#matplotlib inline
import matplotlib.pylab as plt
## for compute y 
from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader

import multiprocessing
multiprocessing.set_start_method('spawn', True)



import IPython.display as ipd
import os 
#os.chdir('/research/crissp/lebbat/NVIDIA-official/tacotron2/')
print(os.getcwd())
import sys
sys.path.append('waveglow/')
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
#import audio
import tensorflow as tf
##
import stft 


#torch.manual_seed(0)


# %%
def split_title_line(title_text, max_words=10):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])
    # %% [markdown]


# %%
def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower', 
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    # To NOT SHOW
    plt.close()

    #%% [markdown]


# %%
def plot_wave(predicted_wave, path=None, title=None, split_title=False, target_wave=None, max_len=None, auto_aspect=False):
    if max_len is not None:
        target_wave = target_wave[:max_len]
        predicted_wave = predicted_wave[:max_len]
    fig, ax = plt.subplots(figsize=(20, 8))
        #fig = plt.figure(figsize=(8, 6)) # it was(10, 8)
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    center = left + width/2
    # Set common labels
    # set title 
    if title is not None:
        if split_title:
            title = split_title_line(title) 
        fig.text(center, top,title, horizontalalignment='center',verticalalignment='center', fontsize=16)
        
    #plt.figure()
    plt.plot(predicted_wave)

    if path is not None :
        plt.savefig(path, format='png')
        # To NOT SHOW
    plt.close()


# %%
def plot_spectrogram(pred_spectrogram, path=None, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if split_title:
        title = split_title_line(title)

    fig = plt.figure(figsize=(8, 6)) # it was(10, 8)
    # Set common labels
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

    #target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram), aspect='auto', interpolation='none')
        else:
            im = ax1.imshow(np.rot90(target_spectrogram), interpolation='none')
        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211) # it was add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(pred_spectrogram, aspect='auto', origin='bottom', interpolation='none')
    else:
        im = ax2.imshow(pred_spectrogram, aspect='auto', origin='bottom', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    if path is not None :
        plt.savefig(path, format='png')
        # To NOT SHOW
        plt.close()



# %%
def plot_gate_outputs_to_numpy(gate_outputs,path=None, title=None, split_title=False):
    fig, ax = plt.subplots(figsize=(12, 3))
        #fig = plt.figure(figsize=(8, 6)) # it was(10, 8)
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    center = left + width/2
    # Set common labels
    # set title 
    if title is not None:
        if split_title:
            title = split_title_line(title) 
        fig.text(center, top,title, horizontalalignment='center',verticalalignment='center', fontsize=16)
    #ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.7,
            #color='green', marker='+', s=3, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=1,
            color='red', marker='.', s=10, label='predicted')

    plt.xlabel("Frames ( Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    
    if path is not None :
        plt.savefig(path, format='png')
        # To NOT SHOW
        plt.close()
    #plt.show()    
    

# %% [markdown]
# #### Setup hparams


# %%
hparams = create_hparams()
hparams.max_decoder_steps=2000
hparams.sampling_rate = 22050
hparams.batch_size = 2
#hparams.training_files = 'FR-fr_Our/HAPPYneuron.xlsx'
data_name = hparams.training_files.strip().split('/')
data_name =data_name[-1]

# %% [markdown]
# #### Load model from checkpoint


# %%
#checkpoint_path = "/research/crissp/lebbat/Tacotron_Synthesis/Tacotron-2-master/Tacotron-2-master/logs-Tacotron-2/taco_pretrained/100k/tacotron_model.ckpt-100000.data-00000-of-00001"
checkpoint_path = "./fr_from_eng_67000.pt"
#checkpoint_path = "/research/crissp/lebbat/NVIDIA-official/tacotron2_gpu2/checkpoint_99000_idris"
#checkpoint_path = "/research/crissp/lebbat/NVIDIA-official/tacotron2_gpu2/out/checkpoint_190000"
#checkpoint_path = "checkpoint_16000"
#checkpoint_path = "tacotron2_statedict.pt"
checkpoint_name = checkpoint_path.strip().split('/')
checkpoint_name = checkpoint_name[-1]
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

#random.seed(0)
#np.random.seed(0)
# train_dir = os.path.join(os.getcwd(), 'train-eval_'+ checkpoint_name + data_name)
# val_dir = os.path.join(os.getcwd(), 'val-eval_'+ checkpoint_name + data_name)
# test_dir = os.path.join(os.getcwd(), 'test-eval_'+ checkpoint_name + data_name)
# difficile_dir = os.path.join(os.getcwd(), 'difficile-eval_'+ checkpoint_name + data_name)

# plot_train_dir = os.path.join(train_dir, 'plots')
# plot_val_dir = os.path.join(val_dir, 'plots')
# plot_test_dir = os.path.join(test_dir, 'plots')
# plot_difficile_dir = os.path.join(difficile_dir, 'plots')

# wav_dir_sTrain = os.path.join(train_dir, 'wavs')
# wav_dir_sVal = os.path.join(val_dir, 'wavs')
# wav_dir_sTest = os.path.join(test_dir, 'wavs')
# wav_dir_difficile = os.path.join(difficile_dir, 'wavs')


# os.makedirs(plot_train_dir, exist_ok=True)
# os.makedirs(plot_val_dir, exist_ok=True)
# os.makedirs(plot_test_dir, exist_ok=True)
# os.makedirs(plot_difficile_dir, exist_ok=True)

# os.makedirs(wav_dir_sTrain, exist_ok=True)
# os.makedirs(wav_dir_sVal, exist_ok=True)
# os.makedirs(wav_dir_sTest, exist_ok=True)
# os.makedirs(wav_dir_difficile, exist_ok=True)

ponctuation_dir = os.path.join(os.getcwd(), 'ponctuation-eval_'+ checkpoint_name + data_name)
plot_ponctiation_dir = os.path.join(ponctuation_dir, 'plots')
wav_dir_ponctiation = os.path.join(ponctuation_dir, 'wavs')

# os.makedirs(ponctuation_dir, exist_ok=True)
# os.makedirs(plot_ponctiation_dir, exist_ok=True)
# os.makedirs(wav_dir_ponctiation, exist_ok=True)

# %% [markdown]
# #### %% [markdown]
# #### Load WaveGlow for mel2audio synthesis and denoiser
# 
# 
# 
# 

# %%
waveglow_path = 'waveglow_256channels_ljs_v2_new.pt'
#waveglow_path = 'waveglow/checkpoints/waveglow_50000'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


# %%
def prepare_dataloaders(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                            sampler=None,
                            batch_size=hparams.batch_size, pin_memory=False,
                            drop_last=False, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


# %%
from scipy.io import wavfile
def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))


# %%
ponctuations = ['!', '\"',  '(',  ')',  ',',  '.',  ':', ';', '?', '«', '»', '—', '…', '-', '[', ']']
len(ponctuations)


# %%
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)


# %%
def dtwPlotThreeWay_BL(d=None, xts=None, yts=None,title=None,path=None, split_title=False,
                    match_indices=None,
                    match_col="gray",
                    xlab="Query index",
                    ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotThreeWay
    
    # ENDIMPORT
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import collections  as mc
    


    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required")

    nn = len(xts)
    mm = len(yts)
    nn1 = np.arange(nn)
    mm1 = np.arange(mm)

    fig = plt.figure()
    
#         #BL:added 
#     if split_title is not None:
#         title = split_title_line(title)
#     plt.suptitle(title)
    
   
        
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 3],
                           height_ratios=[3, 1])
    axr = plt.subplot(gs[0])
    ax = plt.subplot(gs[1])
    axq = plt.subplot(gs[3])

    axq.plot(nn1, xts)  # query, horizontal, bottom
    axq.set_xlabel(xlab)

    axr.plot(yts, mm1)  # ref, vertical
    axr.invert_xaxis()
    axr.set_ylabel(ylab)

    ax.plot(d.index1, d.index2)

    if match_indices is None:
        idx = []
    elif not hasattr(match_indices, "__len__"):
        idx = np.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = np.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], 0),
                    (d.index1[i], d.index2[i])])
        col.append([(0, d.index2[i]),
                    (d.index1[i], d.index2[i])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)
    

    fig.canvas.draw()
    ponctuations
    #if path is not None :
        #plt.savefig(path, format='png')
        # To NOT SHOW
        #plt.close()

    #plt.show()
    #return ax


# %%
from dtw import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def get_dtw(hparams):
    with open(hparams.training_files, encoding="utf-8") as f:
        matrix = []
        #for line in f:
        line = 'FR-fr_Our/_wav_utt_22050/ES_LMP_NEB_01_0010_55|.Rodolphe, surpris, dit à la Goualeuse:|.Rodolphe, surpris, dit à la Goualeuse:'
        # create original matrix
        original = []
        parts = line.strip().split('|')
        #get the text part easy
        text_original = parts[1]
        #replace the first character 'ponctuation' by the one the list
        #for ponctuation in ponctuations:
            # create punctuation matrix
        ponctuation = '.'
        stext = list(text_original)
        # take the first ponctuation
        stext[0] = ponctuation
        #punctuationM.append(ponctuation)
        text = "".join(stext)
        #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
        text_Npath =  parts[0].strip().split('/')
        #take the last part S_LMP_NEB_02_0011_19
        basename = text_Npath[-1]
        mel_original = np.load('./mels/mel-{}.npy'.format(basename))
        sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence,hparams)
        _dir=ponctuation_dir
        # plot spectrogram
#         plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}-{}.png'.format(basename,"".join(text[0:100]))),
#             title='mel postnet \r \n {}'.format(text), split_title=True)  
#         plot_spectrogram(mel_original, os.path.join(_dir, 'plots/mel-post-{}-{}.png'.format(basename,"".join(text[0:100]))),
#             title='mel postnet \r \n {}'.format(text), split_title=True)  
        #print(mel_outputs_postnet.detach().cpu().numpy().shape[2])

        #save alignments
#                 plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}-{}.png'.format(basename,"".join(text[0:100]))),
#                     title='{}'.format(text), split_title=True)
        #save gates 
#                 idx = random.randint(0, alignments.size(0) - 1)
#                 plot_gate_outputs_to_numpy(
#                     torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(basename)),
#                     title='{}'.format(text), split_title=True)
        #generate audio
#         with torch.no_grad():
#             audio = waveglow.infer(mel_original.T, sigma=0.6)
#             audio_denoised = denoiser(audio, strength=0.05)[:, 0]
#             ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
#             audio_waveglow= audio_denoised.cpu().numpy()
        #print(len(mel_outputs_postnet.float().data.cpu().numpy()[0][1]))
        #generate plot audio
        #plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
        #    title='audio_denoised \r \n {}'.format(text), split_title=True)
        #save audio
        #save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-wavegElow-ljs-{}-.wav'.format(split_title_line(text[-50:-1], max_words=1))), sr=22050)
        #generate plot audio
#                 plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}-{}.png'.format(basename,"".join(text[0:100]))),
#                     title='audio_denoised \r \n {}'.format(text), split_title=True)
#                 #save audio
#                 save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-waveglow-ljs-{}-.wav'.format(basename,"".join(text[0:100]))), sr=hparams.sampling_rate)     
#                 inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
#                 audio_grif =griffin_lim(inverse_transform,stft_fn)
#                 save_wav(audio_grif[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-griffin-{}-.wav'.format(basename,"".join(text[0:100]))), sr=hparams.sampling_rate)

        #distance, path = fastdtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_original.T, dist=euclidean)
        original_mel = mel_original.T
        query_mel = mel_outputs_postnet.float().data.cpu().numpy()[0].T
        #print("original shape {}".format(original_mel[0].shape))
        #print("query shape {}".format(query_mel[0].shape))
        if ponctuation == '.':
            fig = plt.figure()
            fig.canvas.draw()
            plt.plot(original_mel[167])
            plt.plot(query_mel[154])
            distance, path = fastdtw(original_mel,query_mel, dist=euclidean)
            norm_distance = distance / len(path)
            print(distance)
            print(norm_distance)
            print(path)

            mel_dir  = os.path.join(os.getcwd(), 'mels-test')
            os.makedirs(mel_dir, exist_ok=True)
            #plt.plot(path[0](0,0))
    #         print("the path is : {}".format(path))
    #         print('The fast DTW distance is {} between original {} and {} : '.format(norm_distance, text_original,text))keep_internals
            original.append((text_original,ponctuation, norm_distance))
            alignment = dtw(original_mel, query_mel, keep_internals=True)
            #alignment = dtw(mel_original.T, mel_outputs_postnet.float().data.cpu().numpy()[0].T, keep_internals=True)

            #plt.suptitle('Categorical Plotting')
            dtwPlotThreeWay_BL(alignment,path = os.path.join(_dir, 'plots/DTW-{}-{}.png'.format(basename,"".join(text[0:100]))),
                xlab='Query: {}'.format(text), ylab=':Reference {}'.format(text_original), split_title=True)
    #             original.append(punctuationM)
    #             matrix.append(original)  
                #plt.suptitle('Categorical Plotting')
                ## Display the warping curve, i.e. the alignment curve
    #                 alignment.plot(type="threeway",title='The fast DTW distance is {} between original {} and {} : '.format(distance, text_original,text))
    #  

    #                 ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    #                 dtw(alignment, keep_internals=True, 
    #                     step_pattern=rabinerJuangStepPattern(6, "c"))\
    #                     .plot(type="twoway",offset=-2)

    #                 ## See the recursion relation, as formula and diagram
    #                 print(rabinerJuangStepPattern(6,"c"))
    #                 rabinerJuangStepPattern(6,"c").plot()
            #file.close()
            plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], None, text, False, original_mel, None, True ) 
            plot_data((original_mel.T,
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T))
            
            mel_dir  = os.path.join(os.getcwd(), 'mels-test')
            os.makedirs(mel_dir, exist_ok=True)


            np.save(os.path.join(mel_dir, 'mel-original_{}.npy'.format(basename)), original_mel.T, allow_pickle=False)
            np.save(os.path.join(mel_dir, 'mel-predicted_{}.npy'.format(basename)), mel_outputs_postnet.float().data.cpu().numpy()[0], allow_pickle=False)


            #idx = random.randint(0, alignments.size(0) - 1)
            #plot_gate_outputs_to_numpy(
                #torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
    #return matrix
matrix = get_dtw(hparams)

# %% [markdown]


# %%
from dtw import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def get_dtw(hparams):
    with open(hparams.training_files, encoding="utf-8") as f:
        matrix = []
        for line in f:
            # create original matrix
            original = []
            parts = line.strip().split('|')
            #get the text part easy
            text_original = parts[1]
            #replace the first character 'ponctuation' by the one the list
            for ponctuation in ponctuations:
                # create punctuation matrix
                stext = list(text_original)
                # take the first ponctuation
                stext[0] = ponctuation
                #punctuationM.append(ponctuation)
                text = "".join(stext)
                #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                text_Npath =  parts[0].strip().split('/')
                #take the last part S_LMP_NEB_02_0011_19
                basename = text_Npath[-1]
                mel_original = np.load('./mels/mel-{}.npy'.format(basename))
                sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
                sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence,hparams)
                _dir=ponctuation_dir
                # plot spectrogram
#                 plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}-{}.png'.format(basename,"".join(text[0:100]))),
#                     title='mel postnet \r \n {}'.format(text), split_title=True)  
                #print(mel_outputs_postnet.detach().cpu().numpy().shape[2])
                    
                #save alignments
#                 plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}-{}.png'.format(basename,"".join(text[0:100]))),
#                     title='{}'.format(text), split_title=True)
                #save gates 
#                 idx = random.randint(0, alignments.size(0) - 1)
#                 plot_gate_outputs_to_numpy(
#                     torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(basename)),
#                     title='{}'.format(text), split_title=True)
                #generate audio
#                 with torch.no_grad():
#                     audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
#                 ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
#                 #save audio
#                 audio_denoised = denoiser(audio, strength=0.05)[:, 0]
#                 ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
                #generate plot audio
#                 plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}-{}.png'.format(basename,"".join(text[0:100]))),
#                     title='audio_denoised \r \n {}'.format(text), split_title=True)
#                 #save audio
#                 save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-waveglow-ljs-{}-.wav'.format(basename,"".join(text[0:100]))), sr=hparams.sampling_rate)     
#                 inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
#                 audio_grif =griffin_lim(inverse_transform,stft_fn)
#                 save_wav(audio_grif[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-griffin-{}-.wav'.format(basename,"".join(text[0:100]))), sr=hparams.sampling_rate)
                
                #distance, path = fastdtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_original.T, dist=euclidean)
                original_mel = mel_original.T
                query_mel = mel_outputs_postnet.float().data.cpu().numpy()[0].T
                #print("original shape {}".format(original_mel[0].shape))
                #print("query shape {}".format(query_mel[0].shape))
                if ponctuation == '.':
                    plt.plot(original_mel[100])
                    plt.plot(query_mel[78])
                distance, path = fastdtw(original_mel,query_mel, dist=euclidean)
                norm_distance = distance / len(path)
                #print(path[:,:][0])
                #plt.plot(path[0](0,0))
                #print("the path is : {}".format(path))
                print('The fast DTW distance is {} between original {} and {} : '.format(norm_distance, text_original,text))
                original.append((text_original,ponctuation, norm_distance))
                alignment = dtw(original_mel,query_mel, keep_internals=True)
                #plt.suptitle('Categorical Plotting')
                dtwPlotThreeWay_BL(alignment,path = os.path.join(_dir, 'plots/DTW-{}-{}.png'.format(basename,"".join(text[0:100]))),
                    xlab='Query: {}'.format(text), ylab=':Reference {}'.format(text_original), split_title=True)
#             original.append(punctuationM)
            matrix.append(original)  
                #plt.suptitle('Categorical Plotting')
                ## Display the warping curve, i.e. the alignment curve
#                 alignment.plot(type="threeway",title='The fast DTW distance is {} between original {} and {} : '.format(distance, text_original,text))
#  

#                 ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
#                 dtw(alignment, keep_internals=True, 
#                     step_pattern=rabinerJuangStepPattern(6, "c"))\
#                     .plot(type="twoway",offset=-2)

#         e tu as vraiment eu ton doctorat! Mais 3laach! ! Ma dihach fihoum tt simplement! Nta insan najeh w ila jiti à chaque fois tred 3la lm7         ## See the recursion relation, as formula and diagram
#                 print(rabinerJuangStepPattern(6,"c"))
#                 rabinerJuangStepPattern(6,"c").plot()
            #file.close()
                #plot_data((mel_outputs.float().data.cpu().numpy()[0],
                    #mel_outputs_postnet.float().data.cpu().numpy()[0],
                    #alignments.float().data.cpu().numpy()[0].T))

                #idx = random.randint(0, alignments.size(0) - 1)
                #plot_gate_outputs_to_numpy(
                    #torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
    return matrix
matrix = get_dtw(hparams)

# %% [markdown]


# %%
import pandas as pd
#np.array(matrix).dump(open('Matrix_'+data_name+'.npy', 'wb'))
print(matrix[1][15])
#print(len(matrix[1]))

mean=[]
for ponctuation in ponctuations:
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j][0][0] == ponctuation:
                #p_mean=[]
                for ponctuation_predict in ponctuations:
                    if matrix[i][j][1] == ponctuation_predict:
                        #print("original: {}, predicted: {}, dist: {} ".format(ponctuation, matrix[i][j][1],matrix[i][j][2]))
                        mean.append((ponctuation, matrix[i][j][1],matrix[i][j][2]))
                #mean.append(p_mean)
#mean
df = pd.DataFrame(mean,columns =['Original', 'Predict', 'Score'])
#print(df)
mean=df.groupby(['Original', 'Predict'], as_index=False).mean()

df.to_csv('export_dataframe.csv', index = False, header=True)

mean.to_csv('export_mean_punctuation.csv', index = False, header=True)




mtr = mean.pivot(index='Predict',columns='Original',values='Score')
mtr.to_csv('export_Matrix_DTW.csv', index = False, header=True)
mtr
#mtr = pd.DataFrame(mtr,columns=[])
#g = mean.groupby("Original")
#mean.iloc[:,[2]]
#pd.concat((mean['Original'], pd.DataFrame(arr, columns=mean['Predict'])), axis=1)

#.groupby('Original')['Score'].mean()



            



#import csv

# with open("Matrix_"+data_name+".csv", 'w+', newline='', encoding="utf-8") as myfile:
#     wr = csv.writer(myfile,delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     wr.writerow(ponctuations)
#     wr.writerow(matrix)

# %% [markdown]
# # #### Prepare TEXT split ###***CHANGE IN MODEL***###
# # #### Prepare TEXT split ###***CHANGE IN MODEL***###
# # #### Prepare TEXT split ###***CHANGE IN MODEL***###

# %%
import pandas as pd
import html2text
import nltk

from bs4 import BeautifulSoup

#nltk.download('punkt')
df = pd.read_excel (hparams.training_files, sheet_name='Commentaires HN v2')
df = df.fillna('')
for i, line in enumerate(df.commentaire):
    if line != 'nan':
        #print (df.commentaire[232])
        #parag=line
        print(i)
        parag = BeautifulSoup(line, 'html.parser').get_text()
        #parag = html2text.html2text(line)
        parag=parag.replace("<p>", "")
        parag=parag.replace("</p>", "")

        parag=parag.replace("\- ", "")
        parag=parag.replace(" !","!")
        #print(parag)
        nltkparag = nltk.sent_tokenize(parag)
        sent_detector = nltk.data.load('tokenizers/punkt/french.pickle')

        for subline in sent_detector.tokenize(parag.strip()):
            subline=subline.strip()
            #print('\n-----\n'.join(sent_detector.tokenize(parag.strip())))
            subline=subline.replace(" !","!")    
            #print(parag)
            #print(parag.splitlines())
            #print( nltkparag)
            #print (df.commentaire[115])
            # create punctuation matrix
            stext = list('.'+subline)
            text = "".join(stext)
            print(text)
            #take the last part S_LMP_NEB_02_0011_19
            basename = str(i)+'-'
            sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
            #print(text) 
            sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence,hparams)
            
            _dir=os.path.join(os.getcwd(), 'ponctuation-eval_'+ checkpoint_name + data_name[0:20])
            plot_dir = os.path.join(_dir, 'plots')
            wav_dir = os.path.join(_dir, 'wavs')
            
            os.makedirs(_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)
            
            #save mel spectrogram plot
#             plot_spectrogram(mel_outputs.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-{}.png'.format(basename)),
#                 title='{}'.format(text), split_title=True)  
            #save mel_outputs_postnet plot
            plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}.png'.format(basename)),
                title='mel postnet \r \n {}'.format(text), split_title=True)  


            #save alignments
            plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}.png'.format(basename)),
                title='{}'.format(text), split_title=True)
            #save gates 
            idx = random.randint(0, alignments.size(0) - 1)
            plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(basename)),
                title='{}'.format(text), split_title=True)
            #generate audio
            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
            ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
            #save audio
            audio_denoised = denoiser(audio, strength=0.05)[:, 0]
            ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
            #generate plot audio
#             plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}.png'.format(basename)),
#                 title='audio_denoised \r \n {}'.format(text), split_title=True)
            #save audio
            save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-waveglow-ljs-{}-.wav'.format(basename,split_title_line("".join(text[0:100]), max_words=1))), sr=22050)     
#             inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
#             audio =griffin_lim(inverse_transform,stft_fn)
#             save_wav(audio[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-griffin-{}-.wav'.format(basename,split_title_line(text[-50:-1], max_words=1))), sr=hparams.sampling_rate)
        



# %%
# #### Prepare TEXT split ###***CHANGE IN MODEL***###
def get_sentence(hparams):
    with open(hparams.training_files, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('|')
            #get the text part easy
            text = parts[1]
            
            #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
            text_Npath =  parts[0].strip().split('/')
            #take the last part S_LMP_NEB_02_0011_19
            basename = text_Npath[-1]
            sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
            #print(text) 
            sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence,hparams)
            #CB-BL: by me save the original duration
            ###***CHANGE Name***###
            ###***CHANGE Name***###
            ###***CHANGE Name***###
            file = open('predicted_duration_difficile.txt', "a+", encoding="utf-8")
            ###***CHANGE Name***###
            ###***CHANGE Name***###
            ###***CHANGE Name***###
            _dir=difficile_dir
            ###***CHANGE Name***###
            mel_outputs_postnet_np = mel_outputs_postnet.detach().cpu().numpy()
            file.write(str(mel_outputs_postnet_np.shape[2]) + "\n")
            #save mel spectrogram plot
            plot_spectrogram(mel_outputs.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-{}.png'.format(basename)),
                title='{}'.format(text), split_title=True)  
            #save mel_outputs_postnet plot
            plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}.png'.format(basename)),
                title='mel postnet \r \n {}'.format(text), split_title=True)  
            print(mel_outputs_postnet.detach().cpu().numpy().shape[2])

            #save alignments
            plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}.png'.format(basename)),
                title='{}'.format(text), split_title=True)
            #save gates 
            idx = random.randint(0, alignments.size(0) - 1)
            plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(basename)),
                title='{}'.format(text), split_title=True)
            #generate audio
            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
            ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
            #save audio
            audio_denoised = denoiser(audio, strength=0.05)[:, 0]
            ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
            #generate plot audio
            plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}.png'.format(basename)),
                title='audio_denoised \r \n {}'.format(text), split_title=True)
            #save audio
            save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-waveglow-ljs-{}-.wav'.format(basename,split_title_line(text[-50:-1], max_words=1))), sr=22050)     
            inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
            audio =griffin_lim(inverse_transform,stft_fn)
            save_wav(audio[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-{}-griffin-{}-.wav'.format(basename,split_title_line(text[-50:-1], max_words=1))), sr=hparams.sampling_rate)
        file.close()
            #plot_data((mel_outputs.float().data.cpu().numpy()[0],
                #mel_outputs_postnet.float().data.cpu().numpy()[0],
                #alignments.float().data.cpu().numpy()[0].T))
                
            #idx = random.randint(0, alignments.size(0) - 1)
            #plot_gate_outputs_to_numpy(
                #torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
    return mel_outputs_postnet
get_sentence(hparams)
# %% [markdown]


# %%
def plot_bis(state):
    X, Y = [], []
    #for line in open('original_duration_{}_{}_{}.txt'.format(state,data,checkpoint_name)):
    for line in open('original_duration.txt'):
      values = [float(s) for s in line.split()]
      X.append(values[0])
    #for line in open('predicted_duration_{}_{}_{}.txt'.format(state,'Y_train_Shape_11121_2020-02-03 17.25.48.002',checkpoint_name)):
    for line in open('predicted_duration_{}.txt'.format('Y_valid_Shape_586_2020-02-03 17.25.48.074_16000')):
      values = [float(s) for s in line.split()]
      Y.append(values[0])
    fig = plt.figure(figsize=(16, 10))
    plt.xlabel("Original number of mel-frames for "+ state + "data"+" iteration "+ checkpoint_name)
    plt.ylabel("predicted number of mel-frames for "+ state + " data"+" iteration "+ checkpoint_name)
    plt.tight_layout()
    ###***CHANGE Name***###
    basename = 'mel-frame_{}_{}'.format(state,checkpoint_name)
    fig.canvas.draw()
    ###***CHANGE Color***###

    if state == "train": 
        plt.plot(X, Y,'ro')
        plt.savefig(os.path.join(plot_test_dir,'comparaison-{}.png'.format(basename)), format='png')
        plt.close()
    elif state == "valid":
        plt.plot(X, Y,'bo')
        plt.savefig(os.path.join(plot_test_dir,'comparaison-{}.png'.format(basename)), format='png')
        plt.close()
    elif state == "test": 
        plt.plot(X, Y,'go')
        plt.savefig(os.path.join(plot_test_dir,'comparaison-{}.png'.format(basename)), format='png')
        plt.close()


##***CHANGE Name***###
plot_bis("valid")



# %%

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))


# %%

def inv_mel_spectrogram_tensorflow(mel_spectrogram, hparams):
	'''Builds computational graph to convert mel spectrogram to waveform using TensorFlow.
	Unlike inv_mel_spectrogram, this does NOT invert the preemphasis. The caller should call
	inv_preemphasis on the output after running the graph.
	'''
	if hparams.signal_normalization:
		D = _denormalize_tensorflow(mel_spectrogram, hparams)
		
	else:
		D = mel_spectrogram

	S = tf.pow(_db_to_amp_tensorflow(D + hparams.ref_level_db), (1/hparams.magnitude_power))
	#print(S.shape)
	S = _mel_to_linear_tensorflow(S, hparams)  # Convert back to linear
	print(S.shape)
	return _griffin_lim_tensorflow(tf.pow(S, hparams.power), hparams)


# %%
text = "sir tkawad ya wald tit malk dayr fiha rajel sir thawa"
#text = ".Bonjour, je m'appelle badreddine."
sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()


# %%
text2 = ".!je m'appelle ."
sequence2 = np.array(text_to_sequence(text2, ['transliteration_cleaners']))[None, :]
sequence2 = torch.autograd.Variable(
    torch.from_numpy(sequence2)).cuda().long()


# %%
text3 = ";je m'appelle ;"
sequence3 = np.array(text_to_sequence(text3, ['transliteration_cleaners']))[None, :]
sequence3 = torch.autograd.Variable(
    torch.from_numpy(sequence3)).cuda().long()


# %%
text4 = "-Bonjour Tacotron ..."
sequence4 = np.array(text_to_sequence(text4, ['transliteration_cleaners']))[None, :]
sequence4 = torch.autograd.Variable(
    torch.from_numpy(sequence4)).cuda().long()

# %% [markdown]
# #### Decode text input and plot results


# %%
mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence,hparams)

plot_data((mel_outputs.float().data.cpu().numpy()[0],
        mel_outputs_postnet.float().data.cpu().numpy()[0],
        alignments.float().data.cpu().numpy()[0].T))

plot_spectrogram(mel_outputs.float().data.cpu().numpy()[0],title='{}'.format(text), split_title=True) 

print(mel_outputs_postnet.detach().cpu().numpy().shape[2])

    
idx = random.randint(0, alignments.size(0) - 1)

plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()) # ,dataformats='HWC'

mel_outputs_postnet.detach().cpu().numpy()[0,79]


# %%
#save mel spectrogram plot
_dir='/home/lebbatb-admin/Images'
plot_spectrogram(mel_outputs.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
    title='{}'.format(text), split_title=True)  
#save mel_outputs_postnet plot
plot_spectrogram(mel_outputs_postnet.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/mel-post-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
    title='mel postnet \r \n {}'.format(text), split_title=True)  
#print(mel_outputs_postnet.detach().cpu().numpy().shape[2])

#save alignments
plot_alignment(alignments.float().data.cpu().numpy()[0].T, os.path.join(_dir, 'plots/alignment-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
    title='{}'.format(text), split_title=True)
#save gates 
idx = random.randint(0, alignments.size(0) - 1)
plot_gate_outputs_to_numpy(
    torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(), os.path.join(_dir, 'plots/gate-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
    title='{}'.format(text), split_title=True)
# generate audio
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
    audio_denoised = denoiser(audio, strength=0.05)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
audio_waveglow= audio_denoised.cpu().numpy()
#print(len(mel_outputs_postnet.float().data.cpu().numpy()[0][1]))
#generate plot audio
#plot_wave(audio_denoised.float().data.cpu().numpy()[0], os.path.join(_dir, 'plots/audio-{}.png'.format(split_title_line(text[-50:-1], max_words=1))),
#    title='audio_denoised \r \n {}'.format(text), split_title=True)
#save audio
save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-waveglow-ljs-{}-.wav'.format(split_title_line(text[-50:-1], max_words=1))), sr=22050)


# %%
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)


# %%
#audio = inv(mel_outputs_postnet.float().T.data.cpu().numpy()[:,:,0])
inverse_transform = stft.mel_inv_spectrogram(mel_outputs_postnet.float().data.cpu())
audio =griffin_lim(inverse_transform,stft_fn)

ipd.Audio(audio.data.cpu().numpy(), rate=22050)
#audio_grif = audio.data.cpu().numpy()


#print(len(audio_grif[0]))
#save_wav(audio[0].data.cpu().numpy(), os.path.join(_dir, 'wavs/wav-griffin-{}-.wav'.format(split_title_line(text[-50:-1], max_words=1))), sr=hparams.sampling_rate)


# %%
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

distance, path = fastdtw(mel_outputs_postnet.float().data.cpu().numpy()[0], mel_outputs.float().data.cpu().numpy()[0], dist=euclidean)
print(distance)


# %%
from dtw import *
## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formulaaudio_grif[0].ravel()
from dtw import *
alignment = dtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_outputs.float().data.cpu().numpy()[0].T, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_outputs.float().data.cpu().numpy()[0].T, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()


# %%
mel_outputs2, mel_outputs_postnet2, _, alignments2 = model.inference(sequence2,hparams)
plot_data((mel_outputs2.float().data.cpu().numpy()[0],
        mel_outputs_postnet2.float().data.cpu().numpy()[0],
        alignments2.float().data.cpu().numpy()[0].T))
idx = random.randint(0, alignments.size(0) - 1)
plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()) # ,dataformats='HWC'


# %%
mel_outputs3, mel_outputs_postnet3, _, alignments3 = model.inference(sequence3,hparams)
plot_data((mel_outputs3.float().data.cpu().numpy()[0],
        mel_outputs_postnet3.float().data.cpu().numpy()[0],
        alignments3.float().data.cpu().numpy()[0].T))

idx = random.randint(0, alignments.size(0) - 1)
plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()) # ,dataformats='HWC'


# %%
mel_outputs4, mel_outputs_postnet4, _, alignments4 = model.inference(sequence4,hparams)
plot_data((mel_outputs4.float().data.cpu().numpy()[0],
        mel_outputs_postnet4.float().data.cpu().numpy()[0],
        alignments4.float().data.cpu().numpy()[0].T))
idx = random.randint(0, alignments.size(0) - 1)
plot_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()) # ,dataformats='HWC'

# %% [markdown]
# #### Synthesize audio from spectrogram using WaveGlow


# %%
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    #print(audio.shape)
ipd.Audio(audio[0].data.cpu().numpy(), rate=22050)

# plt.figure()
# plt.plot(audio[0,:].cpu().numpy())


# %%
with torch.no_grad():
    audio2 = waveglow.infer(mel_outputs_postnet2, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)


# %%
with torch.no_grad():
    audio3 = waveglow.infer(mel_outputs_postnet3, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)


# %%
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet4, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

# %% [markdown]
# #### (Optional) Remove WaveGlow bias


# %%
audio_denoised = denoiser(audio, strength=0.05)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
#save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join('/home/lebbatb-admin/', '+{}jouer.wav'.format('.')), sr=hparams.sampling_rate)


# %%
audio_denoised = denoiser(audio2, strength=0.02)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 


# %%
audio_denoised = denoiser(audio3, strength=0.02)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 


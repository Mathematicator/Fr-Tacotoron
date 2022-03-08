#!/usr/bin/python3
# -*- coding: utf8 -*
from dtw import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import matplotlib
import torch 

#matplotlib inline
# import matplotlib.pylab as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
import math
import os 
# os.chdir('/media/lebbat/lebbat/NVIDIA-official/tacotron2_gricad')

#import plot_DTW_BL
# ENDIMPORT
# import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc

import sys
import argparse


from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser
import tensorflow as tf
import pandas as pd

import layers
from utils import load_wav_to_torch, load_filepaths_and_text

output_dir_dtw=os.path.join(os.getcwd(), 'DTW_output')
os.makedirs(output_dir_dtw, exist_ok=True)

def matrix_to_csv(ponctuations, matrix, filename,  checkpoint_name, data_name):
    mean_dtw = []
    for ponctuation in ponctuations:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                ponctuation_m = matrix[i][j][0][0]
                if matrix[i][j][0][0] == ponctuation:
                    #p_mean_dtw=[]
                    for ponctuation_predict in ponctuations:
                        ponctuation_predict_m = matrix[i][j][0][1] 
                        if matrix[i][j][1] == ponctuation_predict:
                            print("original: {}, predicted: {}, dist: {} ".format(ponctuation, matrix[i][j][1],matrix[i][j][2]))
                            mean_dtw.append((ponctuation, matrix[i][j][1],matrix[i][j][2]))
                    #mean_dtw.append(p_mean_dtw)
    #mean_dtw
    df = pd.DataFrame(mean_dtw,columns =['Original', 'Predict', 'Score_dtw'])
    # convert column "a" of a DataFrame
    df['Score_dtw'] = pd.to_numeric(df['Score_dtw'])
    
    print(df)
    result = df.dtypes
    print(result)
    mean_dtw=df.groupby(['Original', 'Predict'], as_index=False).mean()

    df.to_csv(os.path.join(output_dir_dtw,'export_dataframe_predict_predict_dtw_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mean_dtw.to_csv(os.path.join(output_dir_dtw,'export_mean_predict_predict_dtw_punctuation_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mtr_dtw = mean_dtw.pivot(index='Predict',columns='Original',values='Score_dtw')
    mtr_dtw.to_csv(os.path.join(output_dir_dtw,'export_Matrix_predict_predict_DTW_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mean_long = []
    for ponctuation in ponctuations:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j][0][0] == ponctuation:
                    for ponctuation_predict in ponctuations:
                        if matrix[i][j][1] == ponctuation_predict:
                            mean_long.append((ponctuation, matrix[i][j][1],matrix[i][j][3]))
    df = pd.DataFrame(mean_long,columns =['Original', 'Predict', 'Score_long'])
    
    # convert column "a" of a DataFrame
    df['Score_long'] = pd.to_numeric(df['Score_long'])
    mean_long=df.groupby(['Original', 'Predict'], as_index=False).mean()

    df.to_csv(os.path.join(output_dir_dtw,'export_dataframe_predict_predict_long_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mean_long.to_csv(os.path.join(output_dir_dtw,'export_mean_predict_predict_long_punctuation_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mtr_long = mean_long.pivot(index='Predict',columns='Original',values='Score_long')
    mtr_long.to_csv(os.path.join(output_dir_dtw,'export_Matrix_predict_predict_Long_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)
    
    return mtr_dtw, mtr_long


def get_mel(filename, hparams):
        # return the mels by using stft.mel_spectrogram and squezz after 
        load_mel_from_disk = hparams.load_mel_from_disk
        filter_mel =  hparams.filter_mel
        mel_files = hparams.mel_files
        text_cleaners = hparams.text_cleaners
        max_wav_value = hparams.max_wav_value
        sampling_rate = hparams.sampling_rate
        #BL:added by me 
        max_output_length = hparams.max_output_length
        wav_dir = hparams.wav_dir
        bl_counter = 0
        # BL: end added
        sampling_rate = hparams.sampling_rate
        stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        if not load_mel_from_disk:
            #Bl :added to add wav prefix to the audio 
            audiopath =  filename
            audio, sampling_rate = load_wav_to_torch(audiopath+'.wav')
            if sampling_rate != stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, stft.sampling_rate))
            audio_norm = audio / max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            parts = filename.strip().split('/')
            basename = parts[-1]
            mel_filename = '{}mel-{}.npy'.format(mel_files, basename)
            melspec = torch.from_numpy(np.load(mel_filename))
            assert melspec.size(0) == stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), stft.n_mel_channels))

        return melspec


def get_dtw(hparams, ponctuations,model,filename, checkpoint_name, data_name):
    with open(filename, encoding="utf-8") as f:
        matrix = []
        for i_ligne , line in enumerate(f):
            #line = 'FR-fr_Our/_wav_utt_22050/ES_LMP_NEB_01_0010_55|.Rodolphe, surpris, dit à la Goualeuse:|.Rodolphe, surpris, dit à la Goualeuse:'
            # create original matrix
            original = []
            parts = line.strip().split('|')
            #get the text part easy
            text_original = parts[1]
            #replace the first character 'ponctuation' by the one the list
            for j_poncutuation, ponctuation in enumerate(ponctuations):
                if i_ligne == 0:
                    special_iterator =  j_poncutuation
                else:    
                    special_iterator = (i_ligne-1)*10 + j_poncutuation
                # create punctuation matrix
                stext = list(text_original)
                # take the first ponctuation
                if ponctuation == '¬':
                    print("stop")
                stext[0] = ponctuation
                #punctuationM.append(ponctuation)
                text = "".join(stext)
                #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                text_Npath =  parts[0].strip().split('/')
                #take the last part S_LMP_NEB_02_0011_19
                basename = text_Npath[-1]
                try:
                    #mel_filename = '{}mel-{}.npy'.format(hparams.mel_files, basename)
                    mel_filename = 'mel-{}-predicted-{}-{}.npy'.format(checkpoint_name, text_original[0], basename)
                    mel_dir = os.path.join(os.getcwd(), 'mels-DTW')
                    melspec_original = torch.from_numpy(np.load(os.path.join(mel_dir, mel_filename)))
                    original_frame_size = melspec_original.shape[1]
                    
                except:
                    print('#~ ERROR with opening Original predict/predicted_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name ,data_name, len(ponctuations)))
                    sequence = np.array(text_to_sequence(text_original, ['basic_cleaners']))[None, :]
                    sequence = torch.autograd.Variable(
                    torch.from_numpy(sequence)).cuda().long()
                    _, mel_outputs_postnet, _, _ = model.inference(sequence,hparams)
                    _dir=os.path.join(output_dir_dtw, 'DTW_plot_output')
                    os.makedirs(_dir, exist_ok=True)
                    # distance, path = fastdtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_original.T, dist=euclidean)
                    ###ADD mels-DTW output path ####
                    mel_filename = 'mel-{}-predicted-{}-{}.npy'.format(checkpoint_name, text_original[0], basename)
                    mel_dir = os.path.join(os.getcwd(), 'mels-DTW')
                    os.makedirs(mel_dir, exist_ok=True)
                    np.save(os.path.join(mel_dir, mel_filename), mel_outputs_postnet.float().data.cpu().numpy()[0], allow_pickle=False)
                    print("mel predicted file {} saved in {}".format(mel_filename, mel_dir))
                    melspec_original = mel_outputs_postnet.float().data.cpu().numpy()[0].T
                    original_frame_size = melspec_original.shape[1]
                    
                try:
                    print("special _iterator :{}".format(special_iterator))
                    try:
                        if hparams.flag_predicted_saved == True:
                            with open(os.path.join(output_dir_dtw,'predicted_predicted_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name , data_name, len(ponctuations))), "r", encoding="utf-8") as file_predicted:
                                content = file_predicted.readlines()
                                content = [x.strip().split('|') for x in content]
                                predicted_frame_size = content[special_iterator][3]
                                norm_distance = content[special_iterator][4]
                                norm_long = content[special_iterator][-1]
                                hparams.flag_predicted_saved = False
                        else :
                            predicted_frame_size = content[special_iterator][3]
                            norm_distance = content[special_iterator][4]
                            norm_long = content[special_iterator][-1]
                    except:

                        if hparams.flag_predicted == True and  hparams.flag_truncate :
                            with open(os.path.join(output_dir_dtw,'predicted_predicted_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name , data_name, len(ponctuations))), "a+", encoding="utf-8") as file_predicted:
                                file_predicted.truncate(0)
                            hparams.flag_original = False
                            hparams.flag_truncate = False
                        
                    try:
                        ###ADD mels-DTW output path ####
                        mel_filename = 'mel-{}-predicted-{}-{}.npy'.format(checkpoint_name, ponctuation, basename)
                        mel_dir = os.path.join(os.getcwd(), 'mels-DTW')
                        loaded_predict_mel = torch.from_numpy(np.load(os.path.join(mel_dir, mel_filename)))
                        query_mel = loaded_predict_mel.T
                    except:    
                        print('#~ ERROR with opening the predicted_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name ,data_name, len(ponctuations)))
                        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
                        sequence = torch.autograd.Variable(
                        torch.from_numpy(sequence)).cuda().long()
                        _, mel_outputs_postnet, _, _ = model.inference(sequence,hparams)
                        _dir=os.path.join(output_dir_dtw, 'DTW_plot_output')
                        os.makedirs(_dir, exist_ok=True)
                        # distance, path = fastdtw(mel_outputs_postnet.float().data.cpu().numpy()[0].T, mel_original.T, dist=euclidean)
                        ###ADD mels-DTW output path ####
                        mel_filename = 'mel-{}-predicted-{}-{}.npy'.format(checkpoint_name, ponctuation, basename)
                        mel_dir = os.path.join(os.getcwd(), 'mels-DTW')
                        os.makedirs(mel_dir, exist_ok=True)
                        np.save(os.path.join(mel_dir, mel_filename), mel_outputs_postnet.float().data.cpu().numpy()[0], allow_pickle=False)
                        print("mel predicted file {} saved in {}".format(mel_filename, mel_dir))
                        query_mel = mel_outputs_postnet.float().data.cpu().numpy()[0].T
                    original_mel = melspec_original.T
                    #print("original shape {}".format(original_mel[0].shape))
                    #print("query shape {}".format(query_mel[0].shape))
                    distance, path = fastdtw(original_mel,query_mel,  dist=dist_MSE_BL)
                    norm_distance = distance / len(path)
                    predicted_frame_size = query_mel.shape[0]
                    norm_long = (int(predicted_frame_size) - int(original_frame_size))/int(original_frame_size)
                    with open(os.path.join(output_dir_dtw,'predicted_predicted_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name , data_name, len(ponctuations))), "a+", encoding="utf-8") as file_predicted:
                        file_predicted.write("{}|{}|{}|{}|{}|{}\n".format(basename, text_original[0], ponctuation, predicted_frame_size, norm_distance, norm_long))
                        # alignment = dtw(original_mel,query_mel, keep_internals=True)
                        # dtwPlotThreeWay_BL(alignment,path = os.path.join(_dir, 'DTW-{}-{}.png'.format(basename,"".join(text[0:100]))),
                        #     xlab='Query: {}'.format(text), ylab=':Reference {}'.format(text_original), split_title=True)
                        #plt.plot(path[0](0,0))
                        # print("the path is : {}".format(path))
                except :
                    print("ERROR NO LOAD")
                print('The fast DTW: {} dist_long: {} between original {} and {} : '.format(norm_distance, norm_long, text_original,text))
                original.append((text_original,ponctuation, norm_distance, norm_long))
                #plt.suptitle('Categorical Plotting')
            matrix.append(original)  
    return matrix

def dist_MSE_BL(x,y):
    d =((x-y)**2).mean()
    return d


def dtwPlotThreeWay_BL(d=None, xts=None, yts=None,title=None,path=None, split_title=False,
                    match_indices=None,
                    match_col="gray",
                    xlab="Query index",
                    ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotThreeWay
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

    fig = matplotlib.pyplot.figure()
    
#         #BL:added 
#     if split_title is not None:
#         title = split_title_line(title)
#     plt.suptitle(title)
    
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 3],
                           height_ratios=[3, 1])
    axr = matplotlib.pyplot.subplot(gs[0])
    ax = matplotlib.pyplot.subplot(gs[1])
    axq = matplotlib.pyplot.subplot(gs[3])

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
    if path is not None :
        matplotlib.pyplot.savefig(path, format='png')
        # To NOT SHOW
        matplotlib.pyplot.close()

    #plt.show()
    #return ax

def prepare_params(checkpoint_path, training_files, mode):
    hparams = create_hparams()
    hparams.max_decoder_steps=2000
    hparams.sampling_rate = 22050
    
    if checkpoint_path is not None :
        checkpoint_path = checkpoint_path
    if training_files is not None :
        filename= training_files
    
    if mode == 'train':
        filename = hparams.training_files
    if mode == 'validation':
        filename = hparams.validation_files
    # hparams.n_symbols = 128
    
    data_name = filename.strip().split('/')
    data_name = data_name[-1]
    
    #load model
    checkpoint_name = checkpoint_path.strip().split('/')
    checkpoint_name = checkpoint_name[-1]
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    #¬ (10578)  § (6006) ~ (3672) . (2494) , (2668)  ; (1932) ! (1056)  ? (658) : (416)  « (395)
    ponctuations = ['¬', '§',  '~',  '.',  ',',  ';',  '!', '?', ':', '«']
    len(ponctuations)
    return hparams , filename, checkpoint_name, data_name, model, ponctuations

def split_title_line(title_text, max_words=10):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])
    # %% [markdown]
    
    

if __name__ == '__main__':
    accepted_modes = ['train', 'validation']
    parser = argparse.ArgumentParser(
        description='DTW_MATRIX_2D_SURF')
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-d', '--dataset', type=str,
                        help='full path to the dataset checkpoint file')
    parser.add_argument('-m', '--mode', type=str,
                        help='mode of run: can be one of {}'.format(accepted_modes))
    args, _ = parser.parse_known_args()
    if args.mode not in accepted_modes and args.dataset is None:
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

    hparams , filename, checkpoint_name, data_name, model, ponctuations = prepare_params(args.checkpoint_path, args.dataset, args.mode)
    hparams.flag_original = True
    hparams.flag_predicted = True
    hparams.flag_original_saved = True
    hparams.flag_predicted_saved = True
    hparams.flag_truncate = True
    matrix = get_dtw(hparams, ponctuations, model,filename,  checkpoint_name, data_name)
    mtr = matrix_to_csv(ponctuations, matrix, filename,  checkpoint_name, data_name)
    print(mtr)

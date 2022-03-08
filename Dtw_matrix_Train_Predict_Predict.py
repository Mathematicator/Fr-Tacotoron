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

output_dir_dtw=os.path.join(os.getcwd(), 'DTW_output_Train')
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

    df.to_csv(os.path.join(output_dir_dtw,'export_dataframe_Train_predict_predict_dtw_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mean_dtw.to_csv(os.path.join(output_dir_dtw,'export_mean_Train_predict_predict_dtw_punctuation_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mtr_dtw = mean_dtw.pivot(index='Predict',columns='Original',values='Score_dtw')
    mtr_dtw.to_csv(os.path.join(output_dir_dtw,'export_Matrix_Train_predict_predict_DTW_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

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

    df.to_csv(os.path.join(output_dir_dtw,'export_dataframe_Train_predict_predict_long_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mean_long.to_csv(os.path.join(output_dir_dtw,'export_mean_Train_predict_predict_long_punctuation_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)

    mtr_long = mean_long.pivot(index='Predict',columns='Original',values='Score_long')
    mtr_long.to_csv(os.path.join(output_dir_dtw,'export_Matrix_Train_predict_predict_Long_{}_{}.csv'.format(checkpoint_name, data_name)), index = False, header=True)
    
    return mtr_dtw, mtr_long



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
            if text_original[0] not in ponctuations:
                continue
            #replace the first character 'ponctuation' by the one the list
            for j_poncutuation, ponctuation in enumerate(ponctuations):
                # if i_ligne == 0:
                #     special_iterator =  j_poncutuation
                # else:    
                #     special_iterator = (i_ligne-1)*10 + j_poncutuation
                # # create punctuation matrix
                # stext = list(text_original)
                # # take the first ponctuation
                # stext[0] = ponctuation
                # #punctuationM.append(ponctuation)
                # text = "".join(stext)
                #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                text_Npath =  parts[0].strip().split('/')
                #take the last part S_LMP_NEB_02_0011_19
                basename = text_Npath[-1]
                #mel_filename = '{}mel-{}.npy'.format(hparams.mel_files, basename)
                mel_filename = 'mel-training-{}-predicted-{}-{}.npy'.format(checkpoint_name, text_original[0], basename)
                mel_dir = os.path.join(os.getcwd(), 'mels-DTW-Training')
                melspec_original = torch.from_numpy(np.load(os.path.join(mel_dir, mel_filename)))
                original_frame_size = melspec_original.shape[0]
            

                ###ADD mels-DTW output path ####
                mel_filename = 'mel-training-{}-predicted-{}-{}.npy'.format(checkpoint_name, ponctuation, basename)
                mel_dir = os.path.join(os.getcwd(), 'mels-DTW-Training')
                loaded_predict_mel = torch.from_numpy(np.load(os.path.join(mel_dir, mel_filename)))
                query_mel = loaded_predict_mel.T
                original_mel = melspec_original.T
                #print("original shape {}".format(original_mel[0].shape))
                #print("query shape {}".format(query_mel[0].shape))
                distance, path = fastdtw(original_mel,query_mel,  dist=dist_MSE_BL)
                norm_distance = distance / len(path)
                # norm_distance_MSE = dist_MSE_BL(original_mel,query_mel)
                predicted_frame_size = query_mel.shape[0]
                norm_long = (int(predicted_frame_size) - int(original_frame_size))/int(original_frame_size)
                with open(os.path.join(output_dir_dtw,'predicted_predicted_training_mel_frames_length_{}_{}_punc_{}.txt'.format(checkpoint_name , data_name, len(ponctuations))), "a+", encoding="utf-8") as file_predicted:
                    file_predicted.write("{}|{}|{}|{}|{}|{}\n".format(basename, text_original[0], ponctuation, predicted_frame_size, norm_distance, norm_long))
                    # alignment = dtw(original_mel,query_mel, keep_internals=True)
                    # dtwPlotThreeWay_BL(alignment,path = os.path.join(_dir, 'DTW-{}-{}.png'.format(basename,"".join(text[0:100]))),
                    #     xlab='Query: {}'.format(text), ylab=':Reference {}'.format(text_original), split_title=True)
                    # print("the path is : {}".format(path))
                # print('The fast DTW: {} dist_long: {} between original {} and {} : '.format(norm_distance, norm_long, text_original,text))
                original.append((text_original,ponctuation, norm_distance, norm_long))
                #plt.suptitle('Categorical Plotting')
            matrix.append(original)  
    return matrix

def dist_MSE_BL(x,y):
    d =((x-y)**2).mean()
    return d


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

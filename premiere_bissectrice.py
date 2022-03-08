from data_utils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader

import multiprocessing
multiprocessing.set_start_method('spawn', True)
import os 
print(os.getcwd())
import sys
import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

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

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
import argparse

# Bokeh Libraries
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, HoverTool

#for stats
import pandas as pd
from math import pi
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from collections import Counter
from bokeh.layouts import column


def config(checkpoint_path, training_files, mode):
    hparams = create_hparams()
    
    if checkpoint_path is not None :
        checkpoint_path = checkpoint_path
    if training_files is not None :
        filename= training_files
    
    if mode == 'train':
        filename = hparams.training_files
    if mode == 'validation':
        filename = hparams.validation_files
        
    
        
    hparams.max_decoder_steps=1000
    hparams.sampling_rate = 22050
    hparams.batch_size = 2
    data_name = filename.strip().split('/')
    data_name = data_name[-1]


    #load model
    checkpoint_name = checkpoint_path.strip().split('/')
    checkpoint_name = checkpoint_name[-1]
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    return hparams , filename, checkpoint_name, data_name, model


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


# #### Prepare TEXT split ###***CHANGE IN MODEL***###
def get_mel_frame(hparams, dataset, checkpoint_name, data_name, model):
    output_save = os.path.join(os.getcwd(), 'premiere_bissectrice_output')
    os.makedirs(output_save, exist_ok=True)
    with open(os.path.join(output_save,'original_mel_frames_length_{}_{}.txt'.format(checkpoint_name ,data_name)), "a+", encoding="utf-8") as file_original:
        with open(os.path.join(output_save,'predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "a+", encoding="utf-8") as file_predicted:
            with open(dataset, encoding="utf-8") as f:
                # file_original.truncate(0)
                # file_predicted.seek(0)
                # file_original.seek(0)
                for i, line in enumerate(f):
                    parts = line.strip().split('|')
                    text = parts[1]
                    #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                    text_Npath =  parts[0].strip().split('/')
                    #take the last part S_LMP_NEB_02_0011_19
                    basename = text_Npath[-1]
                    try:
                        mel_filename = '{}mel-{}.npy'.format(hparams.mel_files, basename)
                        melspec_original = torch.from_numpy(np.load(mel_filename))
                    except:
                            wav_dir = hparams.wav_dir
                            audiopath = wav_dir + basename
                            mel_filename = 'mel-{}.npy'.format(basename)
                            melsize = get_mel(audiopath, hparams).size(1)
                            mel_dir = os.path.join(os.getcwd(), 'mels')
                            os.makedirs(mel_dir, exist_ok=True)
                            np.save(os.path.join(mel_dir, mel_filename), get_mel(audiopath, hparams), allow_pickle=False)
                            print("mel file {} saved in {}".format(mel_filename, mel_dir))
                            melspec_original = torch.from_numpy(np.load(os.path.join(mel_dir, mel_filename)))
                    original_frame_size = melspec_original.shape[1]
                    try : 
                        if hparams.flag_original_saved == True:
                            with open(os.path.join(output_save,'original_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "r", encoding="utf-8") as file_original:
                                content_original = file_original.readlines()
                                content_original = [x.strip().split('|')[-1] for x in content_original]
                                original_frame_size = content_original[i]
                                hparams.flag_original_saved = False
                        else :
                            original_frame_size = content_original[i]
                    
                    except: 
                        # file_original.seek(0)
                        if hparams.flag_original == True:
                            with open(os.path.join(output_save,'original_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "a+", encoding="utf-8") as file_original:
                                file_original.truncate(0)
                            hparams.flag_original = False
                        with open(os.path.join(output_save,'original_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "a+", encoding="utf-8") as file_original:
                            file_original.write(basename + '|' + str(original_frame_size) + "\n")
                        print('Saving original {} {} in original file '.format(basename ,original_frame_size))
                    # except :
                    try : 
                        if hparams.flag_predicted_saved == True:
                            with open(os.path.join(output_save,'predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "r", encoding="utf-8") as file_predicted:
                                content = file_predicted.readlines()
                                content = [x.strip().split('|')[-1] for x in content]
                                predicted_frame_size = content[i]
                                hparams.flag_predicted_saved = False
                        else :
                            predicted_frame_size = content[i]
                            
                    except:
                        if hparams.flag_predicted == True:
                            with open(os.path.join(output_save,'predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "a+", encoding="utf-8") as file_predicted:
                                file_predicted.truncate(0)
                            hparams.flag_original = False
                        print('#~ ERROR with opening the predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name ,data_name))
                        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
                        #print(text) 
                        sequence = torch.autograd.Variable(
                        torch.from_numpy(sequence)).cuda().long()
                        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(sequence, hparams)
                        #CB-BL: by me save the original duration
                        _dir = os.path.join(os.getcwd(), 'bissectrice-eval_'+ checkpoint_name + data_name)
                        mel_outputs_postnet = torch.squeeze(mel_outputs_postnet, 0)
                        mel_outputs_postnet_np = mel_outputs_postnet.detach().cpu().numpy()
                        ###***CHANGE Name***###
                        predicted_frame_size = mel_outputs_postnet_np.shape[1]
                        with open(os.path.join(output_save,'predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)), "a+", encoding="utf-8") as file_predicted:
                            file_predicted.write(basename + '|' + str(predicted_frame_size) + "\n")
                        if mel_outputs_postnet_np.shape[1] >= 1000:
                            suprior_file = open(os.path.join(output_save,'Superior_{}_{}.txt'.format(checkpoint_name ,data_name)), "a+", encoding="utf-8")
                            suprior_file.write(mel_filename + "\n")
                    print("Original mel frame length : {}  Predicted mel frame length: {}".format(original_frame_size, predicted_frame_size ))
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
                f.close()
            #plot_data((mel_outputs.float().data.cpu().numpy()[0],
                #mel_outputs_postnet.float().data.cpu().numpy()[0],
                #alignments.float().data.cpu().numpy()[0].T))
                
            #idx = random.randint(0, alignments.size(0) - 1)
            #plot_gate_outputs_to_numpy(
                #torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
    #return mel_outputs_postnet
def plot_mel_frames_length(data_name, checkpoint_name):
    X, Y, Basename = [], [], []
    output_save = os.path.join(os.getcwd(), 'premiere_bissectrice_output')
    
    #for line in open('original_duration_{}_{}_{}.txt'.format(state,data,checkpoint_name)):
    file_original = open(os.path.join(output_save,'original_mel_frames_length_{}_{}.txt'.format(checkpoint_name ,data_name)))
    file = open(os.path.join(output_save,'predicted_mel_frames_length_{}_{}.txt'.format(checkpoint_name , data_name)))

    for line_f_o in file_original:
        half = line_f_o.strip().split('|')
        values = half[-1]
        desc = half[0]
        X.append(int(values))
        Basename.append(desc)
        


    for line_f in file:
        half = line_f.strip().split('|')
        values = half[-1]
        Y.append(int(values))
        
    if len(Y)==len(X):
   
        Group = []
        for i in range(len(X)):
            if X[i] > Y[i]:
                Group.append("inferior")
            if X[i]< Y[i]:
                if Y[i] != 1000:
                    Group.append("superior")
                else :
                    Group.append("error")
            if X[i] == Y[i]:
                Group.append("equal")
            
        print("The shape of X {} , the shape of Y {}".format(len(X), len(Y)))
        fig = plt.figure(figsize=(16, 10))
        plt.xlabel("Mel original frames length")
        plt.ylabel("Mel predicted frames length ")
        plt.tight_layout()
        plt.title("Mel frames length originales vs predites {} {}".format(checkpoint_name , data_name))
        fig.canvas.draw()

        
        # axes= fig.add_axes([0,1000,0,1000])
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(32))
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(32))
        plt.gca().set_aspect(aspect = 'equal')
        
        source = ColumnDataSource(data=dict(x=X, y=Y, desc=Basename, group = Group))
        
        
        # Some example data
        x= Counter(Group)

        
        
        data = pd.DataFrame.from_dict(dict(x), orient='index').reset_index().rename(index=str, columns={0:'value', 'index':'state'})
        data['percent'] = data['value'] / sum(x.values()) * 100
        data['angle'] = data['value'] / sum(x.values()) * 2*pi
        try:
            data['color'] = ['blue','green', 'yellow', 'red']
        except:
            data['color'] = ['green', 'yellow','blue']

        
        p = figure(plot_height=1000, plot_width=1000, tooltips="@state: @percent{0.2f} %")

        p.wedge(x=0, y=1, radius=0.8,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend='state', source=data)

        # show(p)
        # x =  Counter(data=dict(group = Group))
        # print(x['group'])
        # data['percent'] = data['group'] / sum(x.values()) * 100
        # data['angle'] = data['group'] / sum(x.values()) * 2*pi
        # data['color'] = Category20c[len(x)]
        
        # p = figure(plot_height=350, tooltips="@country: @percent{0.2f} %")

        # p.wedge(x=0, y=1, radius=0.4,
        #         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        #         line_color="white", fill_color='color', legend='country', source=data)
        
        # view for just inferior
        inferior = CDSView(source=source, filters=[GroupFilter(column_name='group', group="inferior")])
        
        # view for just superior
        superior = CDSView(source=source, filters=[GroupFilter(column_name='group', group="superior")])
        
        # view for just superior
        error = CDSView(source=source, filters=[GroupFilter(column_name='group', group="error")])
        
        # view for just equal
        equal = CDSView(source=source, filters=[GroupFilter(column_name='group', group="equal")])

        #plt.plot(X[:38000], Y[:38000], 'o')


        # Output the visualization directly in the notebook
        output_file(os.path.join(os.getcwd(), 'originales_vs_predites_mel_length_frames_{}_{}.html'.format(checkpoint_name , data_name)), title='originales_vs_predites_mel_length_frames_{}_{}'.format(checkpoint_name , data_name))

        TOOLTIPS = [
            ("index", "$index"),
            ("(original,predicted)", "($x, $y)"),
            ("basename", "@desc"),
            ("group", "@group"),
            
        ]
        
        # Create a figure with no toolbar and axis ranges of [0,3]
        fig = figure(title= 'originales_vs_predites_mel_length_frames_{}_{}.png'.format(checkpoint_name , data_name),
                    plot_height=1100, plot_width=1100,
                    x_axis_label='Mel original frames length', y_axis_label='Mel predicted frames length',
                    # x_range=(0, 1050), y_range=(0, 1050),
                    tooltips=TOOLTIPS)
        
        x = np.linspace(0 ,1000, 100)
        y1 = x
        # The cumulative sum will be a trend line
        fig.line(x=x, y=y1, 
                color='gray', line_width=1,
                legend='Premiere bissectrice')

        # Draw the coordinates as circles
        fig.circle('x', 'y',
                color='green', size=10, alpha=0.5,source=source, view = inferior, legend = "inferior")
        
        fig.triangle('x', 'y',
                color='blue', size=10, alpha=0.5,source=source, view = superior, legend = "superior")
        
        fig.circle('x', 'y',
                color='yellow', size=10, alpha=0.5,source=source, view = equal, legend = "equal")
        
        fig.square('x', 'y',
                color='red', size=10, alpha=0.5,source=source, view = error, legend = "error")
        
        fig.legend.location = 'bottom_right'

        # Show plot
        show(column(fig,p))
        
        # scatter(X, Y, s=50, c='r', marker='+')
        # plot(X, Y, 'g')
    else:
        print("ERROR SIZE #DIFFERENT The shape of X {} , the shape of Y {}".format(len(X), len(Y)))

if __name__ == '__main__':
    accepted_modes = ['train', 'validation']
    parser = argparse.ArgumentParser(
        description='PREMIERE BISSECTRICE')
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-d', '--dataset', type=str,
                        help='full path to the dataset checkpoint file')
    parser.add_argument('-m', '--mode', type=str,
                        help='mode of run: can be one of {}'.format(accepted_modes))
    args, _ = parser.parse_known_args()
    if args.mode not in accepted_modes and args.dataset is None:
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))
    hparams, dataset, checkpoint_name, data_name, model = config(args.checkpoint_path, args.dataset, args.mode)
    hparams.flag_original = True
    hparams.flag_predicted = True
    hparams.flag_original_saved = True
    hparams.flag_predicted_saved = True
    get_mel_frame(hparams, dataset, checkpoint_name, data_name, model)
    plot_mel_frames_length(data_name, checkpoint_name)
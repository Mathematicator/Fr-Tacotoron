import random
import os
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.filter_mel =  hparams.filter_mel
        self.mel_files = hparams.mel_files
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        #BL:added by me 
        self.max_output_length = hparams.max_output_length
        self.bl_counter = 0
        # BL: end added
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        #Bl :added to wavs prefix to hparams 
        self.wav_dir = hparams.wav_dir
        
        #random.shuffle(self.audiopaths_and_text) # BL: remove rhe shuffle
        #BL added: filter bad length < maimum length
        #BL added to create mel in file
        mel_dir = os.path.join(os.getcwd(), 'mels')
        os.makedirs(mel_dir, exist_ok=True)
        #BL added to save mel in file
        audiopaths_and_text_filtred = []
        if self.filter_mel:
            for index, x in enumerate(self.audiopaths_and_text):
                if not self.load_mel_from_disk:
                    #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                    text_Npath =  x[0].strip().split('/')
                    #take the last part ES_LMP_NEB_02_0011_19
                    basename = text_Npath[-1]
                    mel_filename = 'mel-{}.npy'.format(basename)
                    melsize = self.get_mel(x[0]).size(1)
                    np.save(os.path.join(mel_dir, mel_filename), self.get_mel(x[0]), allow_pickle=False)
                    print("mel file {} saved in {}".format(mel_filename, mel_dir))
                else:
                    melsize = self.get_mel(x[0]).size(1)
                    if melsize <= self.max_output_length:
                        audiopaths_and_text_filtred.append(x)
                    else:
                        print("the rejected mel size {}".format(melsize))
            self.audiopaths_and_text = audiopaths_and_text_filtred

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        # ! BL: added to fixe outputsize
        #Before
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text_torch, cleaned_text  = self.get_text(text)
        mel, basename = self.get_mel(audiopath)
        return (text_torch, mel, basename, cleaned_text )
        
        """
        # BL: after
        melsize = self.get_mel(audiopath_and_text[0]).size(1)
        if melsize <= self.max_output_length:
            audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
            text = self.get_text(text)
            mel = self.get_mel(audiopath)
            return (text, mel)
        else:
            print("Output length {} is superior than max_output_length {}".format(melsize,self.max_output_length))
            return -999
        """    


    def get_mel(self, filename):
        # return the mels by using stft.mel_spectrogram and squezz after 
        if not self.load_mel_from_disk:
            #Bl :added to add wav prefix to the audio 
            basename = filename
            audiopath = self.wav_dir + filename
            audio, sampling_rate = load_wav_to_torch(audiopath+'.wav')
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            parts = filename.strip().split('/')
            basename = parts[-1]
            mel_filename = '{}mel-{}.npy'.format(self.mel_files, basename)
            melspec = torch.from_numpy(np.load(mel_filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec , basename

    def get_text(self, text):
        #BL :added to get cleaned text
        cleaned_text = text
        #BL: end
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm, cleaned_text

    def __getitem__(self, index):
        #BL:modfied
        # Before
        return self.get_mel_text_pair(self.audiopaths_and_text[index])
        
        """
        #Bl modification
        if self.get_mel_text_pair(self.audiopaths_and_text[index]) == -999:
            index = index+1
            self.bl_counter += 1
            print(" the counter of shame {}".format(self.bl_counter))
            return self.get_mel_text_pair(self.audiopaths_and_text[index])
        else:
            index = index-self.bl_counter
            return self.get_mel_text_pair(self.audiopaths_and_text[index])
        """

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        #BL : 
        batch:[text_torch, mel, basename, cleaned_text]
        """
        # Right zero-pad all one-hot text sequences to max input length
        """
        #  BL : added by me 
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch if isinstance(x, (str, list, tuple))]),
            dim=0, descending=True)
        """
        
        #BL added :
        basename = [x[2] for x in batch ]
        cleaned_text = [x[3] for x in batch ]
        #BL: it was : 
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch ]),
            dim=0, descending=True)
        
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            #BL: if added by me 
            if isinstance(batch[ids_sorted_decreasing[i]], (str, list, tuple)):
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text
            else: print("This the vatch causing error: {}".format(batch[ids_sorted_decreasing[i]]))

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        #BL: added by me if isinstance(x, (str, list, tuple))
        max_target_len = max([x[1].size(1) for x in batch if isinstance(x, (str, list, tuple))])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            #BL: if added by me if isinstance(batch[ids_sorted_decreasing[i]] 
            if isinstance(batch[ids_sorted_decreasing[i]], (str, list, tuple)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1)-1:] = 1
                output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, basename, cleaned_text
    

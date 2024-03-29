
from text import text_to_sequence
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import sys
import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from apex import amp
from waveglow.denoiser import Denoiser

from train import load_model
from hparams import create_hparams
import os 


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the csv file (pharases separated by | and new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('-t', '--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-w', '--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true',
                        help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')


    return parser

def split_title_line(title_text, max_words=10):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])
    # %% [markdown]

from scipy.io import wavfile
def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, amp_run, forward_is_infer=False):
    # model_parser = models.parse_model_args(model_name, parser, add_help=False)
    # model_args, _ = model_parser.parse_known_args()

    # model_config = models.get_model_config(model_name, model_args)
    # model = models.get_model(model_name, model_config, to_cuda=True,
    #                          forward_is_infer=forward_is_infer)
    hparams = create_hparams()
    # hparams.n_symbols = 128
    # to add to args
    hparams.max_decoder_steps=2000
    

    if checkpoint is not None and model_name == "Tacotron2":
        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        
        #state_dict = torch.load(checkpoint)['state_dict']
        state_dict = model.load_state_dict(torch.load(checkpoint)['state_dict'])
        # if checkpoint_from_distributed(state_dict):
        #     state_dict = unwrap_distributed(state_dict)
        _ = model.cuda().eval().half()
        
    if checkpoint is not None and model_name == "WaveGlow":
        model = torch.load(checkpoint)['model']
        model.cuda().eval().half()

        model = model.remove_weightnorm(model)

        model.eval()

        if amp_run:
            model.half()
    return model, hparams


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['basic_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if torch.cuda.is_available():
        text_padded = torch.autograd.Variable(text_padded).cuda().long()
        input_lengths = torch.autograd.Variable(input_lengths).cuda().long()
    else:
        text_padded = torch.autograd.Variable(text_padded).long()
        input_lengths = torch.autograd.Variable(input_lengths).long()

    return text_padded, input_lengths




def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    os.makedirs(args.output, exist_ok=True)

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              args.output+'/'+args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2, hparams= load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.amp_run, forward_is_infer=True)
    waveglow, _ = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.amp_run, forward_is_infer=True)
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    #denoiser = Denoiser(waveglow).cuda()


    texts = []
    try:
        f = open(args.input, 'r',encoding="utf-8")
        # if f.endswith(".csv"):
        print("file found in : {} ".format(args.input))
        for line in f:
            if line is not None :
                parts = line.strip().split('|')
                #get the text part easy
                text = parts[1]
                #get the FR-fr_Our/_wav_utt/ES_LMP_NEB_02_0011_19 part which is parts[0]
                text_Npath =  parts[0].strip().split('/')
                #take the last part S_LMP_NEB_02_0011_19
                basename = text_Npath[-1]
                sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
                print(text) 
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(sequence,hparams)
                with torch.no_grad():
                    audio = waveglow.infer(mel_outputs_postnet, sigma=0.6)
                #save audio
                audio_denoised = denoiser(audio, strength=0.05)[:, 0]
                os.makedirs(os.path.join(os.getcwd(),args.output) , exist_ok=True)
                save_wav(audio_denoised[0].data.cpu().numpy(), os.path.join(args.output, 'wav-{}-waveglow-ljs-{}-.wav'.format(basename, "".join(text[0:100]))), sr=22050) 
            else :
                continue
                    
    except:
        print("Error has occured")
        sys.exit(1)
    

if __name__ == '__main__':
    main()

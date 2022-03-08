import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
#BL :added form live audio 
from hparams import create_hparams
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
import sys
sys.path.append('waveglow/')
from waveglow.denoiser import Denoiser




class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, epoch):
            self.add_scalar("training.loss.iteration", reduced_loss, iteration)
            self.add_scalar("training.loss.epoch", reduced_loss, epoch)
            self.add_scalar("grad.norm.iteration", grad_norm, iteration)
            self.add_scalar("grad.norm.epoch", grad_norm,  epoch)
            self.add_scalar("learning.rate.iteration", learning_rate, iteration)
            self.add_scalar("learning.rate.epoch", learning_rate, epoch)
            
            self.add_scalar("duration.iteration", duration, iteration)
            self.add_scalar("duration.epoch", duration, epoch)
            
    def log_global_training(self, reduced_loss, grad_norm, learning_rate, duration,
                    iteration, epoch):
        self.add_scalar("global.training.loss.epoch", reduced_loss, epoch)
            

    def log_validation(self, reduced_loss, model, y, y_pred, iteration, epoch):
        self.add_scalar("validation.loss.iteration", reduced_loss, iteration)
        self.add_scalar("validation.loss.epoch", reduced_loss, epoch)
        
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        
        # audio display 
        hparams = create_hparams()
        
        idx = random.randint(0, alignments.size(0) - 1)
        
        #BL: waveglow load
        waveglow_path = 'waveglow_256channels_ljs_v2_new.pt'
        # waveglow = WaveGlow(80, 12, 8, 4, 
        #   2, {
        #     "n_layers": 8,
        #     "n_channels": 256,
        #     "kernel_size": 3
        # }).cuda()
        #waveglow = torch.load(waveglow_path)
        waveglow = torch.load(waveglow_path)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        
        #waveglow.load_state_dict(torch.load(waveglow_path)['model'])
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        denoiser = Denoiser(waveglow)
        
        #generate audio Waveglow
        with torch.no_grad():
            mel_outputs_Half = mel_outputs.type(torch.cuda.HalfTensor)
            audio = waveglow.infer(mel_outputs_Half, sigma=0.6)
            audio_denoised = denoiser(audio, strength=0.05)[:, 0]
            self.add_audio("audio.waveglow.epoch", audio_denoised[0].data.cpu().numpy(), epoch, hparams.sampling_rate)
            
            

        #BL: griffin Lim
        stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax)
        stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)
        
        inverse_transform = stft.mel_inv_spectrogram(mel_outputs.float().data.cpu())
        audio_grif =griffin_lim(inverse_transform,stft_fn)
        
        #self.add_audio("audio.iteration", audio, iteration, hparams.sampling_rate)
        self.add_audio("audio.griffin.epoch", audio_grif[0].data.cpu().numpy(), epoch, hparams.sampling_rate)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration ) # ,dataformats='HWC'
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)# ,dataformats='HWC'
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration) # ,dataformats='HWC'
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration) # ,dataformats='HWC'

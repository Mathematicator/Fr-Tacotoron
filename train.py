import os
import time
import argparse
import math
from numpy import finfo

#BL log
import infolog
from infolog import log
from time import sleep
import warnings
warnings.filterwarnings("ignore")

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
import multiprocessing
multiprocessing.set_start_method('spawn', True)

#BL 
from text.symbols import symbols




#BL:added for plot reasons
import matplotlib
import numpy as np
import random
import matplotlib.pylab as plt

#BL: to print each step the val 
global_val_loss = -99

#BL log infio
log = infolog.log

#BL: split title line
def split_title_line(title_text, max_words=10):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])
    # %% [markdown]

#BL :added to visualize spectrograms
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
            im = ax1.imshow(target_spectrogram, aspect='auto', origin='bottom', interpolation='none')
        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211) # it was add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram), aspect='auto', interpolation='none')
    else:
        im = ax2.imshow(pred_spectrogram, aspect='auto', origin='bottom', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    if path is not None :
        plt.savefig(path, format='png')
        # To NOT SHOW
        plt.close()

#BL: added to visualize  plot  gates
def plot_gate_outputs(gate_outputs, gate_targets, path=None, title=None, split_title=False):
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
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.7,
            color='green', marker='+', s=3, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=1,
            color='red', marker='.', s=10, label='predicted')

    plt.xlabel("Frames ( Red predicted)")
    plt.ylabel("Gate State")
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()

    fig.canvas.draw()
    
    if path is not None :
        plt.savefig(path, format='png')
        # To NOT SHOW
        plt.close()
    #plt.show()    

#BL:added to visualize alignement 
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



def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    log("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    log("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        #BL: shuffle was True : default
        shuffle = False

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank, checkpoint_path, warm_start, n_gpus, hparams):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
        #BL info log
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
        run_name = str('checkpoint_path:{}   warm_start {}    n_gpus {} '.format(checkpoint_path, warm_start, n_gpus))
        infolog.init(os.path.join(args.output_directory, 'Terminal_train_log'), run_name, str(hparams))
        #BL :Checking the input chain
        #BL log infio
        log("+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+-+")
        log("|L|i|s|t| |o|f| |s|y|m|b|o|l|s|:{}".format(len(symbols)))
        log("+-+-+-+-+ +-+-+ +-+-+-+-+-+-+-+-+-+")
        [x.encode('utf-8') for x in symbols]
        log("{}".format(symbols))   
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    log("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    log("Loading checkpoint '{}'".format(checkpoint_path.encode('utf8', 'replace')))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    log("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    log("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, epoch):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
        
        log("len(val_loader): "+str(val_loader))

        val_loss = 0.0
        for batch_iterator, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            #Bl:added by me 
            _, mel_outputs, gate_outputs, alignments = y_pred
            mel_targets, gate_targets = y
            log(' mel_target_valid shape {} , gate_target_valid_shape {}'.format(mel_outputs.shape, gate_outputs.shape))
            file = open('loss_valid.txt', "a+", encoding="utf-8")
            file.write(str(loss.float().data.cpu().numpy())+"\n")
            log ("the loss by criterion is : {}".format(loss))
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                # Compute and print loss using operations on Tensors.
            # Now loss is a Tensor of shape (1,)
            # loss.item() gets the scalar value held in the loss.
            #BL: added By to visualize mel_outputs, gate_outputs, alignments


                # # plot alignment, mel target and predicted, gate target and predicted
                # for j in range(len(mel_outputs[:,0,0])):
                
                #     idx = random.randint(0, alignments.size(0) - 1)
                #     dir_BL_valid = os.path.join(os.getcwd(), 'Valid_Inside_Visualize')
                #     os.makedirs(dir_BL_valid, exist_ok=True)
                #     title = "loss {} and iteration: {} , index {} ".format(loss, iteration, i)
                #     path = dir_BL_valid + '/' +title

                #     plot_alignment(alignments[j].data.cpu().numpy().T, path+'alignment'+str(j), title) 

                #     plot_spectrogram(mel_outputs[j].data.cpu().numpy(), path+'mel_output'+str(j), title, False, mel_targets[j].data.cpu().numpy())

                #     plot_gate_outputs(torch.sigmoid(gate_outputs[j]).data.cpu().numpy(), torch.sigmoid(gate_targets[j]).data.cpu().numpy(), path+'gate_outputs'+str(j), title)

                # #BL save meloutput alignmment and gateoutput  for validation 
                #     plot_alignment(alignments[j].data.cpu().numpy().T, path+'alignment'+str(j), title) 

                #     plot_spectrogram(mel_outputs[j].data.cpu().numpy(), path+'mel_output'+str(j), title, False, mel_targets[j].data.cpu().numpy())

                #     plot_gate_outputs(torch.sigmoid(gate_outputs[j]).data.cpu().numpy(), torch.sigmoid(gate_targets[j]).data.cpu().numpy(), path+'gate_outputs'+str(j), title)

                #     mel_save_dir = os.path.join(dir_BL_valid, 'mel_save_dir')
                #     align_save_dir = os.path.join(dir_BL_valid, 'align_save_dir')
                #     gate_save_dir = os.path.join(dir_BL_valid, 'gate_save_dir')
                #     os.makedirs(mel_save_dir, exist_ok=True)
                #     os.makedirs(align_save_dir, exist_ok=True)
                #     os.makedirs(gate_save_dir, exist_ok=True)

                #     mel_filename = 'mel-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)
                #     align_filename = 'align-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)
                #     gate_filename = 'gate-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)

                    
                #     np.save(os.path.join(mel_save_dir, mel_filename), mel_outputs[j].data.cpu().numpy(), allow_pickle=False)
                #     np.save(os.path.join(align_save_dir, align_filename), alignments[j].data.cpu().numpy().T, allow_pickle=False)
                #     np.save(os.path.join(gate_save_dir, gate_filename),torch.sigmoid(gate_outputs[j]).data.cpu().numpy() , allow_pickle=False)

                #     log("alignment file {} saved in {}".format(align_filename, align_save_dir))
                #     log("mel file {} saved in {}".format(mel_filename, mel_save_dir))
                #     log("gate file {} saved in {}".format(gate_filename, gate_save_dir))



                #B-L: running_loss += loss.item() #we can calculate the average loss of the epoch
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (batch_iterator + 1)

    model.train()
    #Bl :added to print the valid loss each iteration
    global global_val_loss
    global_val_loss = val_loss
    if rank == 0:
        #print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        log("Validation loss iteration {} epoch {} : {:9f}  ".format(iteration, epoch, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration, epoch)

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams, logger):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    # logger = prepare_directories_and_logger(
    #     output_directory, log_directory, rank, checkpoint_path, warm_start, n_gpus, hparams)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            log("epoch offset: "+str(epoch_offset))
            log("len(train_loader): "+str(len(train_loader)))

    model.train()
    is_overflow = False
    #BL : add the loss_values_train list
    loss_values_train = []

    # ================ MAIN TRAINNIG LOOP! ===================
    log("\n hparams : " +str(hparams) + "\n")
    for epoch in range(epoch_offset, hparams.epochs):
        log("  _______              _                       ___    _______        _       _             ")
        log(" |__   __|            | |                     |__ \  |__   __|      (_)     (_)            ")
        log("    | | __ _  ___ ___ | |_ _ __ ___  _ __ ______ ) |    | |_ __ __ _ _ _ __  _ _ __   __ _ ")
        log("    | |/ _` |/ __/ _ \| __| '__/ _ \| '_ \______/ /     | | '__/ _` | | '_ \| | '_ \ / _` |")
        log("    | | (_| | (_| (_) | |_| | | (_) | | | |    / /_     | | | | (_| | | | | | | | | | (_| |")
        log("    |_|\__,_|\___\___/ \__|_|  \___/|_| |_|   |____|    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |")
        log("                                     &                                                 __/ |")
        log("                                                                                     |___/ ")
        log("                      _        _                             _  _  _  _ __                 ")
        log("                     /_)   .  /_)_   _/__   _/ _/._  _   /  /_`/_)/_)/_//                  ")
        log("                    /_)/_/.  /_)/_|/_///_'/_//_/// //_' /_,/_,/_)/_)/ //                   ")
        log("                       _/                                                                  ")
        log("         +-+-+-+-+-+-+-+-+-+-+ +-+-+  +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+          ") 
        log("         |A|d|a|p|t|a|t|i|o|n| |o|f|  |t|h|e| |N|v|i|d|i|a| |t|a|c|o|t|r|o|n|-|2|          ") 
        log("         +-+-+-+-+-+-+-+-+-+-+ +-+-+  +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+          ")
        log("Epoch: {}".format(epoch))
        mean_train_epoch_loss = 0.0
        for i_batch, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            #BL: added By to visualize mel_outputs, gate_outputs, alignments
            _, mel_outputs, gate_outputs, alignments = y_pred
            mel_targets, gate_targets = y

            # plot alignment, mel target and predicted, gate target and predicted
            idx = random.randint(0, alignments.size(0) - 1)
            dir_BL = os.path.join(os.getcwd(), 'Train_Inside_Visualize')
            os.makedirs(dir_BL, exist_ok=True)
            title = "loss {} and iteration: {} , index {} ".format(loss, iteration, i_batch )
            path = dir_BL + '/' +title
            # if (iteration % len(train_loader) == 0):
            #     for j in range(len(mel_outputs[:,0,0])):
            #         plot_alignment(alignments[j].data.cpu().numpy().T, path+'alignment'+str(j), title) 

            #         plot_spectrogram(mel_outputs[j].data.cpu().numpy(), path+'mel_output'+str(j), title, False, mel_targets[j].data.cpu().numpy())

            #         plot_gate_outputs(torch.sigmoid(gate_outputs[j]).data.cpu().numpy(), torch.sigmoid(gate_targets[j]).data.cpu().numpy(), path+'gate_outputs'+str(j), title)

            #         mel_save_dir = os.path.join(dir_BL, 'mel_save_dir')
            #         align_save_dir = os.path.join(dir_BL, 'align_save_dir')
            #         gate_save_dir = os.path.join(dir_BL, 'gate_save_dir')
            #         os.makedirs(mel_save_dir, exist_ok=True)
            #         os.makedirs(align_save_dir, exist_ok=True)
            #         os.makedirs(gate_save_dir, exist_ok=True)

            #         mel_filename = 'mel-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)
            #         align_filename = 'align-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)
            #         gate_filename = 'gate-{}-{}-{}-{}.npy'.format(loss, iteration, i, j)

                    
            #         np.save(os.path.join(mel_save_dir, mel_filename), mel_outputs[j].data.cpu().numpy(), allow_pickle=False)
            #         np.save(os.path.join(align_save_dir, align_filename), alignments[j].data.cpu().numpy().T, allow_pickle=False)
            #         np.save(os.path.join(gate_save_dir, gate_filename),torch.sigmoid(gate_outputs[j]).data.cpu().numpy() , allow_pickle=False)

            #         log("mel file {} saved in {}".format(mel_filename, mel_save_dir))
            #         log("alignment file {} saved in {}".format(align_filename, align_save_dir))
            #         log("gate file {} saved in {}".format(gate_filename, gate_save_dir))



            #BL: for multi Gpu
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
            #BL: reduced loss 
                reduced_loss = loss.item()
            loss_values_train.append(reduced_loss)
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()
            global global_val_loss
            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                log("Train loss {} {:.6f} Valid loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, global_val_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration, epoch)


        #BL: added : if not is_overflow and rank == 0:
        # if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
            iteration += 1
            mean_train_epoch_loss += reduced_loss
        mean_train_epoch_loss = mean_train_epoch_loss / (i_batch+ 1)
        log("The average epoch global loss {}".format(mean_train_epoch_loss))
        if rank == 0:
            logger.log_global_training(mean_train_epoch_loss, grad_norm, learning_rate, duration, iteration, epoch)
        
            
        if not is_overflow:
            log("#### Validation #####")
            validate(model, criterion, valset, iteration,
                        hparams.batch_size, n_gpus, collate_fn, logger,
                        hparams.distributed_run, rank, epoch)
            if rank == 0:
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    #BL log infio
    log = infolog.log
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    #hparams + '\n'
    logger = prepare_directories_and_logger(
        args.output_directory, args.log_directory, args.rank, args.checkpoint_path, args.warm_start, args.n_gpus, hparams)
    log("                                                                                              ")
    log("██╗███╗   ██╗██╗████████╗██╗ █████╗ ██╗     ██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗")
    log("██║████╗  ██║██║╚══██╔══╝██║██╔══██╗██║     ██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║")
    log("██║██╔██╗ ██║██║   ██║   ██║███████║██║     ██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║")
    log("██║██║╚██╗██║██║   ██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║")
    log("██║██║ ╚████║██║   ██║   ██║██║  ██║███████╗██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║")
    log("╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝")
    log("FP16 Run: {} ".format(hparams.fp16_run))
    log("Dynamic Loss Scaling: {}".format(hparams.dynamic_loss_scaling))
    log("Distributed Run: {}".format(hparams.distributed_run))
    log("cuDNN Enabled: {}".format(hparams.cudnn_enabled))
    log("cuDNN Benchmark: {}".format(hparams.cudnn_benchmark))

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams, logger)

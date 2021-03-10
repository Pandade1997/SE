# --batch_size=64 --dropout_p=0.2 --attn_use=True --stacked_encoder=True --attn_len=5 --hidden_size=448 --num_epochs=61
import os
import argparse
import sys

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

import tensorboardX
from tensorboardX import SummaryWriter

from scipy.io import wavfile
import librosa

import soundfile as sf
from pystoi.stoi import stoi
from pypesq import pesq

from tqdm import tqdm
from models.layers.istft import ISTFT
import train_utils
from load_dataset import AudioDataset
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiment/SE_model.json', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=1000, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default=0.2, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--attn_use', default=True, type=bool)
parser.add_argument('--stacked_encoder', default=True, type=bool)
parser.add_argument('--attn_len', default=5, type=int)
parser.add_argument('--hidden_size', default=448, type=int)
parser.add_argument('--ck_name', default='final.pt')  # modify
args = parser.parse_args()

n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()


def normalized(tensor):
    output = [[] for i in range(len(tensor))]

    for i in range(len(tensor)):
        nummer = tensor[i] - torch.min(tensor[i])
        denomi = torch.max(tensor[i]) - torch.min(tensor[i])

        output[i] = (nummer / (denomi + 1e-5)).tolist()

    return torch.tensor(output)


def main():
    summary = SummaryWriter('./log')
    # os.system('tensorboard --logdir=log')
    #
    # set Hyper parameter
    # json_path = os.path.join(args.model_dir)
    # params = train_utils.Params(json_path)

    # data loader
    train_dataset = AudioDataset(data_type='train')
    # modify:num_workers=4
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate,
                                   shuffle=True, num_workers=6)
    test_dataset = AudioDataset(data_type='valid')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
                                  shuffle=False, num_workers=6)

    # # data loader
    # train_dataset = AudioDataset(data_type='test')
    # # modify:num_workers=4
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate,
    #                                shuffle=True, num_workers=0)
    # test_dataset = AudioDataset(data_type='test')
    # test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
    #                               shuffle=False, num_workers=0)
    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(257, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = arg0s.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    scheduler = ExponentialLR(optimizer, 0.5)

    # check point load
    # Check point load

    print('Trying Checkpoint Load\n')
    # ckpt_dir = 'ckpt_dir_stoi'
    ckpt_dir = 'ckpt_dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_PESQ = 0.
    best_STOI = 0.
    best_loss = 200000.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_loss = ckpt['best_loss']

            print('checkpoint is loaded !')
            print('current best loss : %.4f' % best_loss)
        except RuntimeError as e:
            print('wrong checkpoint\n')
    else:
        print('checkpoint not exist!')
        print('current best loss : %.4f' % best_loss)

    print('Training Start!')
    # train
    iteration = 0
    train_losses = []
    test_losses = []
    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        # train_bar = train_data_loader876\
        n = 0
        avg_loss = 0
        avg_pesq = 0
        avg_stoi = 0
        net.train()
        for input in train_bar:
            iteration += 1
            # load data
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)

            mixed = stft(train_mixed)
            cleaned = stft(train_clean)
            mixed = mixed.transpose(1, 2)
            cleaned = cleaned.transpose(1, 2)
            real, imag = mixed[..., 0], mixed[..., 1]
            clean_real, clean_imag = cleaned[..., 0], cleaned[..., 1]
            mag = torch.sqrt(real ** 2 + imag ** 2)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
            phase = torch.atan2(imag, real)

            # feed data
            out_mag, attn_weight = net(mag)
            out_real = out_mag * torch.cos(phase)
            out_imag = out_mag * torch.sin(phase)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            out_real = out_real.transpose(1, 2)
            out_imag = out_imag.transpose(1, 2)

            out_audio = istft(out_real, out_imag, train_mixed.size(1))
            out_audio = torch.squeeze(out_audio, dim=1)
            for i, l in enumerate(seq_len):
                out_audio[i, l:] = 0

            loss = 0
            PESQ = 0
            STOI = 0
            origin_PESQ = 0
            origin_STOI = 0

            loss = F.mse_loss(out_mag, clean_mag, True)
            if torch.any(torch.isnan(loss)):
                torch.save({'clean_mag': clean_mag, 'out_mag': out_mag, 'mag': mag}, 'nan_mag')
                raise ('loss is NaN')
            avg_loss += loss.item()
            n += 1
            # gradient optimizer
            optimizer.zero_grad()

            # backpropagate LOSS20+

            loss.backward()

            # update weight
            optimizer.step()

        avg_loss /= n
        avg_pesq /= n
        avg_stoi /= n
        print('result:')
        print(
            '[epoch: {}, iteration: {}] avg_loss : {:.4f} avg_pesq : {:.4f} avg_stoi : {:.4f} '.format(epoch, iteration,
                                                                                                       avg_loss,
                                                                                                       avg_pesq,
                                                                                                       avg_stoi))

        summary.add_scalar('Train Loss', avg_loss, iteration)

        train_losses.append(avg_loss)
        if (len(train_losses) > 2) and (train_losses[-2] < avg_loss):
            print("Learning rate Decay")
            scheduler.step()

        # test phase
        n = 0
        avg_test_loss = 0
        avg_test_pesq = 0
        avg_test_stoi = 0
        test_bar = tqdm(test_data_loader)

        net.eval()
        with torch.no_grad():
            for input in test_bar:
                test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)
                mixed = stft(test_mixed)
                cleaned = stft(test_clean)
                mixed = mixed.transpose(1, 2)
                cleaned = cleaned.transpose(1, 2)
                real, imag = mixed[..., 0], mixed[..., 1]
                clean_real, clean_imag = cleaned[..., 0], cleaned[..., 1]
                mag = torch.sqrt(real ** 2 + imag ** 2)
                clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
                phase = torch.atan2(imag, real)

                logits_mag, logits_attn_weight = net(mag)
                logits_real = logits_mag * torch.cos(phase)
                logits_imag = logits_mag * torch.sin(phase)
                logits_real, logits_imag = torch.squeeze(logits_real, 1), torch.squeeze(logits_imag, 1)
                logits_real = logits_real.transpose(1, 2)
                logits_imag = logits_imag.transpose(1, 2)

                logits_audio = istft(logits_real, logits_imag, test_mixed.size(1))
                logits_audio = torch.squeeze(logits_audio, dim=1)
                for i, l in enumerate(seq_len):
                    logits_audio[i, l:] = 0

                test_PESQ = 0
                test_STOI = 0

                test_loss = F.mse_loss(logits_mag, clean_mag, True)

                avg_test_loss += test_loss.item()
                n += 1

            avg_test_loss /= n
            avg_test_pesq /= n
            avg_test_stoi /= n

            test_losses.append(avg_test_loss)
            summary.add_scalar('Test Loss', avg_test_loss, iteration)

            print('[epoch: {}, iteration: {}] test loss : {:.4f} avg_test_pesq : {:.4f} avg_test_stoi : {:.4f}'.format(
                epoch, iteration, avg_test_loss, avg_test_pesq, avg_test_stoi))
            if avg_test_loss < best_loss:
                best_PESQ = test_PESQ
                best_STOI = test_STOI
                best_loss = avg_test_loss
                # Note: optimizer also has states ! don't forget to save them as well.
                ckpt = {'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss}
                torch.save(ckpt, ckpt_path)
                print('checkpoint is saved !')


if __name__ == '__main__':
    main()

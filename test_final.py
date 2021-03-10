import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
from pystoi.stoi import stoi
from pypesq import pesq
from tqdm import tqdm
from models.layers.istft import ISTFT
from load_dataset import AudioDataset
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default=0.2, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--stacked_encoder', default=True, type=bool)
parser.add_argument('--attn_len', default=5, type=int)
parser.add_argument('--hidden_size', default=448, type=int)
parser.add_argument('--ck_dir', default='ckpt_dir', help='ck path')
parser.add_argument('--ck_name', help='ck file', default='test2.pt')
parser.add_argument('--test_set', default='train')
parser.add_argument('--attn_use', default=True, type=bool)
parser.add_argument('--out_path',
                    default='/data01/AuFast/Pan_dataset/SE_asr/finaltest/gen_flac/libri_train/', type=str)

args = parser.parse_args()

n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()


def main():
    test_dataset = AudioDataset(data_type=args.test_set)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
                                  shuffle=False, num_workers=4)

    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(257, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = args.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = args.ck_dir
    best_PESQ = 0.
    best_STOI = 0.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # best_STOI = ckpt['best_STOI']

            print('checkpoint is loaded !')
            # print('current best loss : %.4f' % best_loss)
        except RuntimeError as e:
            print('wrong checkpoint\n')
    else:
        print('checkpoint not exist!')
        # print('current best loss : %.4f' % best_loss)

    # test phase
    n = 0
    name_sum = 0
    avg_test_loss = 0

    net.eval()
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
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

            test_loss = 0
            test_PESQ = 0
            test_STOI = 0
            origin_test_PESQ = 0
            origin_test_STOI = 0

            test_loss = F.mse_loss(logits_mag, clean_mag, True)

            for i in range(len(test_mixed)):
                name = str(test_dataset.file_names[name_sum]).split('/', 9)[8]
                librosa.output.write_wav(args.out_path + name,
                                         logits_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()],
                                         16000)
                cur_PESQ = pesq(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000)
                cur_STOI = stoi(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000,
                                extended=False)
                origin_cur_PESQ = pesq(test_clean[i].detach().cpu().numpy(), test_mixed[i].detach().cpu().numpy(),
                                       16000)
                origin_cur_STOI = stoi(test_clean[i].detach().cpu().numpy(), test_mixed[i].detach().cpu().numpy(),
                                       16000,
                                       extended=False)

                test_PESQ += cur_PESQ
                test_STOI += cur_STOI
                origin_test_PESQ += origin_cur_PESQ
                origin_test_STOI += origin_cur_STOI

                name_sum += 1

            test_PESQ /= len(test_mixed)
            test_STOI /= len(test_mixed)
            origin_test_PESQ /= len(test_mixed)
            origin_test_STOI /= len(test_mixed)
            avg_test_loss += test_loss
            n += 1

        avg_test_loss /= n
        print(
            'test loss : {:.4f} origin_test_PESQ : {:.4f} origin_test_STOI : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(
                avg_test_loss, origin_test_PESQ, origin_test_STOI,
                test_PESQ, test_STOI))


if __name__ == '__main__':
    main()

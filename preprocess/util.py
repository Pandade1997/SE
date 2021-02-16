import os
import sys
from tqdm import tqdm
from pathlib import Path
import subprocess
import numpy as np
import torch

# from config import FFT_SIZE

EPSILON = 1e-7


def dir_format_change(src_dir, dest_dir, src_suffix, dest_suffix='wav'):
    """
    dir_format_change('/home/lx/data/LibriSpeech/',
                      '/home/lx/data/LibriSpeech_wav/',
                      'flac', 'wav')
    转换格式 生成文件夹结构与源文件夹结构完全相同 需要安装ffmpeg
    :param src_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    :param src_suffix: 源文件后缀
    :param dest_suffix: 目标文件后缀
    :return:
    """
    src_list = list(Path(src_dir).rglob('*.' + src_suffix))
    for src_file in tqdm(src_list):
        dest_file = dest_dir + str(src_file).partition(src_dir)[2].replace(src_suffix, dest_suffix)

        if not os.path.exists(dest_file.rpartition('/')[0]):
            os.makedirs(dest_file.rpartition('/')[0])

        if os.path.exists(dest_file):
            continue

        format_change(src_file, dest_file)


def format_change(src_file, dest_file):
    cmd = 'ffmpeg -i ' + str(src_file) + ' ' + str(dest_file)
    # os.system(cmd)
    subprocess_call(cmd)


def normalize(speech, noise):
    temp_mixture = noise + speech
    alpha_pow = 1 / (np.sqrt(np.sum(temp_mixture ** 2) / len(speech)) + EPSILON)

    speech = alpha_pow * speech
    noise = alpha_pow * noise

    mix = speech + noise

    mix = mix / np.max(mix)

    return mix


def mix2signal(sig1, sig2, snr):
    return np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + EPSILON)) / 10.0 ** (snr / 10.0))


def subprocess_call(*args, **kwargs):
    # also works for Popen. It creates a new *hidden* window, so it will work in frozen apps (.exe).
    if 'win32' in str(sys.platform).lower():
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
    return subprocess.call(*args, **kwargs)


def gen_list(src_dir, dest_dir, dest_filename, src_suffix='wav'):
    """
    gen_list('/home/lx/data/WSJ0/WSJ0_wav/', '/home/lx/data/WSJ0/', 'gen_train.lst')
    :param src_dir:
    :param dest_dir:
    :param dest_filename:
    :param src_suffix:
    :return:
    """
    file = open(dest_dir + dest_filename, "w")
    src_list = list(Path(src_dir).rglob('*.' + src_suffix))
    for src_file in src_list:
        file.writelines(str(src_file) + '\n')
    file.close()

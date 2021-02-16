import os
import random
import numpy as np
from pathlib import Path
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.io as sio
import csv
import pre_config as cfg
import util as util

N_JOBS = 100  # 设置任务并行数
EPSILON = 1e-7


class AddNoise:
    def __init__(self, speech_dir, noise_dir, dest_dir, speech_suffix, noise_suffix='wav',
                 dest_suffix='flac', mat_output_dir=None, use_noise_type=cfg.TRAIN_NOISE):
        """
        :param speech_dir: 源语音的根文件夹
        :param noise_dir:使用的noise是NoiseX-92
        :param dest_dir: 目标存储位置
        :param speech_suffix: 源文件的后缀
        :param mat_output_dir: .mat存储位置
        """
        super(AddNoise, self).__init__()
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.dest_dir = dest_dir
        self.src_suffix = speech_suffix
        self.dest_suffix = dest_suffix
        # self.mat_output_dir = mat_output_dir
        self.mat_output_dir = None
        self.mk_dir()

        self.speech_list = list(Path(speech_dir).rglob('*.' + speech_suffix))
        noise_list_all = list(Path(noise_dir).rglob("*." + noise_suffix))

        print(len(self.speech_list), 'audio files found in', self.speech_dir)
        print(len(noise_list_all), 'noise files found in', self.noise_dir)

        self.noise_list = [noise for noise in noise_list_all]

        random.shuffle(self.speech_list)

    def mix_audio(self, gen_type, gen_num=None):
        # self.add_noise(self.speech_list[0], self.noise_list[0], snr=cfg.SNR[0], rename=False)  # test

        if gen_type == cfg.GEN_TYPES[0]:  # x times 生成x倍数量混合数据 x=len(noise_list) * len(SNR)
            tr_x = Parallel(n_jobs=N_JOBS) \
                (delayed(self.add_noise)(speech_path=speech_path, noise_path=noise_path,
                                         snr=snr, change_name=True)
                 for speech_path in tqdm(self.speech_list)
                 for snr in cfg.SNR
                 for noise_path in self.noise_list)
        elif gen_type == cfg.GEN_TYPES[1]:  # mix 生成相同数量的混合数据
            tr_x = Parallel(n_jobs=N_JOBS) \
                (delayed(self.add_noise)(speech_path=speech_path,
                                         noise_path=self.noise_list[random.randint(0, len(self.noise_list) - 1)],
                                         snr=cfg.SNR[random.randint(0, len(cfg.SNR) - 1)],
                                         change_name=False)
                 for speech_path in tqdm(self.speech_list))
        else:
            tr_x = Parallel(n_jobs=N_JOBS) \
                (delayed(self.add_noise)(speech_path=self.speech_list[random.randint(0, len(self.speech_list) - 1)],
                                         noise_path=self.noise_list[random.randint(0, len(self.noise_list) - 1)],
                                         snr=cfg.SNR[random.randint(0, len(cfg.SNR) - 1)],
                                         change_name=True)
                 for _ in tqdm(range(gen_num)))

        headers = ['filename', 'noise', 'start', 'SNR']
        rows = []
        for row in tr_x:
            rows.append(row)
        with open(self.dest_dir[:-1] + '-list.csv', 'w', newline='') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(rows)

        return len(rows)

    def add_noise(self, speech_path, noise_path, snr, change_name=False):
        noise = sf.read(noise_path)[0]
        speech = sf.read(speech_path)[0]

        assert len(speech) < len(noise)

        start = random.randint(0, len(noise) - len(speech))
        noise = noise[slice(start, start + len(speech))]

        if change_name:
            output_dir = self.get_save_dir(str(speech_path), snr,
                                           str(noise_path).split('/')[-1].split('.')[0], str(start))
        else:
            output_dir = self.get_save_dir(str(speech_path))
        mix_filename = str(output_dir).rpartition('/')[-1]

        rows = {'filename': mix_filename,
                'noise': str(noise_path).split('/')[-1].split('.')[0],
                'start': start, 'SNR': snr}

        alpha = util.mix2signal(np.array(speech), noise, snr)
        noise = noise * alpha

        # util.normalize(speech,noise)
        mix = noise + speech
        sf.write(output_dir, mix.tolist(), cfg.SAMPLE_RATE)

        if self.mat_output_dir is not None:
            sio.savemat(os.path.join(self.mat_output_dir, str(mix_filename).partition('.')[0] + '.mat'),
                        {'speech': speech.reshape(-1, 1), 'noise': noise.reshape(-1, 1)})

        return rows

    def get_save_dir(self, speech_name: str, snr: str = None, noise_name: str = None, noise_start: str = None):
        speech_split = speech_name.split('/')
        save_dir = os.path.join(self.dest_dir, str(speech_name).partition(self.speech_dir)[2].rpartition('/')[0])

        speech_name = speech_split[-1].split('.')[0]
        if snr is not None:
            speech_name = speech_name + '_' + str(snr)
        if noise_name is not None:
            speech_name = speech_name + '_' + noise_name
        if noise_start is not None:
            speech_name = speech_name + '-' + noise_start
        speech_name = speech_name + '.' + self.dest_suffix

        return os.path.join(save_dir, speech_name)

    def mk_dir(self):
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

        if self.mat_output_dir is not None:
            if not os.path.exists(self.mat_output_dir):
                os.makedirs(self.mat_output_dir)

        src_list = list(Path(self.speech_dir).rglob('*.' + self.src_suffix))
        for src_file in src_list:
            dest_file_dir = self.dest_dir + str(src_file).partition(self.speech_dir)[2].rpartition('/')[0]
            if not os.path.exists(dest_file_dir):
                os.makedirs(dest_file_dir)

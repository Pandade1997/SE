import os

GEN_TYPES = ['x_times',  # x times 生成x倍数量混合数据 x=len(noise_list) * len(SNR)
             'mix',  # mix 生成相同数量的混合数据
             'gen_num']  # 生成GEN_NUM条数据
GEN_TYPE = GEN_TYPES[1]

# SPEECH_DIR = '/home/lx/data/LibriSpeech/LibriSpeech_wav/'
# NOISE_DIR = '/home/lx/data/NoiseX-92/NoiseX-92-16000/'
# MIX_OUTPUT_DIR = '/home/lx/data/LibriSpeech/LibriSpeech_mix/'
# MAT_OUTPUT_DIR = '/home/lx/data/LibriSpeech/gen_mat/'

# SPEECH_DIR = '/data01/AuFast/Pan_dataset/Exp3_SE/data/creat/'
# NOISE_DIR = '/data01/AuFast/Pan_dataset/Exp3_SE/NoiseX-92-16000'
# MIX_OUTPUT_DIR = '/data01/AuFast/Pan_dataset/Exp3_SE/data/output/'
# MAT_OUTPUT_DIR = '/data01/AuFast/Pan_dataset/Exp3_SE/data/creat/'

SPEECH_DIR = '/data01/AuFast/origin_dataset/dataset/LibriSpeech/'
NOISE_DIR = '/data01/AuFast/Pan_dataset/Exp3_SE/data3/Test_NoiseX-92-16000'
# MIX_OUTPUT_DIR = '/data01/AuFast/origin_dataset/dataset/LibriSpeech/test_dataset/SE/'
# MAT_OUTPUT_DIR = '/data01/AuFast/origin_dataset/dataset/LibriSpeech/test_dataset/SE/'
MIX_OUTPUT_DIR = '/data01/AuFast/Pan_dataset/SE_asr/add_noisy_libri/'
MAT_OUTPUT_DIR = '/data01/AuFast/Pan_dataset/SE_asr/add_noisy_libri/'

TR_DIR = MAT_OUTPUT_DIR + 'train/'
CV_DIR = MAT_OUTPUT_DIR + 'dev/'
TT_DIR = MAT_OUTPUT_DIR + 'test/'

SNR = [-5, 0, 5]
NOISE_list = os.path.basename('/data01/AuFast/Pan_dataset/Exp3_SE/data3/Test_NoiseX-92-16000')
NOISE_list = NOISE_list.split('.')[0]

TRAIN_NOISE = os.path.basename('/data01/AuFast/Pan_dataset/Exp3_SE/data3/Test_NoiseX-92-16000/')
TRAIN_NOISE = TRAIN_NOISE.split('.')[0]

NOISE_SUFFIX = 'wav'
SPEECH_SUFFIX = 'flac'
SAMPLE_RATE = 16000

GEN_TR_NUM = 13000
GEN_CV_NUM = 4000
GEN_TT_NUM = 3000

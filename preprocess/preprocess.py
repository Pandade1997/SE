import sys
import pre_config as cfg
from do_add_noise import AddNoise

Libri_TYPE = ['dev-clean', 'train-clean-100', 'train-clean-360', 'test-clean']
WSJ0_TYPE = ['trainset_clean', 'validset_clean', 'testset_clean']


def libri_start():
    # 指定数据集
    libri_types = [
        # Libri_TYPE[0],
        # Libri_TYPE[1],
        # Libri_TYPE[2],
        Libri_TYPE[3]
    ]

    # start...
    for libri_type in libri_types:
        print('gen', libri_type, 'data...', flush=True)

        speech_dir = cfg.SPEECH_DIR + libri_type + '/'
        # noise_type = cfg.TEST_NOISE if 'test' in libri_type else cfg.TRAIN_NOISE
        noise_type = cfg.NOISE_list
        if 'test' in libri_type:
            mat_output_dir = cfg.TT_DIR
            gen_num = cfg.GEN_TT_NUM
        elif 'dev' in libri_type:
            mat_output_dir = cfg.CV_DIR
            gen_num = cfg.GEN_CV_NUM
        else:
            mat_output_dir = cfg.TR_DIR
            gen_num = cfg.GEN_TR_NUM

        # mk_dir(speech_list, audio_output_path, libri_type)
        dest_dir = cfg.MIX_OUTPUT_DIR + libri_type + '/'

        add_noise = AddNoise(speech_dir, cfg.NOISE_DIR, dest_dir, cfg.SPEECH_SUFFIX,
                             mat_output_dir=mat_output_dir, use_noise_type=noise_type)
        num = add_noise.mix_audio(cfg.GEN_TYPE, gen_num)
        print('files saved at', dest_dir, 'generate num =', num)


def wsj_start():
    wsj0_types = [
        # WSJ0_TYPE[0],
        # WSJ0_TYPE[1],
        WSJ0_TYPE[2]
    ]

    for wsj0_type in wsj0_types:
        print('gen', wsj0_type, 'data...', flush=True)

        speech_dir = cfg.SPEECH_DIR + wsj0_type + '/'
        noise_type = cfg.NOISE_list

        if 'testset_clean' in wsj0_type:
            mat_output_dir = cfg.TT_DIR
            gen_num = cfg.GEN_TT_NUM
        elif 'validset_clean' in wsj0_type:
            mat_output_dir = cfg.CV_DIR
            gen_num = cfg.GEN_CV_NUM
        else:
            mat_output_dir = cfg.TR_DIR
            gen_num = cfg.GEN_TR_NUM

        dest_dir = cfg.MIX_OUTPUT_DIR + wsj0_type + '/'

        add_noise = AddNoise(speech_dir, cfg.NOISE_DIR, dest_dir, cfg.SPEECH_SUFFIX,
                             mat_output_dir=mat_output_dir, use_noise_type=noise_type)
        num = add_noise.mix_audio(cfg.GEN_TYPE, gen_num)
        print('files saved at', dest_dir, 'generate num =', num)


if __name__ == '__main__':
    # gen_list('/home/lx/data/WSJ0/WSJ0_wav/', '/home/lx/data/WSJ0/', 'gen_train.lst')
    # gen_list('/home/lx/data/WSJ0/WSJ0_wav/', '/home/lx/data/WSJ0/', 'gen_train.lst')
    # gen_list('F:/audio/WSJ0_wav/wsj0/', 'F:/audio/WSJ0_wav/', 'gen_wsj_train.l    pst')
    libri_start()
    # wsj_start()
    sys.exit()

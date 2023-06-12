"""
Script to create datasets as specified in https://github.com/lk-mt-sc/thesis/wiki/Datasets.
Prerequisites:
a) videos in FullHD and .mp4 file format stored in ./videos named after their YouTube-ID
b) FFmpeg installed and accessible from the command line
"""
import os
import glob
import math
import random
import shutil
import argparse
import subprocess

import torch
import imageio
import cv2 as cv
import numpy as np
from tqdm import tqdm
from basicsr.models.archs.gshift_deblur1 import GShiftNet
from ext_nets.ema_vfi.interpolator import Interpolator


def time_to_ms(time):
    h = int(time[0:2])
    m = int(time[3:5])
    s = int(time[6:8])
    ms = int(time[9:])

    h_ms = h * 60 * 60 * 1000
    m_ms = m * 60 * 1000
    s_ms = s * 1000
    return h_ms + m_ms + s_ms + ms


def create_std_dataset(id_start):
    if os.path.exists(STD_DATASET_DIR):
        shutil.rmtree(STD_DATASET_DIR)

    os.mkdir(STD_DATASET_DIR)
    os.mkdir(STD_RUNS_VIDEO_DIR)
    os.mkdir(STD_RUNS_IMAGE_DIR)

    markdown_table_entries = []

    id_ = id_start
    print('Creating standard dataset...')
    with open(os.path.join('dataset.CSV'), 'r', encoding='utf8') as metadata:
        runs = metadata.readlines()
        for run in tqdm(runs[1:]):
            run_strip = run.strip()
            run_split = run_strip.split(';')

            competition_id = int(run_split[0])
            competition_name = run_split[1]
            competition_url = run_split[2]
            youtube_channel = run_split[3]
            video_id = run_split[4]
            video_fps = int(run_split[5])
            video_res = run_split[6]
            video_bitrate = run_split[7]
            climber_id = int(run_split[8])
            climber_name = run_split[9]
            start_time = run_split[10]
            end_time = run_split[11]
            start_frame = int(run_split[12])
            end_frame = int(run_split[13])
            frames_uncut = int(run_split[14])
            frames_cut = int(run_split[15])
            left_or_right_climber = run_split[16]
            spotlight = run_split[17]
            gender = run_split[18]
            offset = int(run_split[19])
            start_at_rest = run_split[20]

            markdown_table_entry = '|'
            markdown_table_entry += str(competition_id) + '|'
            markdown_table_entry += competition_name.replace('||', '-') + '|'
            markdown_table_entry += competition_url + '|'
            markdown_table_entry += youtube_channel + '|'
            markdown_table_entry += video_id + '|'
            markdown_table_entry += str(video_fps) + '|'
            markdown_table_entry += video_res + '|'
            markdown_table_entry += video_bitrate + '|'
            markdown_table_entry += str(climber_id) + '|'
            markdown_table_entry += climber_name + '|'
            markdown_table_entry += start_time + '|'
            markdown_table_entry += end_time + '|'
            markdown_table_entry += str(start_frame) + '|'
            markdown_table_entry += str(end_frame) + '|'
            markdown_table_entry += str(frames_uncut) + '|'
            markdown_table_entry += str(frames_cut) + '|'
            markdown_table_entry += left_or_right_climber + '|'
            markdown_table_entry += spotlight + '|'
            markdown_table_entry += gender + '|'
            markdown_table_entry += str(offset) + '|'
            markdown_table_entry += start_at_rest + '|'
            markdown_table_entries.append(markdown_table_entry)

            start_time_ms = time_to_ms(start_time)
            end_time_ms = time_to_ms(end_time)
            video_path = os.path.join(VIDEOS_DIR, video_id + '.mp4')

            out_str = []
            out_str.append(f'{str(id_).zfill(3)} ID')
            out_str.append(f'{str(frames_cut).zfill(4)} IMG')
            if spotlight == 'yes':
                out_str.append('SPOTLIGHT')
            elif spotlight == 'no':
                out_str.append('NO SPOTLIGHT')
            if start_at_rest == 'yes':
                out_str.append('START AT REST')
            elif start_at_rest == 'no':
                out_str.append('START IN MOTION')
            out_str.append(f'{str(video_fps).zfill(3)} FPS')
            out_str = ' - '.join(out_str)
            out_path = os.path.join(STD_RUNS_VIDEO_DIR, out_str + '.mp4')

            subprocess_args = ['ffmpeg', '-ss', f'{start_time_ms}ms',
                               '-to', f'{end_time_ms}ms', '-i', video_path, out_path]
            subprocess.check_call(subprocess_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            out_folder = os.path.join(STD_RUNS_IMAGE_DIR, out_str)
            os.mkdir(out_folder)
            run_video = cv.VideoCapture(out_path)

            img_id = 0
            out_id = 0
            while True:
                ret, frame = run_video.read()
                if not ret:
                    break
                if img_id < start_frame:
                    img_id += 1
                    continue
                if img_id > end_frame:
                    break
                out_path = os.path.join(out_folder, f'{str(out_id).zfill(3)}.png')
                width = frame.shape[1]
                if left_or_right_climber == 'L':
                    frame = frame[:, offset:int(offset+width/2)]
                if left_or_right_climber == 'R':
                    frame = frame[:, int(width/2-offset):int(width-offset)]
                cv.imwrite(out_path, frame)
                img_id += 1
                out_id += 1

            id_ += 1

        # for entry in markdown_table_entries:
        #     print(entry)


def create_deb_dataset(id_start):
    if not os.path.exists(STD_DATASET_DIR):
        print('Standard dataset not found.')
        create_std_dataset()

    if os.path.exists(DEB_DATASET_DIR):
        shutil.rmtree(DEB_DATASET_DIR)

    os.mkdir(DEB_DATASET_DIR)

    device = 'cuda:0'
    checkpoint = os.path.join(SHIFT_NET_DIR, 'ckpt', 'net_gopro_deblur.pth')

    net = GShiftNet(future_frames=2, past_frames=2)
    net.load_state_dict(torch.load(checkpoint)['params'])
    net.half()
    net = net.to(device)
    net.eval()

    id_ = id_start
    print('Creating deblurred dataset...')
    runs = sorted(glob.glob(os.path.join(STD_RUNS_IMAGE_DIR, '*')))
    for run in tqdm(runs, desc='Overall'):
        out_str = run.split('/')[-1]
        out_str_split = out_str.split(' - ')
        out_str_split[0] = f'{str(id_).zfill(3)} ID'
        out_str_split.append('DEBLURRED')
        out_str = ' - '.join(out_str_split)
        out_dir = os.path.join(DEB_DATASET_DIR, out_str)
        os.mkdir(out_dir)

        frames = sorted(glob.glob(os.path.join(run, '*')))
        frames.insert(0, frames[0])
        frames.insert(0, frames[0])
        frames.append(frames[-1])
        frames.append(frames[-1])

        inputs = []
        for i in range(2, len(frames) - 2):
            input_ = []
            input_.append(frames[i - 2])
            input_.append(frames[i - 1])
            input_.append(frames[i])
            input_.append(frames[i + 1])
            input_.append(frames[i + 2])
            inputs.append(input_)

        with torch.no_grad():
            for i, input_ in enumerate(tqdm(inputs, desc='Current', leave=False)):
                images = [imageio.v2.imread(input_[i]) for i in range(0, len(input_))]
                images = numpy2tensor(images).to(device)
                outputs = net(images.half())
                outputs = outputs.float()
                outputs = tensor2numpy(outputs)
                cv.imwrite(os.path.join(out_dir, str(i).zfill(3) + '.png'), outputs[..., ::-1])

        torch.cuda.empty_cache()

        id_ += 1


def create_itp_dataset(id_start):
    if not os.path.exists(STD_DATASET_DIR):
        print('Standard dataset not found.')
        create_std_dataset()

    if os.path.exists(ITP_DATASET_DIR):
        shutil.rmtree(ITP_DATASET_DIR)

    os.mkdir(ITP_DATASET_DIR)

    n = 5
    interpolator = Interpolator(n=n)

    id_ = id_start
    print('Creating interpolated dataset...')
    runs = sorted(glob.glob(os.path.join(STD_RUNS_IMAGE_DIR, '*')))
    for run in tqdm(runs, desc='Overall'):
        frames = sorted(glob.glob(os.path.join(run, '*')))
        n_frames = len(frames)

        old_run_duration_sec = n_frames / 25.0
        n_frames_interpolated = (n_frames - 1) * n + 1
        new_fps = int(math.ceil(n_frames_interpolated / old_run_duration_sec))

        out_str = run.split('/')[-1]
        out_str_split = out_str.split(' - ')
        out_str_split[0] = f'{str(id_).zfill(3)} ID'
        out_str_split[1] = f'{str(n_frames_interpolated).zfill(4)} IMG'
        out_str_split[4] = f'{str(new_fps).zfill(3)} FPS'
        out_str_split.append('INTERPOLATED')
        out_str = ' - '.join(out_str_split)
        out_dir = os.path.join(ITP_DATASET_DIR, out_str)
        os.mkdir(out_dir)

        for i in tqdm(range(0, len(frames) - 1), desc='Current', leave=False):
            img1 = frames[i]
            img2 = frames[i + 1]
            imgs_out = []
            for j in range(0, n):
                imgs_out.append(os.path.join(out_dir, img1.split(
                    '/')[-1].replace('.png', f'_{str(j).zfill(2)}.png')))

            interpolator.interpolate(img1, img2, imgs_out)

        shutil.copyfile(frames[-1], os.path.join(out_dir, frames[-1].split('/')
                        [-1].replace('.png', f'_{str(0).zfill(2)}.png')))

        torch.cuda.empty_cache()

        id_ += 1

    os.chdir(WORKING_DIR)


def create_d_i_dataset(id_start):
    if not os.path.exists(DEB_DATASET_DIR):
        print('Deblurred dataset not found.')
        create_deb_dataset()

    if os.path.exists(D_I_DATASET_DIR):
        shutil.rmtree(D_I_DATASET_DIR)

    os.mkdir(D_I_DATASET_DIR)

    n = 5
    interpolator = Interpolator(n=n)

    id_ = id_start
    print('Creating deblurred -> interpolated dataset...')
    runs = sorted(glob.glob(os.path.join(DEB_DATASET_DIR, '*')))
    for run in tqdm(runs, desc='Overall'):
        frames = sorted(glob.glob(os.path.join(run, '*')))
        n_frames = len(frames)

        old_run_duration_sec = n_frames / 25.0
        n_frames_interpolated = (n_frames - 1) * n + 1
        new_fps = int(math.ceil(n_frames_interpolated / old_run_duration_sec))

        out_str = run.split('/')[-1]
        out_str_split = out_str.split(' - ')
        out_str_split[0] = f'{str(id_).zfill(3)} ID'
        out_str_split[1] = f'{str(n_frames_interpolated).zfill(4)} IMG'
        out_str_split[4] = f'{str(new_fps).zfill(3)} FPS'
        out_str_split[5] = 'DEBLURRED-INTERPOLATED'
        out_str = ' - '.join(out_str_split)
        out_dir = os.path.join(D_I_DATASET_DIR, out_str)
        os.mkdir(out_dir)

        for i in tqdm(range(0, len(frames) - 1), desc='Current', leave=False):
            img1 = frames[i]
            img2 = frames[i + 1]
            imgs_out = []
            for j in range(0, n):
                imgs_out.append(os.path.join(out_dir, img1.split(
                    '/')[-1].replace('.png', f'_{str(j).zfill(2)}.png')))

            interpolator.interpolate(img1, img2, imgs_out)

        shutil.copyfile(frames[-1], os.path.join(out_dir, frames[-1].split('/')
                        [-1].replace('.png', f'_{str(0).zfill(2)}.png')))

        torch.cuda.empty_cache()

        id_ += 1

    os.chdir(WORKING_DIR)


def create_i_d_dataset(id_start):
    if not os.path.exists(ITP_DATASET_DIR):
        print('Interpolated dataset not found.')
        create_itp_dataset()

    if os.path.exists(I_D_DATASET_DIR):
        shutil.rmtree(I_D_DATASET_DIR)

    os.mkdir(I_D_DATASET_DIR)

    device = 'cuda:0'
    checkpoint = os.path.join(SHIFT_NET_DIR, 'ckpt', 'net_gopro_deblur.pth')

    net = GShiftNet(future_frames=2, past_frames=2)
    net.load_state_dict(torch.load(checkpoint)['params'])
    net.half()
    net = net.to(device)
    net.eval()

    id_ = id_start
    print('Creating interpolated -> deblurred dataset...')
    runs = sorted(glob.glob(os.path.join(ITP_DATASET_DIR, '*')))
    for run in tqdm(runs, desc='Overall'):
        out_str = run.split('/')[-1]
        out_str_split = out_str.split(' - ')
        out_str_split[0] = f'{str(id_).zfill(3)} ID'
        out_str_split[5] = 'INTERPOLATED-DEBLURRED'
        out_str = ' - '.join(out_str_split)
        out_dir = os.path.join(I_D_DATASET_DIR, out_str)
        os.mkdir(out_dir)

        frames = sorted(glob.glob(os.path.join(run, '*')))
        frames.insert(0, frames[0])
        frames.insert(0, frames[0])
        frames.append(frames[-1])
        frames.append(frames[-1])

        inputs = []
        for i in range(2, len(frames) - 2):
            input_ = []
            input_.append(frames[i - 2])
            input_.append(frames[i - 1])
            input_.append(frames[i])
            input_.append(frames[i + 1])
            input_.append(frames[i + 2])
            inputs.append(input_)

        with torch.no_grad():
            for i, input_ in enumerate(tqdm(inputs, desc='Current', leave=False)):
                images = [imageio.v2.imread(input_[i]) for i in range(0, len(input_))]
                images = numpy2tensor(images).to(device)
                outputs = net(images.half())
                outputs = outputs.float()
                outputs = tensor2numpy(outputs)
                cv.imwrite(os.path.join(out_dir, input_[2].split('/')[-1]), outputs[..., ::-1])

        torch.cuda.empty_cache()

        id_ += 1


def create_det_dataset():
    if not os.path.exists(STD_DATASET_DIR):
        print('Standard dataset not found.')
        create_std_dataset()

    if os.path.exists(DET_DATASET_DIR):
        shutil.rmtree(DET_DATASET_DIR)

    os.mkdir(DET_DATASET_DIR)
    os.mkdir(DET_TRAIN_DIR)
    os.mkdir(DET_VAL_DIR)
    os.mkdir(DET_TEST_DIR)

    n_train = 16
    n_val = 4
    n_test = 4

    forbidden = ['31_115.png', '63_204.png', '30_091.png', '4_144.png', '33_094.png', '13_126.png', '1_072.png',
                 '43_116.png', '25_117.png', '30_045.png', '16_146.png', '57_133.png', '57_058.png', '50_111.png',
                 '23_083.png', '1_026.png', '64_123.png', '39_080.png', '42_161.png', '33_167.png', '27_204.png',
                 '60_157.png', '54_175.png', '21_070.png', '23_104.png', '59_160.png', '33_105.png', '41_235.png',
                 '10_050.png', '52_131.png', '12_227.png', '30_135.png', '21_025.png', '40_118.png', '14_048.png',
                 '29_022.png', '10_111.png', '51_178.png', '56_039.png', '23_218.png', '52_101.png', '37_050.png',
                 '60_030.png', '13_019.png']

    print('Creating detection dataset...')
    runs = sorted(glob.glob(os.path.join(STD_RUNS_IMAGE_DIR, '*')))
    for i, run in enumerate(runs):
        images = sorted(glob.glob(os.path.join(run, '*')))

        train_files = []
        val_files = []
        test_files = []

        train_chunks = np.array_split(images, n_train)
        val_chunks = np.array_split(images, n_val)
        test_chunks = np.array_split(images, n_test)

        n_max = max(n_train, n_val, n_test)
        for j in range(0, n_max):

            if j < len(train_chunks):
                train_chunk = train_chunks[j]
                random_index = random_1.randint(0, len(train_chunk) - 1)
                random_image = train_chunk[random_index]
                image_filename = f"{i + 1}_{random_image.split('/')[-1]}"
                # fix to allow reuse of already labeled data while maintaining randomness
                # when creating a new dataset using a seed != 0, use the following block instead of this one
                if image_filename in forbidden:
                    while True:
                        random_index = random_2.randint(0, len(train_chunk) - 1)
                        random_image = train_chunk[random_index]
                        image_filename = f"{i + 1}_{random_image.split('/')[-1]}"
                        if image_filename not in forbidden \
                                and image_filename not in val_files \
                                and image_filename not in test_files:
                            break
                out_image = os.path.join(DET_TRAIN_DIR, image_filename)
                shutil.copyfile(random_image, out_image)
                train_files.append(image_filename)

            """
            if j < len(train_chunks):
                train_chunk = train_chunks[j]
                while True:
                    random_index = random_1.randint(0, len(train_chunk) - 1)
                    random_image = train_chunk[random_index]
                    image_filename = f"{i + 1}_{random_image.split('/')[-1]}"
                    out_image = os.path.join(DET_TRAIN_DIR, image_filename)
                    if (image_filename not in val_files and image_filename not in test_files):
                        shutil.copyfile(random_image, out_image)
                        train_files.append(image_filename)
                        break
            """

            if j < len(val_chunks):
                val_chunk = val_chunks[j]
                while True:
                    random_index = random_1.randint(0, len(val_chunk) - 1)
                    random_image = val_chunk[random_index]
                    image_filename = f"{i + 1}_{random_image.split('/')[-1]}"
                    out_image = os.path.join(DET_VAL_DIR, image_filename)
                    if (image_filename not in train_files and image_filename not in test_files):
                        shutil.copyfile(random_image, out_image)
                        val_files.append(image_filename)
                        break

            if j < len(test_chunks):
                test_chunk = test_chunks[j]
                while True:
                    random_index = random_1.randint(0, len(test_chunk) - 1)
                    random_image = test_chunk[random_index]
                    image_filename = f"{i + 1}_{random_image.split('/')[-1]}"
                    out_image = os.path.join(DET_TEST_DIR, image_filename)
                    if (image_filename not in train_files and image_filename not in val_files):
                        shutil.copyfile(random_image, out_image)
                        test_files.append(image_filename)
                        break


def numpy2tensor(input_seq, rgb_range=1.):
    # taken from https://github.com/dasongli1/Shift-Net/blob/main/inference/test_deblur.py
    tensor_list = []
    for img in input_seq:
        img = np.array(img).astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
        tensor_list.append(tensor)
    stacked = torch.stack(tensor_list).unsqueeze(0)
    return stacked


def tensor2numpy(tensor, rgb_range=1.):
    # taken from https://github.com/dasongli1/Shift-Net/blob/main/inference/test_deblur.py
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


if __name__ == '__main__':

    random_1 = random.Random(0)
    random_2 = random.Random(0)

    WORKING_DIR = os.path.dirname(__file__)
    EXTERNAL_NETS_DIR = os.path.join(WORKING_DIR, 'ext_nets')
    SHIFT_NET_DIR = os.path.join(EXTERNAL_NETS_DIR, 'shift_net')
    EMA_VFI_DIR = os.path.join(EXTERNAL_NETS_DIR, 'ema_vfi')
    VIDEOS_DIR = os.path.join(WORKING_DIR, 'videos')
    STD_DATASET_DIR = os.path.join(WORKING_DIR, 'std_dataset')
    STD_RUNS_VIDEO_DIR = os.path.join(STD_DATASET_DIR, 'runs_video')
    STD_RUNS_IMAGE_DIR = os.path.join(STD_DATASET_DIR, 'runs_image')
    DET_DATASET_DIR = os.path.join(WORKING_DIR, 'det_dataset')
    DET_TRAIN_DIR = os.path.join(DET_DATASET_DIR, 'train')
    DET_VAL_DIR = os.path.join(DET_DATASET_DIR, 'val')
    DET_TEST_DIR = os.path.join(DET_DATASET_DIR, 'test')
    DEB_DATASET_DIR = os.path.join(WORKING_DIR, 'deb_dataset')
    ITP_DATASET_DIR = os.path.join(WORKING_DIR, 'itp_dataset')
    D_I_DATASET_DIR = os.path.join(WORKING_DIR, 'd_i_dataset')
    I_D_DATASET_DIR = os.path.join(WORKING_DIR, 'i_d_dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()

    match args.dataset:
        case 'std':
            create_std_dataset(id_start=1)
        case 'deb':
            create_deb_dataset(id_start=65)
        case 'itp':
            create_itp_dataset(id_start=129)
        case 'd_i':
            create_d_i_dataset(id_start=193)
        case 'i_d':
            create_i_d_dataset(id_start=257)
        case 'det':
            create_det_dataset()

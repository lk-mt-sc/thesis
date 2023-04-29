""" 
Script to create dataset as specified in https://github.com/lk-mt-sc/thesis/wiki/Dataset. 
Prerequisites: 
a) videos in FullHD and .mp4 file format stored in ./videos named after their YouTube-ID
b) FFmpeg installed and accessible from the command line
"""
import os
import shutil
import subprocess

from tqdm import tqdm
import cv2 as cv


def time_to_ms(time):
    h = int(time[0:2])
    m = int(time[3:5])
    s = int(time[6:8])
    ms = int(time[9:])

    h_ms = h * 60 * 60 * 1000
    m_ms = m * 60 * 1000
    s_ms = s * 1000
    return h_ms + m_ms + s_ms + ms


if __name__ == '__main__':

    WORKING_DIR = os.path.dirname(__file__)
    VIDEOS_DIR = os.path.join(WORKING_DIR, 'videos')
    RUNS_VIDEO_DIR = os.path.join(WORKING_DIR, 'runs_video')
    RUNS_IMAGE_DIR = os.path.join(WORKING_DIR, 'runs_image')

    if os.path.exists(RUNS_VIDEO_DIR):
        shutil.rmtree(RUNS_VIDEO_DIR)

    if os.path.exists(RUNS_IMAGE_DIR):
        shutil.rmtree(RUNS_IMAGE_DIR)

    os.mkdir(RUNS_VIDEO_DIR)
    os.mkdir(RUNS_IMAGE_DIR)

    markdown_table_entries = []

    print('Creating dataset...')
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
            out_str.append(str(climber_id).zfill(2))
            out_str.append(f'{str(frames_cut)} IMG')
            if spotlight == 'yes':
                out_str.append('SPOTLIGHT')
            elif spotlight == 'no':
                out_str.append('NO SPOTLIGHT')
            if start_at_rest == 'yes':
                out_str.append('START AT REST')
            elif start_at_rest == 'no':
                out_str.append('NO START AT REST')
            out_str.append(f'{video_fps} FPS')
            out_str = ' - '.join(out_str)
            out_path = os.path.join(RUNS_VIDEO_DIR, out_str + '.mp4')

            args = ['ffmpeg', '-ss', f'{start_time_ms}ms', '-to', f'{end_time_ms}ms', '-i', video_path, out_path]
            with open(os.devnull, 'wb') as devnull:
                subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            out_folder = os.path.join(RUNS_IMAGE_DIR, out_str)
            os.mkdir(out_folder)
            run_video = cv.VideoCapture(out_path)

            img_id = 0
            while True:
                ret, frame = run_video.read()
                if not ret:
                    break
                if img_id < start_frame:
                    img_id += 1
                    continue
                if img_id > end_frame:
                    break
                out_path = os.path.join(out_folder, f'{str(img_id).zfill(3)}.png')
                width = frame.shape[1]
                if left_or_right_climber == 'L':
                    frame = frame[:, offset:int(offset+width/2)]
                if left_or_right_climber == 'R':
                    frame = frame[:, int(width/2-offset):int(width-offset)]
                cv.imwrite(out_path, frame)
                img_id += 1

    for entry in markdown_table_entries:
        print(entry)

import os
import re
import time
import glob
import logging
import argparse
import subprocess
from collections import deque
from datetime import datetime
from datetime import timedelta

if __name__ == '__main__':
    WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    CONFIGS_DIR = os.path.join(WORKING_DIR, 'configs')
    TRAININGS_DIR = os.path.join(WORKING_DIR, 'trainings')
    MMDETECTION_DIR = os.environ['MMDETECTION_DIR']
    MMDETECTION_TRAIN_SCRIPT = os.path.join(MMDETECTION_DIR, 'tools', 'train.py')
    MMDETECTION_TEST_SCRIPT = os.path.join(MMDETECTION_DIR, 'tools', 'test.py')

    configs_all = [
        os.path.join(CONFIGS_DIR, 'faster-rcnn.py'),
        os.path.join(CONFIGS_DIR, 'rtmdet.py'),
        os.path.join(CONFIGS_DIR, 'tood.py'),
        os.path.join(CONFIGS_DIR, 'vfnet.py'),
        os.path.join(CONFIGS_DIR, 'yolox.py'),
    ]

    default_training_name = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-name', type=str, default=default_training_name, required=False)
    parser.add_argument('--configs', nargs='+', type=str, default='all', required=False)
    parser.add_argument('--patience', type=int, default=10, required=False)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    training_name = args.training_name
    patience = args.patience
    resume = args.resume

    training_dir = os.path.join(TRAININGS_DIR, training_name)
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    else:
        assert resume, f'A training session with name {training_name} already exists, '\
            'it can be resumed using --resume argument. If it should not be resumed, '\
            'please use another training session name.'

    configs = []
    if args.configs != 'all':
        for config in args.configs:
            config_path = os.path.join(WORKING_DIR, config)
            configs.append(config_path)
    else:
        configs = configs_all.copy()

    for config in configs:
        assert os.path.exists(config), f'{config} configuration not found.'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(training_dir, training_name + '.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f'Start training session: {training_name}.')

    n_configs = len(configs)
    for i, config in enumerate(configs):
        base_name = config.split('/')[-1].split('.')[0]
        train_work_dir = os.path.join(training_dir, base_name, 'train')
        test_work_dir = os.path.join(training_dir, base_name, 'test')

        training_args = [
            'python',
            '-W',
            'ignore',
            MMDETECTION_TRAIN_SCRIPT,
            config,
            '--work-dir',
            train_work_dir,
        ]

        if resume:
            training_args.append('--resume')
            logger.info(f'Resume training of configuration: {config} ({i+1}/{n_configs}).')
        else:
            logger.info(f'Start training of configuration: {config} ({i+1}/{n_configs}).')

        start = time.time()
        training = subprocess.Popen(
            training_args,
            cwd=MMDETECTION_DIR,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )

        val_results = deque(maxlen=patience + 1)
        best_result = 0
        while True:
            line = training.stdout.readline()
            if not line:
                print()
                break
            line = line.rstrip()
            if 'Epoch(train)' in line:
                log_str = line[line.find('Epoch(train)'):]
                logger.info(log_str)
            if 'coco/bbox_mAP:' in line:
                log_str = line[line.find('Epoch(val)'):]
                logger.info(log_str)
                result = float(re.findall("[+-]?\d+\.\d+", line)[0])
                if result > best_result:
                    best_result = result
                val_results.append(result)
                if len(val_results) == patience + 1:
                    saturated = all(i <= best_result for i in val_results) and val_results[0] == best_result
                else:
                    saturated = False
                if saturated:
                    end = time.time()
                    duration = end - start
                    log_str = f'Stop training due to saturated validation set bbox_mAP (patience={patience}). '
                    log_str += f'Train duration: {timedelta(seconds=duration)}'
                    logger.info(log_str)
                    training.terminate()

        best_checkpoint = glob.glob(os.path.join(train_work_dir, 'best*.pth'))[0]
        best_epoch = int(best_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info(f'Evaluate test set on checkpoint {best_checkpoint} of best epoch {best_epoch}.')

        testing_args = [
            'python',
            '-W',
            'ignore',
            MMDETECTION_TEST_SCRIPT,
            config,
            best_checkpoint,
            '--work-dir',
            test_work_dir,
        ]

        testing = subprocess.Popen(
            testing_args,
            cwd=MMDETECTION_DIR,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )

        while True:
            line = testing.stdout.readline()
            if not line:
                print()
                break
            line = line.rstrip()
            if 'coco/bbox_mAP:' in line:
                log_str = line[line.find('Epoch(test)'):]
                logger.info(log_str)

        if i != n_configs - 1:
            logger.info('.')

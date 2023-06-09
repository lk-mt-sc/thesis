import os
import imp
import glob
import json
import time
import shutil
import subprocess
from statistics import mean
from datetime import datetime

import torch
from torchvision.ops import box_iou

from data_types.run import Run
from data_types.feature import Feature
from utils import collect_image_infos, cvt_to_coco_json
from manager.dataset_manager import KeypointsInterpolation
from manager.metric_manager import InferenceMetrics, RunMetrics
from common import MMPOSE_DIR, MMPOSE_TEST_SCRIPT, MMPOSE_DATASET_DIR
from common import MMPOSE029_DIR, MMPOSE029_VENV_DIR, MMPOSE029_INFERENCE_SCRIPT
from common import MMDETECTION_DIR, MMDETECTION_TEST_SCRIPT
from common import INFERENCES_DIR


class Inference:
    def __init__(self, metadata):
        self.id = metadata['id']
        self.name = metadata['name']
        self.start_datetime_timestamp = metadata['start_datetime'] if 'start_datetime' in metadata else None
        if self.start_datetime_timestamp is None:
            self.start_datetime = None
        else:
            self.start_datetime = datetime.fromtimestamp(self.start_datetime_timestamp).strftime('%d.%m.%Y %H:%M:%S')
        self.end_datetime_timestamp = metadata['end_datetime'] if 'end_datetime' in metadata else None
        if self.end_datetime_timestamp is None:
            self.end_datetime = None
        else:
            self.end_datetime = datetime.fromtimestamp(self.end_datetime_timestamp).strftime('%d.%m.%Y %H:%M:%S')
        self.mmpose_model = metadata['mmpose_model']
        self.mmdetection_model = metadata['mmdetection_model']
        self.data = metadata['data']
        self.detection_duration = metadata['detection_duration'] if 'detection_duration' in metadata else None
        self.pose_estimation_duration = metadata['pose_estimation_duration'] if 'pose_estimation_duration' in metadata else None
        self.score_detection = metadata['score_detection'] if 'score_detection' in metadata else None
        self.score_pose_estimation = metadata['score_pose_estimation'] if 'score_pose_estimation' in metadata else None
        self.description = metadata['description']
        self.path = metadata['path'] if 'path' in metadata else None
        self.runs = []

        self.load_runs()

    def __str__(self):
        print_str = f'{self.name}'
        if self.start_datetime is not None:
            print_str += f' | {self.start_datetime} | {self.id}'
        else:
            print_str += f' | {self.id}'
        return print_str

    def infer(self, inference_progress, existing_dataset, dataset_type):
        self.start_datetime_timestamp = datetime.timestamp(datetime.now())

        out_dir = os.path.join(INFERENCES_DIR, self.id)
        os.mkdir(out_dir)
        self.path = out_dir

        mmdetection_result_dir = os.path.join(out_dir, 'mmdetection_result_dir')
        os.mkdir(mmdetection_result_dir)

        mmdetection_work_dir = os.path.join(out_dir, 'mmdetection_work_dir')
        os.mkdir(mmdetection_work_dir)

        mmdetection_config = self.mmdetection_model.config
        mmdetection_checkpoint = self.mmdetection_model.checkpoint

        mmpose_config = self.mmpose_model.config
        mmpose_checkpoint = self.mmpose_model.checkpoint
        multi_frame_mmpose029 = self.mmpose_model.multi_frame_mmpose029

        if multi_frame_mmpose029:
            data_mode = 'topdown'
        else:
            config_name = mmpose_config.split('/')[-1].split('.')[0]
            config = imp.load_source(config_name, mmpose_config)

            data_mode = config.data_mode
            assert data_mode == 'topdown' or data_mode == 'bottomup'

        if existing_dataset is None:
            dataset_dir = MMPOSE_DATASET_DIR + f'_{self.id}'
            os.mkdir(dataset_dir)
            for i, data in enumerate(self.data):
                inference_progress.value = f'DATA PREP. {i}/{len(self.data)}'
                images = data.get_images()
                for image in images:
                    src = image
                    dst = os.path.join(dataset_dir, f"{data.id}_{image.split('/')[-1]}")
                    shutil.copyfile(src, dst)

            inference_progress.value = 'ANN. FILE CREATION'
            image_infos = collect_image_infos(dataset_dir)
            image_list_coco_format = cvt_to_coco_json(image_infos)
            ann_file = os.path.join(dataset_dir, 'ann_file.json')

            with open(ann_file, 'w', encoding='utf8') as file:
                json.dump(image_list_coco_format, file)
        else:
            dataset_dir = existing_dataset
            ann_file = os.path.join(dataset_dir, 'ann_file.json')

        n_runs = len(self.data)
        n_images = len(glob.glob(os.path.join(dataset_dir, '*.png')))

        persistent_detection_found = False
        if existing_dataset is not None:
            dataset_properties = existing_dataset.split('/')[-1]
            dataset_properties = dataset_properties.replace('dataset_', '')
            config_name = mmdetection_config.split('/')[-1].split('.')[0]
            persistent_result_dir = os.path.join(INFERENCES_DIR, 'persistent_detections',
                                                 dataset_properties + '_' + config_name, 'mmdetection_result_dir')
            persistent_results_pickle_file_path = os.path.join(persistent_result_dir, 'persistent.results.pkl')
            persistent_results_json_file_path = os.path.join(persistent_result_dir, 'persistent.bbox.json')

            if os.path.exists(persistent_results_pickle_file_path) and \
                    os.path.exists(persistent_results_json_file_path):
                persistent_detection_found = True
                results_pickle_file_path = os.path.join(mmdetection_result_dir, str(self.id) + '.results.pkl')
                results_json_file_path = os.path.join(mmdetection_result_dir, str(self.id) + '.bbox.json')
                shutil.copyfile(persistent_results_pickle_file_path, results_pickle_file_path)
                shutil.copyfile(persistent_results_json_file_path, results_json_file_path)
                ann_file = os.path.join(existing_dataset, 'ann_file.json')
                self.detection_duration = (0, 0, 0)

        if not persistent_detection_found:
            mmdetection_outfile_prefix = os.path.join(mmdetection_result_dir, str(self.id))
            mmdetection_result_dump_file = os.path.join(mmdetection_result_dir,  str(self.id) + '.results.pkl')

            inference_progress.value = 'BB. DETECTION STARTUP'

            mmdetection_args = [
                'python',
                MMDETECTION_TEST_SCRIPT,
                mmdetection_config,
                mmdetection_checkpoint,
                '--work-dir',
                mmdetection_work_dir,
                '--out',
                mmdetection_result_dump_file,
                '--cfg-options',
                f'test_dataloader.dataset.ann_file={ann_file}',
                f'test_evaluator.ann_file={ann_file}',
                f'test_evaluator.outfile_prefix={mmdetection_outfile_prefix}'
            ]

            start = time.time()

            detection = subprocess.Popen(
                mmdetection_args,
                cwd=MMDETECTION_DIR,
                stdout=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )

            while True:
                line = detection.stdout.readline()
                if not line:
                    print()
                    break
                line = line.rstrip()
                if 'mmengine - INFO - Epoch(test)' in line:
                    tracker = line[line.find('[') + 1: line.find(']')].split('/')
                    progress_percentage = int(int(tracker[0])/int(tracker[1]) * 100)
                    inference_progress.value = 'BB. DETECTION ' + f'{progress_percentage}%'

            end = time.time()
            duration = end - start
            self.detection_duration = (duration / 60, duration / n_runs, duration / n_images)
            results_pickle_file_path = mmdetection_result_dump_file
            results_json_file_path = mmdetection_outfile_prefix + '.bbox.json'

            inference_progress.value = 'REMOVING LOW SCORES'

            with open(results_json_file_path, 'r', encoding='utf8') as results_file:
                results = json.load(results_file)

                image_ids = []
                for result in results:
                    image_ids.append(result['image_id'])

                ids = set()
                duplicate_ids = []
                for id_ in image_ids:
                    if id_ in ids:
                        duplicate_ids.append(id_)
                    else:
                        ids.add(id_)

                for id_ in duplicate_ids:
                    duplicates = []
                    for result in results:
                        if result['image_id'] == id_:
                            duplicates.append(result)
                    duplicates = sorted(duplicates, key=lambda d: d['score'], reverse=True)
                    for duplicate in duplicates[1:]:
                        results.remove(duplicate)

            with open(results_json_file_path, 'w', encoding='utf8') as results_file:
                json.dump(results, results_file)

            inference_progress.value = 'CALC. AVG. CONFIDENCE'

            with open(results_json_file_path, 'r', encoding='utf8') as results_file:
                results = json.load(results_file)
                n_results = len(results)
                assert n_results == n_images, f'Missing detection for {n_images - n_results} images.'

        mmpose_result_dir = os.path.join(out_dir, 'mmpose_result_dir')
        os.mkdir(mmpose_result_dir)

        mmpose_work_dir = os.path.join(out_dir, 'mmpose_work_dir')
        os.mkdir(mmpose_work_dir)

        mmpose_outfile_prefix = os.path.join(mmpose_result_dir, str(self.id))

        inference_progress.value = 'POSE EST. STARTUP'

        if multi_frame_mmpose029:
            mmpose_args = [
                os.path.join(MMPOSE029_VENV_DIR, 'bin', 'python'),
                MMPOSE029_INFERENCE_SCRIPT,
                '--config',
                mmpose_config,
                '--checkpoint',
                mmpose_checkpoint,
                '--dataset-dir',
                dataset_dir,
                '--bbox-file',
                results_json_file_path,
                '--ann-file',
                ann_file,
                '--pose-estimations-file',
                mmpose_outfile_prefix + '.keypoints.json'
            ]

            bbox_file_path = os.path.join(mmpose_work_dir, 'bbox_file.json')
            shutil.copyfile(results_json_file_path, bbox_file_path)
            shutil.copyfile(mmpose_config, os.path.join(mmpose_work_dir, os.path.basename(mmpose_config)))

            start = time.time()

            pose_estimation = subprocess.Popen(
                mmpose_args,
                cwd=MMPOSE029_DIR,
                stdout=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )

            inference_progress.value = 'POSE EST. v029'
            while True:
                line = pose_estimation.stdout.readline()
                if not line:
                    print()
                    break

            end = time.time()
            duration = end - start
            self.pose_estimation_duration = (duration / 60, duration / n_runs, duration / n_images)
            results_json_file_path = mmpose_outfile_prefix + '.keypoints.json'
        else:
            mmpose_args = [
                'python',
                MMPOSE_TEST_SCRIPT,
                mmpose_config,
                mmpose_checkpoint,
                '--work-dir',
                mmpose_work_dir,
                '--cfg-options',
                f'test_dataloader.dataset.ann_file={ann_file}',
                f'test_evaluator.ann_file={ann_file}',
                f'test_evaluator.outfile_prefix={mmpose_outfile_prefix}'
            ]

            bbox_file_path = os.path.join(mmpose_work_dir, 'bbox_file.json')
            shutil.copyfile(results_json_file_path, bbox_file_path)

            if data_mode == 'topdown':
                mmpose_args.append(f'test_dataloader.dataset.bbox_file={bbox_file_path}')

            start = time.time()

            pose_estimation = subprocess.Popen(
                mmpose_args,
                cwd=MMPOSE_DIR,
                stdout=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )

            while True:
                line = pose_estimation.stdout.readline()
                if not line:
                    print()
                    break
                line = line.rstrip()
                if 'mmengine - INFO - Epoch(test)' in line:
                    tracker = line[line.find('[') + 1: line.find(']')].split('/')
                    progress_percentage = int(int(tracker[0])/int(tracker[1]) * 100)
                    inference_progress.value = 'POSE EST. ' + f'{progress_percentage}%'

            end = time.time()
            duration = end - start
            self.pose_estimation_duration = (duration / 60, duration / n_runs, duration / n_images)
            results_json_file_path = mmpose_outfile_prefix + '.keypoints.json'

        with open(ann_file, 'r') as annotations_file:
            annotations = json.load(annotations_file)

        with open(bbox_file_path, 'r') as bbox_file:
            pred_bboxes = json.load(bbox_file)

        with open(results_json_file_path, 'r') as results_file:
            results = json.load(results_file)

        n_data = len(self.data)
        runs = []
        for i, data in enumerate(self.data):

            progress_percentage = int(((i + 1)/n_data) * 100)
            inference_progress.value = 'PREP. INFER RES. ' + f'{progress_percentage}%'

            images = data.get_images()
            bboxes = []
            bboxes_bottomup = []
            ious = []
            detection_scores = []
            pose_estimation_scores = []
            features = []
            dataset_keypoints = dataset_type.keypoints
            n_keypoints = len(dataset_keypoints)
            for keypoint in dataset_keypoints:
                features.append(Feature(name=keypoint, fps=data.fps))
            for step, image in enumerate(images):
                image_filename = str(data.id) + '_' + image.split('/')[-1]
                for dataset_image in annotations['images']:
                    if dataset_image['file_name'].split('/')[-1] == image_filename:
                        image_id = dataset_image['id']

                pred_bbox = None
                for bbox in pred_bboxes:
                    if bbox['image_id'] == image_id:
                        pred_bbox = bbox

                if data_mode == 'topdown':
                    result = next(result for result in results if result['image_id'] == image_id)
                    keypoints = result['keypoints']
                    x_coord = keypoints[0:: 3]
                    y_coord = keypoints[1:: 3]
                    keypoint_scores = keypoints[2::3]

                    keypoint_scores_for_image_mean = keypoints[17::3]
                    score = mean(keypoint_scores_for_image_mean)
                    result['score'] = score

                    for i, (x, y, s) in enumerate(zip(x_coord, y_coord, keypoint_scores)):
                        features[i * 2].add(step, x, s)
                        features[i * 2 + 1].add(step, y, s)

                if data_mode == 'bottomup':
                    x = int(pred_bbox['bbox'][0])
                    y = int(pred_bbox['bbox'][1])
                    w = int(pred_bbox['bbox'][2])
                    h = int(pred_bbox['bbox'][3])
                    pred_bbox_xyxy = [x, y, x + w, y + h]
                    pred_bbox_tensor = torch.FloatTensor(pred_bbox_xyxy)
                    pred_bbox_tensor = pred_bbox_tensor.unsqueeze(0)

                    poses = []
                    for result in results:
                        if result['image_id'] == image_id:
                            poses.append(result)

                    preds = []
                    for pose in poses:
                        keypoints = pose['keypoints']
                        x_coord = keypoints[0::3]
                        y_coord = keypoints[1::3]
                        keypoint_scores = keypoints[2::3]

                        keypoint_scores_for_image_mean = keypoints[17::3]
                        score = mean(keypoint_scores_for_image_mean)

                        bbox = [int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))]
                        bbox_tensor = torch.FloatTensor(bbox)
                        bbox_tensor = bbox_tensor.unsqueeze(0)

                        iou = box_iou(pred_bbox_tensor, bbox_tensor)
                        if iou > 0.3:
                            preds.append({
                                'iou': iou.item(),
                                'bbox': bbox,
                                'x_coord': x_coord,
                                'y_coord': y_coord,
                                'keypoint_scores': keypoint_scores,
                                'score': score
                            })

                    preds = sorted(preds, key=lambda d: d['score'], reverse=True)

                    if not preds:
                        preds.append({
                            'iou': -1,
                            'bbox': -1,
                            'x_coord': [-1 for i in range(int((n_keypoints - 4) / 2))],
                            'y_coord': [-1 for i in range(int((n_keypoints - 4) / 2))],
                            'keypoint_scores': [-1 for i in range(int((n_keypoints - 4) / 2))],
                            'score': -1
                        })

                    result = preds[0]
                    x_coord = result['x_coord']
                    y_coord = result['y_coord']
                    keypoint_scores = result['keypoint_scores']
                    for i, (x, y, s) in enumerate(zip(x_coord, y_coord, keypoint_scores)):
                        features[i * 2].add(step, x, s)
                        features[i * 2 + 1].add(step, y, s)

                bboxes.append(pred_bbox['bbox'])
                detection_scores.append(pred_bbox['score'])
                pose_estimation_scores.append(result['score'])

                if data_mode == 'bottomup':
                    bboxes_bottomup.append(result['bbox'])
                    ious.append(result['iou'])

            detection_scores_for_inference_mean = [score for score in detection_scores if score != -1]
            self.score_detection = mean(detection_scores_for_inference_mean)

            pose_estimation_scores_for_inference_mean = [score for score in pose_estimation_scores if score != -1]
            self.score_pose_estimation = mean(pose_estimation_scores_for_inference_mean)

            self.interpolate_keypoint(features,
                                      KeypointsInterpolation.NECK,
                                      KeypointsInterpolation.LEFT_SHOULDER,
                                      KeypointsInterpolation.RIGHT_SHOULDER)
            self.interpolate_keypoint(features,
                                      KeypointsInterpolation.HEAD,
                                      KeypointsInterpolation.LEFT_EAR,
                                      KeypointsInterpolation.RIGHT_EAR)

            for feature in features:
                feature.interpolate_values()

            run_id = data.id
            run_path = os.path.join(self.path, f'run_{str(run_id).zfill(3)}.pkl')
            runs.append({
                'id': run_id,
                'path': run_path,
                'data': data,
                'features': features,
                'bboxes': bboxes,
                'bboxes_bottomup': bboxes_bottomup,
                'ious': ious,
                'detection_scores': detection_scores,
                'pose_estimation_scores': pose_estimation_scores
            })

        inference_progress.value = 'CALC. METRICS'

        inference_metrics = InferenceMetrics()
        for run in runs:
            for feature in run['features']:
                inference_metrics.add_feature(feature)
        inference_metrics.calculate()

        for run in runs:
            run['metrics'] = RunMetrics(run['features']).calculate().copy()

        inference_progress.value = 'SAVING INFER RES. '
        for run in runs:
            new_run = Run(
                run['id'],
                run['path'],
                run['data'],
                run['features'],
                run['bboxes'],
                run['bboxes_bottomup'],
                run['ious'],
                run['detection_scores'],
                run['pose_estimation_scores'],
                run['metrics'])
            new_run.save(run['path'])

        self.end_datetime_timestamp = datetime.timestamp(datetime.now())
        self.store_metadata(out_dir)
        self.load_runs()

        inference_progress.value = 'DONE'

    def interpolate_keypoint(self, features, target, source_1, source_2):
        s1_x_f = next(f for f in features if f.name == source_1.value + '_x')
        s1_y_f = next(f for f in features if f.name == source_1.value + '_y')
        s2_x_f = next(f for f in features if f.name == source_2.value + '_x')
        s2_y_f = next(f for f in features if f.name == source_2.value + '_y')
        t_x_f = next(f for f in features if f.name == target.value + '_x')
        t_y_f = next(f for f in features if f.name == target.value + '_y')
        for s, (s1_x, s1_y, s2_x, s2_y) in enumerate(zip(s1_x_f.values, s1_y_f.values, s2_x_f.values, s2_y_f.values)):
            if -1 in (s1_x, s1_y, s2_x, s2_y):
                t_x_f.add(s, -1, -1)
                t_y_f.add(s, -1, -1)
                continue
            x1 = min(s1_x, s2_x)
            y1 = min(s1_y, s2_y)
            x2 = max(s1_x, s2_x)
            y2 = max(s1_y, s2_y)
            t_x_f.add(s, x1 + (x2 - x1) / 2, -1)
            t_y_f.add(s, y1 + (y2 - y1) / 2, -1)

    def store_metadata(self, out_dir):
        metadata_file = open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf8')
        metadata = {
            'id': self.id,
            'name': self.name,
            'start_datetime': self.start_datetime_timestamp,
            'end_datetime': self.end_datetime_timestamp,
            'mmpose_model': str(self.mmpose_model),
            'mmpose_model_config': self.mmpose_model.config,
            'mmpose_model_checkpoint': self.mmpose_model.checkpoint,
            'mmdetection_model': str(self.mmdetection_model),
            'mmdetection_model_config': self.mmdetection_model.config,
            'mmdetection_model_checkpoint': self.mmdetection_model.checkpoint,
            'data': [int(d.id) for d in self.data],
            'detection_duration': self.detection_duration,
            'pose_estimation_duration': self.pose_estimation_duration,
            'score_detection': self.score_detection,
            'score_pose_estimation': self.score_pose_estimation,
            'description': self.description,
            'path': self.path
        }
        json.dump(metadata, metadata_file)

    def load_runs(self):
        runs = sorted(glob.glob(os.path.join(INFERENCES_DIR, self.id, '*.pkl')))
        if not runs:
            self.runs = []
        else:
            self.runs.clear()
            for run in runs:
                self.runs.append(Run.load(run))

    def get_run(self, id_):
        for run in self.runs:
            if run.id == id_:
                return run
        return None

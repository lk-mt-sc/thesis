"""
Script to convert output annotation files from label-studio into mmdetection suitable format.
"""

import os
import json

if __name__ == '__main__':

    WORKING_DIR = os.path.dirname(__file__)
    DET_DATASET_DIR = os.path.join(WORKING_DIR, 'det_dataset')
    DET_LABELED_DATASET_DIR = os.path.join(DET_DATASET_DIR, 'det')
    DET_LABELS_DIR = os.path.join(DET_DATASET_DIR, 'labels')
    DET_ANNOTATIONS_DIR = os.path.join(DET_LABELED_DATASET_DIR, 'annotations')

    # train
    train = []
    for i in range(0, 4):
        train_file_path_in = os.path.join(DET_LABELS_DIR, f'train_{i+1}_4.json')
        with open(train_file_path_in, 'r', encoding='utf8') as train_file:
            train.append(json.load(train_file))

    train_template = train[0]
    images = train_template['images']
    annotations = train_template['annotations']
    for i in range(0, 3):
        train_ = train[i + 1]
        for image in train_['images']:
            image['id'] = int(image['id'] + (i + 1) * 256)
            images.append(image)

        for annotation in train_['annotations']:
            annotation['id'] = int(annotation['id'] + (i + 1) * 256)
            annotation['image_id'] = int(annotation['image_id'] + (i + 1) * 256)
            annotations.append(annotation)

    for image in images:
        image['file_name'] = image['file_name'].split('/')[-1]

    train_template['categories'][0]['name'] = 'climber'

    train_file_path_out = os.path.join(DET_ANNOTATIONS_DIR, 'train.json')
    with open(train_file_path_out, 'w', encoding='utf8') as train_file:
        json.dump(train_template, train_file)

    # val
    val_file_path_in = os.path.join(DET_LABELS_DIR, 'val.json')
    with open(val_file_path_in, 'r', encoding='utf8') as val_file:
        val = json.load(val_file)

    for image in val['images']:
        image['file_name'] = image['file_name'].split('/')[-1]

    val['categories'][0]['name'] = 'climber'

    val_file_path_out = os.path.join(DET_ANNOTATIONS_DIR, 'val.json')
    with open(val_file_path_out, 'w', encoding='utf8') as val_file:
        json.dump(val, val_file)

    # test
    test_file_path_in = os.path.join(DET_LABELS_DIR, 'test.json')
    with open(test_file_path_in, 'r', encoding='utf8') as test_file:
        test = json.load(test_file)

    for image in test['images']:
        image['file_name'] = image['file_name'].split('/')[-1]

    test['categories'][0]['name'] = 'climber'

    test_file_path_out = os.path.join(DET_ANNOTATIONS_DIR, 'test.json')
    with open(test_file_path_out, 'w', encoding='utf8') as test_file:
        json.dump(test, test_file)

import os
import sys
import traceback
import string
import random

from mmengine.utils import scandir, track_iter_progress
from PIL import Image


class Suppressor(object):
    # modified from https://stackoverflow.com/a/40054132

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass
            # do normal exception handling

    def write(self, x):
        pass

    def flush(self):
        pass


def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    # modified from https://stackoverflow.com/a/2257449
    return ''.join(random.choice(chars) for _ in range(size))


def collect_image_infos(path, exclude_extensions=None):
    # taken from https://github.com/open-mmlab/mmdetection/blob/main/tools/dataset_converters/images2coco.py
    img_infos = []

    images_generator = scandir(path, recursive=True)
    for image_path in list(images_generator):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def cvt_to_coco_json(img_infos):
    # modified from https://github.com/open-mmlab/mmdetection/blob/main/tools/dataset_converters/images2coco.py
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    category_item = dict()
    category_item['supercategory'] = 'person'
    category_item['id'] = 1
    category_item['name'] = 'person'
    coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        image_id += 1
    return coco

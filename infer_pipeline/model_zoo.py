import os
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

from common import MMPOSE_CHECKPOINTS_DIR
from data_types.mmpose_model import MMPoseModel


class ModelZoo:
    def __init__(self, redownload_model_zoo):
        self.url = 'https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html'
        self.local = os.path.join(os.path.dirname(__file__), 'model_zoo.html')
        self.retrieve_html(redownload_model_zoo)
        self.datasets = ['coco', 'crowdpose', 'mpii', 'aic']

    def retrieve_html(self, redownload_model_zoo):
        if not os.path.exists(self.local) or redownload_model_zoo:
            try:
                request = requests.get(self.url, timeout=10)
                try:
                    open(self.local, 'wb').write(request.content)
                except (FileNotFoundError, PermissionError, OSError) as e:
                    print(
                        f'Model Zoo: Could not write mmpose model zoo \
                        html page to local file. Exception: {e}')
                    exit(-1)
            except requests.exceptions.ConnectTimeout:
                print('Model Zoo: Connection timeout in retrieving mmpose model zoo html page.')
                exit(-1)

    def get_models(self, dataset, redownload_model_zoo=False, get_checkpoint_links_only=False):
        if redownload_model_zoo:
            self.retrieve_html(redownload_model_zoo=True)

        if dataset == 'all':
            models = list()
            for dataset in self.datasets:
                models += self.get_models(dataset, redownload_model_zoo=False,
                                          get_checkpoint_links_only=get_checkpoint_links_only)
            return models

        if not dataset in self.datasets:
            print(
                f"Model Zoo: Dataset {dataset} not available. \
                Available datasets: {', '.join(self.datasets)}")
            exit(-1)

        mmpose_configs = BeautifulSoup(open(self.local, 'r', encoding='utf8'), 'html.parser')

        dataset_div = mmpose_configs.find(id=f'{dataset}-dataset')
        sections = dataset_div.find_all('div', class_='section')

        match dataset:
            case 'coco':
                return self._get_models_coco(sections, get_checkpoint_links_only)
            case 'crowdpose':
                return self._get_models_crowdpose(sections, get_checkpoint_links_only)
            case 'mpii':
                return self._get_models_mpii(sections, get_checkpoint_links_only)
            case 'aic':
                return self._get_models_aic(sections, get_checkpoint_links_only)

    def _get_models_coco(self, sections, get_checkpoints_links_only):
        models = list()
        for section in sections:
            tables = section.find_all(class_='docutils')
            if len(tables) > 2:
                tables = tables[:2]
            for table in tables:
                tr = table.find_all('tr')
                for config in tr[1:]:
                    td = config.find_all('td')
                    train_set_column_exists = len(td) == 10
                    config_link = td[0].find('a')
                    if train_set_column_exists:
                        arch = td[1].text
                        input_size = td[2].text
                    else:
                        arch = config_link.text
                        input_size = td[1].text
                    config = self.get_config_rtmpose_fix(config_link, 'coco')
                    checkpoint = td[-2].find('a')['href']

                    if get_checkpoints_links_only:
                        models.append(checkpoint)
                        continue
                    else:
                        checkpoint = checkpoint.split('/')[-1]

                    AP = 0.0
                    if train_set_column_exists:
                        AP = td[3].text
                    else:
                        AP = td[2].text
                    key_metric = f'AP ({AP})'
                    section_ = section.find('h3').text[:-1]
                    section_ = section_.replace('on Coco', '')
                    models.append(
                        MMPoseModel(section=section_,
                                    arch=arch,
                                    dataset='coco',
                                    input_size=input_size,
                                    key_metric=key_metric,
                                    checkpoint=checkpoint,
                                    config=config))

        return models

    def _get_models_crowdpose(self, sections, get_checkpoints_links_only):
        models = list()
        for section in sections:
            tables = section.find_all(class_='docutils')
            if len(tables) > 2:
                tables = tables[:2]
            for table in tables:
                tr = table.find_all('tr')
                for config in tr[1:]:
                    td = config.find_all('td')
                    config_link = td[0].find('a')
                    arch = config_link.text
                    input_size = td[1].text
                    config = self.get_config_rtmpose_fix(config_link, 'crowdpose')
                    checkpoint = td[-2].find('a')['href']

                    if get_checkpoints_links_only:
                        models.append(checkpoint)
                        continue
                    else:
                        checkpoint = checkpoint.split('/')[-1]

                    AP = td[2].text
                    key_metric = f'AP ({AP})'
                    section_ = section.find('h3').text[:-1]
                    section_ = section_.replace('on Crowdpose', '')
                    models.append(
                        MMPoseModel(section=section_,
                                    arch=arch,
                                    dataset='crowdpose',
                                    input_size=input_size,
                                    key_metric=key_metric,
                                    checkpoint=checkpoint,
                                    config=config))

        return models

    def _get_models_mpii(self, sections, get_checkpoints_links_only):
        models = list()
        for section in sections:
            tables = section.find_all(class_='docutils')
            if len(tables) > 2:
                tables = tables[:2]
            for table in tables:
                tr = table.find_all('tr')
                for config in tr[1:]:
                    td = config.find_all('td')
                    config_link = td[0].find('a')
                    arch = config_link.text
                    input_size = td[1].text
                    config = self.get_config_rtmpose_fix(config_link, 'mpii')
                    checkpoint = td[-2].find('a')['href']

                    if get_checkpoints_links_only:
                        models.append(checkpoint)
                        continue
                    else:
                        checkpoint = checkpoint.split('/')[-1]

                    mean = td[2].text
                    key_metric = f'Mean ({mean})'
                    section_ = section.find('h3').text[:-1]
                    section_ = section_.replace('on MPII', '')
                    models.append(
                        MMPoseModel(section=section_,
                                    arch=arch,
                                    dataset='mpii',
                                    input_size=input_size,
                                    key_metric=key_metric,
                                    checkpoint=checkpoint,
                                    config=config))

        return models

    def _get_models_aic(self, sections, get_checkpoints_links_only):
        models = list()
        for section in sections:
            tables = section.find_all(class_='docutils')
            if len(tables) > 2:
                tables = tables[:2]
            for table in tables:
                tr = table.find_all('tr')
                for config in tr[1:]:
                    td = config.find_all('td')
                    config_link = td[0].find('a')
                    arch = config_link.text
                    input_size = td[1].text
                    config = self.get_config_rtmpose_fix(config_link, 'aic')
                    checkpoint = td[-2].find('a')['href']

                    if get_checkpoints_links_only:
                        models.append(checkpoint)
                        continue
                    else:
                        checkpoint = checkpoint.split('/')[-1]

                    AP = td[2].text
                    key_metric = f'AP ({AP})'
                    section_ = section.find('h3').text[:-1]
                    section_ = section_.replace('on AIC', '')
                    models.append(
                        MMPoseModel(section=section_,
                                    arch=arch,
                                    dataset='aic',
                                    input_size=input_size,
                                    key_metric=key_metric,
                                    checkpoint=checkpoint,
                                    config=config))

        return models

    def get_config_rtmpose_fix(self, config_link, dataset):
        config_link = config_link['href']
        if 'rtmpose' in config_link:
            return '/configs/body_2d_keypoint/rtmpose/' + dataset + '/' + config_link.split('/')[-1]
        else:
            return '/'.join(config_link.split('/')[-5:])

    @classmethod
    def redownload_checkpoints(cls, overwrite=False):
        model_zoo = ModelZoo(redownload_model_zoo=False)
        checkpoint_links = model_zoo.get_models(dataset='all', get_checkpoint_links_only=True)
        print(f'Model Zoo: Downloading {len(checkpoint_links)} checkpoints.')
        for checkpoint_link in tqdm(checkpoint_links):
            checkpoint_filename = checkpoint_link.split('/')[-1]
            checkpoint_local = os.path.join(MMPOSE_CHECKPOINTS_DIR, checkpoint_filename)
            if os.path.exists(checkpoint_local):
                if overwrite:
                    os.remove(checkpoint_local)
                else:
                    continue
            try:
                request = requests.get(checkpoint_link, timeout=10)
                try:
                    open(checkpoint_local, 'wb').write(request.content)
                except (FileNotFoundError, PermissionError, OSError) as e:
                    print(
                        f'Model Zoo: Could not write checkpoint to local file. Exception: {e}')
                    exit(-1)
            except requests.exceptions.ConnectTimeout:
                print('Model Zoo: Connection timeout in retrieving current checkpoint. Skipping current checkpoint.')
                continue

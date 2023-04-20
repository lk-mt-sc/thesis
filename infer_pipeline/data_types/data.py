import os
import glob


class Data:
    def __init__(self, path):
        data = path.split('/')[-1]
        data = data.split('_')
        self.path = path
        self.id = int(data[1])
        self.fps = int(data[2])
        self.n_images = int(data[3])
        if len(data) > 4:
            self.keys = data[4:]
        else:
            self.keys = None

    def __str__(self):
        data_str = f'{str(self.id).zfill(3)} | {str(self.fps).zfill(2)} FPS | {str(self.n_images).zfill(3)} IMG'
        if self.keys is not None:
            for key in self.keys:
                data_str += ' | ' + key
        return data_str

    def get_images(self):
        return sorted(glob.glob(os.path.join(self.path, '*')))

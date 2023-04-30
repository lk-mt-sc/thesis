import os
import glob


class Data:
    def __init__(self, path):
        self.path = path
        self.folder_name = self.path.split('/')[-1]
        data_properties = self.folder_name.split(' - ')
        self.id = int(data_properties[0][:2])
        self.n_images = int(data_properties[1][:3])
        self.spotlight = data_properties[2] == 'SPOTLIGHT'
        self.start_at_rest = data_properties[3] == 'START AT REST'
        self.fps = int(data_properties[4][:2])
        if self.fps == 25:
            self.interpolated = False
        else:
            self.interpolated = True
        if len(data_properties) > 5:
            self.deblurred = data_properties[5] == 'DEBLURRED'
        else:
            self.deblurred = False

    def __str__(self):
        return self.folder_name

    def get_images(self):
        return sorted(glob.glob(os.path.join(self.path, '*')))

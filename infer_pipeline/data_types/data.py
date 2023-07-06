import os
import glob


class Data:
    def __init__(self, path):
        self.path = path
        self.folder_name = self.path.split('/')[-1]
        data_properties = self.folder_name.split(' - ')
        self.id = int(data_properties[0][:3])
        self.n_images = int(data_properties[1][:4])
        self.spotlight = data_properties[2] == 'SPOTLIGHT'
        self.fps = int(data_properties[3][:3])
        self.deblurred = False
        self.interpolated = False
        self.deblurred_interpolated = False
        self.interpolated_deblurred = False
        self.still = False

        if len(data_properties) > 4:
            match data_properties[4]:
                case 'DEBLURRED':
                    self.deblurred = True
                case 'INTERPOLATED':
                    self.interpolated = True
                case 'DEBLURRED-INTERPOLATED':
                    self.deblurred_interpolated = True
                case 'INTERPOLATED-DEBLURRED':
                    self.interpolated_deblurred = True
                case 'STILL':
                    self.still = True

    def __str__(self):
        return self.folder_name

    def get_images(self):
        return sorted(glob.glob(os.path.join(self.path, '*')))

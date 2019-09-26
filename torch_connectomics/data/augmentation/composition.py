from __future__ import division

import random
import warnings
import numpy as np

from skimage.morphology import dilation,erosion
from skimage.filters import gaussian

class Compose(object):
    """Compose transforms

    Args:
        transforms (list): list of transformations to compose.
        input_size (tuple): input size of model in (z, y, x).
        keep_uncropped (bool): keep uncropped images and labels (default: False).
        keep_non_smooth (bool): return also the non-smoothed masks (default: False).
    """
    def __init__(self, 
                 transforms, 
                 input_size = (8,196,196),
                 keep_uncropped = False,
                 keep_non_smoothed = False):
        self.transforms = transforms
        self.input_size = np.array(input_size)
        self.sample_size = self.input_size.copy()
        self.set_sample_params()
        self.keep_uncropped = keep_uncropped
        self.keep_non_smoothed = keep_non_smoothed

    def set_sample_params(self):
        for _, t in enumerate(self.transforms):
            self.sample_size = np.ceil(self.sample_size * t.sample_params['ratio']).astype(int)
            self.sample_size = self.sample_size + (2 * np.array(t.sample_params['add']))
        print('Sample size required for the augmentor:', self.sample_size)

    def smooth_edge(self, data):
        smoothed_label = data['label'].copy()

        for z in range(smoothed_label.shape[0]):
            temp = smoothed_label[z].copy()
            for idx in np.unique(temp):
                if idx != 0:
                    binary = (temp==idx).astype(np.uint8)
                    for _ in range(2):
                        binary = dilation(binary)
                        binary = gaussian(binary, sigma=2, preserve_range=True)
                        binary = dilation(binary)
                        binary = (binary > 0.8).astype(np.uint8)
            
                    temp[np.where(temp==idx)]=0
                    temp[np.where(binary==1)]=idx
            smoothed_label[z] = temp

        data['label'] = smoothed_label
        return data

    def crop(self, data):
        image, label = data['image'], data['label']

        if 'mask' in data and data['mask'] is not None:
            mask = data['mask']
        else:
            mask = None

        assert image.shape[-3:] == label.shape
        assert image.ndim == 3 or image.ndim == 4
        margin = (label.shape[1] - self.input_size[1]) // 2
        margin = int(margin)
        
        # whether need to crop z or not (missing section augmentation)
        if label.shape[0] > self.input_size[0]:
            z_low = np.random.choice(label.shape[0]-self.input_size[0]+1, 1)[0]
        else:
            z_low = 0
        z_high = z_low + self.input_size[0] 
        z_low, z_high = int(z_low), int(z_high)

        if margin==0:
            if mask is None:
                return {'image': image, 'label': label}
            else:
                return {'image': image, 'label': label, 'mask': mask}
        else:    
            low = margin
            high = margin + self.input_size[1]
            if image.ndim == 3:
                if self.keep_uncropped == True:
                    if mask is None:
                        return {'image': image[z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high],
                                'image_uncropped': image,
                                'label_uncropped': label}
                    else:
                        return {'image': image[z_low:z_high, low:high, low:high],
                            'label': label[z_low:z_high, low:high, low:high],
                            'mask': mask[z_low:z_high, low:high, low:high],
                            'image_uncropped': image,
                            'label_uncropped': label,
                            'mask_uncropped': mask}

                else:
                    if mask is None:
                        return {'image': image[z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high]}
                    else:
                        return {'image': image[z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high],
                                'mask': mask[z_low:z_high, low:high, low:high]}

            else:
                if self.keep_uncropped == True:
                    if mask is not None:
                        return {'image': image[:, z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high],
                                'mask': mask[z_low:z_high, low:high, low:high],
                                'image_uncropped': image,
                                'label_uncropped': label,
                                'mask_uncropped': mask}
                    else:
                        return {'image': image[:, z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high],
                                'image_uncropped': image,
                                'label_uncropped': label}
                else:
                    if mask is not None:
                        return {'image': image[:, z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high],
                                'mask': mask[z_low:z_high, low:high, low:high]}
                    else:
                        return {'image': image[:, z_low:z_high, low:high, low:high],
                                'label': label[z_low:z_high, low:high, low:high]}

    def __call__(self, data, random_state=None):
        data['image'] = data['image'].astype(np.float32)
        for t in reversed(self.transforms):
            if random.random() < t.p:
                data = t(data, random_state)

        # crop the data to input size
        if self.keep_uncropped:
            data['uncropped_image'] = data['image']
            data['uncropped_label'] = data['label']
            data['uncropped_mask']  = data['mask']
        data = self.crop(data)
        if self.keep_non_smoothed:
            data['non_smoothed'] = data['label']
        data = self.smooth_edge(data)
        return data
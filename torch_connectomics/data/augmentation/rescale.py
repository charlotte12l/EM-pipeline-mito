from __future__ import division
import cv2
import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class Rescale(DataAugment):
    """
    Rescale augmentation.
    
    Args:
        low (float): lower bound of the random scale factor.
        high (float): higher bound of the random scale factor.
        fix_aspect (bool): fix aspect ratio or not.
        p (float): probability of applying the augmentation
    """
    def __init__(self, low=0.8, high=1.2, fix_aspect=False, p=0.5):
        super(Rescale, self).__init__(p=p) 
        self.low = low
        self.high = high
        self.fix_aspect = fix_aspect

        self.image_interpolation = 1
        self.label_interpolation = 0
        self.set_params()

    def set_params(self):
        assert (self.low >= 0.5)
        assert (self.low <= 1.0)
        ratio = 1.0 / self.low
        self.sample_params['ratio'] = [1.0, ratio, ratio]

    def random_scale(self, random_state):
        rand_scale = random_state.rand() * (self.high - self.low) + self.low
        return rand_scale

    def apply_rescale(self, image, label, mask, sf_x, sf_y, random_state):
        # apply image and mask at the same time
        transformed_image = image.copy()
        transformed_label = label.copy()
        
        if mask is not None:
            transformed_mask = mask.copy()

        y_length = int(sf_y * image.shape[1])
        if y_length <= image.shape[1]:
            y0 = random_state.randint(low=0, high=image.shape[1]-y_length+1)
            y1 = y0 + y_length
            transformed_image = transformed_image[:, y0:y1, :]
            transformed_label = transformed_label[:, y0:y1, :]
            if mask is not None:
                transformed_mask = transformed_mask[:, y0:y1, :]
        else:
            y0 = int(np.floor((y_length - image.shape[1]) / 2))
            y1 = int(np.ceil((y_length - image.shape[1]) / 2))
            transformed_image = np.pad(transformed_image, ((0, 0),(y0, y1),(0, 0)), mode='constant')
            transformed_label = np.pad(transformed_label, ((0, 0),(y0, y1),(0, 0)), mode='constant')

            if mask is not None:
                transformed_mask = np.pad(transformed_mask, ((0, 0), (y0, y1), (0, 1)), mode='constant')

        x_length = int(sf_x * image.shape[2])
        if x_length <= image.shape[2]:
            x0 = random_state.randint(low=0, high=image.shape[2]-x_length+1)
            x1 = x0 + x_length
            transformed_image = transformed_image[:, :, x0:x1]
            transformed_label = transformed_label[:, :, x0:x1]
            if mask is not None:
                transformed_mask = transformed_mask[:, :, x0:x1]

        else:
            x0 = int(np.floor((x_length - image.shape[2]) / 2))
            x1 = int(np.ceil((x_length - image.shape[2]) / 2))
            transformed_image = np.pad(transformed_image, ((0, 0),(0, 0),(x0, x1)), mode='constant')
            transformed_label = np.pad(transformed_label, ((0, 0),(0, 0),(x0, x1)), mode='constant')
            if mask is not None:
                transformed_mask = np.pad(transformed_mask, ((0, 0),(0, 0),(x0, x1)), mode='constant')
        output_image = resize(transformed_image, image.shape, order=self.image_interpolation, mode='constant', cval=0, 
                              clip=True, preserve_range=True, anti_aliasing=True)
        output_label = resize(transformed_label, image.shape, order=self.label_interpolation, mode='constant', cval=0, 
                              clip=True, preserve_range=True, anti_aliasing=False)  
        if mask is not None:
            output_mask = resize(transformed_mask, image.shape, order=self.image_interpolation, mode='constant', cval=0, 
                              clip=True, preserve_range=True, anti_aliasing=True)
        else:
            output_mask = None

        return output_image, output_label, output_mask

    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)

        # if 'label' in data and data['label'] is not None:
        #     image, label = data['image'], data['label']
        # else:
        #     image, label = data['image'], None

        image = data['image']

        if 'label' in data and data['label'] is not None:
            label = data['label']
        else:
            label = None

        if 'mask' in data and data['mask'] is not None:
            mask = data['mask']
        else:
            mask = None

        if self.fix_aspect:
            sf_x = self.random_scale(random_state)
            sf_y = sf_x
        else:
            sf_x = self.random_scale(random_state)
            sf_y = self.random_scale(random_state)

        output = {}
        if mask is not None:
            output['image'], output['label'], output['mask'] = self.apply_rescale(image, label, mask, sf_x, sf_y, random_state)
        else:
            output['image'], output['label'], _ = self.apply_rescale(image, label, None, sf_x, sf_y, random_state)
        # print(f'Rescale Keys: {output.keys()}')
        return output
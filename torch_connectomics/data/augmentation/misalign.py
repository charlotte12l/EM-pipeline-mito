import math
import numpy as np
from .augmentor import DataAugment

class MisAlignment(DataAugment):
    """Mis-alignment data augmentation of image stacks.
    
    Args:
        displacement (int): maximum pixel displacement in each direction (x and y).
        p (float): probability of applying the augmentation.
    """
    def __init__(self, displacement=16, p=0.5):
        super(MisAlignment, self).__init__(p=p)
        self.displacement = 16
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [0, 
                                     int(math.ceil(self.displacement / 2.0)), 
                                     int(math.ceil(self.displacement / 2.0))]

    def misalignment(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        if 'mask' in data and data['mask'] is not None:
            mask = data['mask'].copy()
        else:
            mask = None

        out_shape = (images.shape[0], 
                     images.shape[1]-self.displacement, 
                     images.shape[2]-self.displacement)    
        new_images = np.zeros(out_shape, images.dtype)
        new_labels = np.zeros(out_shape, labels.dtype)

        if mask is not None:
            new_mask = np.zeros(out_shape, mask.dtype)

        x0 = random_state.randint(self.displacement)
        y0 = random_state.randint(self.displacement)
        x1 = random_state.randint(self.displacement)
        y1 = random_state.randint(self.displacement)

        idx = np.random.choice(np.array(range(1, out_shape[0]-1)), 1)[0]
        if random_state.rand() < 0.5:
            # slip misalignment
            new_images = images[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_labels = labels[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_images[idx] = images[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
            new_labels[idx] = labels[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]

            if mask is not None:
                new_mask = mask[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
                new_mask[idx] = mask[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
        else:
            # translation misalignment
            new_images[:idx] = images[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_labels[:idx] = labels[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            new_images[idx:] = images[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]
            new_labels[idx:] = labels[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]

            if mask is not None:
                new_mask[:idx] = mask[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
                new_mask[idx:] = mask[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]

        data = {}
        data['image'] = new_images
        data['label'] = new_labels

        if mask is not None:
            data['mask'] = new_mask

        return data

    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)

        output = self.misalignment(data, random_state)

        # print(f'MissingSection Keys: {output.keys()}')

        return output
        # new_images, new_labels = self.misalignment(data, random_state)
        # return {'image': new_images, 'label': new_labels}

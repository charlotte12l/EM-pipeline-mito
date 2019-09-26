import math
import numpy as np
from .augmentor import DataAugment

class MissingSection(DataAugment):
    """Missing-section augmentation of image stacks
    
    Args:
        num_sections (int): number of missing sections.
        p (float): probability of applying the augmentation.
    """
    def __init__(self, num_sections=2, p=0.5):
        super(MissingSection, self).__init__(p=p)
        self.num_sections = 2
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [int(math.ceil(self.num_sections / 2.0)), 0, 0]

    def missing_section(self, data, random_state):
        images, labels = data['image'], data['label']

        if 'mask' in data and data['mask'] is not None:
            mask = data['mask']
        else:
            mask = None

        new_images = images.copy()   
        new_labels = labels.copy()

        idx = random_state.choice(np.array(range(1, images.shape[0]-1)), self.num_sections, replace=False)

        new_images = np.delete(new_images, idx, 0)
        new_labels = np.delete(new_labels, idx, 0)

        data = {}
        data['image'] = new_images
        data['label'] = new_labels

        if mask is not None:
            new_mask = mask.copy()
            new_mask = np.delete(new_mask, idx, 0)
            data['mask'] = new_mask

        return data
    
    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)

        output = self.missing_section(data, random_state)
        
        # print(f'MissingSection Keys: {output.keys()}')

        return output

        # if 'mask' in data and data['mask'] is not None:
        #     new_images, new_labels, new_mask = self.missing_section(data, random_state)
        #     return {'image': new_images, 'label': new_labels, 'mask': new_mask}
        # else:
        #     new_images, new_labels, _ = self.missing_section(data, random_state)
        #     return {'image': new_images, 'label': new_labels}
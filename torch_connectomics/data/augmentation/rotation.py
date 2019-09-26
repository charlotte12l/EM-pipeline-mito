import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    """
    Continuous rotatation.

    The sample size for x- and y-axes should be at least sqrt(2) times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation
    """
    def __init__(self, p=0.5):
        super(Rotate, self).__init__(p=p) 
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        self.sample_params['ratio'] = [1.0, 1.42, 1.42]

    def rotate(self, imgs, M, interpolation):
        height, width = imgs.shape[-2:]
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            dst = cv2.warpAffine(img, M ,(height,width), 1.0, flags=interpolation, borderMode=self.border_mode)
            transformedimgs[z, :, :] = dst

        return transformedimgs

    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)
        # import pdb; pdb.set_trace()
        # print(data.keys())
        image = data['image']

        height, width = image.shape[-2:]
        M = cv2.getRotationMatrix2D((height/2, width/2), random_state.rand()*360.0, 1)

        output = {}
        output['image'] = self.rotate(image, M, self.image_interpolation)

        if 'label' in data and data['label'] is not None:
            output['label'] = self.rotate(data['label'], M, self.label_interpolation)

        if 'mask' in data and data['mask'] is not None:
            output['mask'] = self.rotate(data['mask'], M, self.label_interpolation)
        # print(f'Rotate Keys: {output.keys()}')
        return output
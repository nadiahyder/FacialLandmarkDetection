from skimage import io, transform, util
import numpy as np
import torch
import skimage.color as skc
import skimage.transform as skt
import random
from math import *

class BW(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = skc.rgb2gray(image)
        image = image.astype(np.float32) / 255 - 0.5
        return {'image': image, 'landmarks': landmarks}

class RandomNoise(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = util.random_noise(image)
        return {'image': image, 'landmarks': landmarks}

class ToFloat(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.astype(np.float32) / 255 - 0.5
        return {'image': image, 'landmarks': landmarks}

class Jitter(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        while True:
            val = random.uniform(1, 1.1)
            if val != 0.0:
                break

        contrast = val

        while True:
            val = random.uniform(0, 0.1)
            if val != 0.0:
                break

        brightness = val

        image = image * contrast + brightness
        return {'image': image, 'landmarks': landmarks}

class HorizontalShift(object):
    def __call__(self, sample):
        pixels = [-10, -5, 0, 5, 10]
        num_pixels = pixels[random.randint(0, 4)]
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        image = np.roll(image, num_pixels, axis=1)
        landmarks["x"] += (num_pixels / w)
        return {'image': image, 'landmarks': landmarks}

class VerticalShift(object):
    def __call__(self, sample):
        pixels = [-10, -5, 0, 5, 10]
        num_pixels = pixels[random.randint(0, 4)]
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        image = np.roll(image, num_pixels, axis=0)
        landmarks["y"] += (num_pixels / h)
        return {'image': image, 'landmarks': landmarks}

class Resize(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = skt.resize(image, (self.height, self.width), anti_aliasing=True)
        #reader.adjust_shape(landmarks, image)
        #reader.show_landmarks(landmarks, image)
        return {'image': image, 'landmarks': landmarks}

class Rotate(object):
    def __call__(self, sample):
        angles = [-10, -5, 0, 5, 10]
        angle = angles[random.randint(0, 4)]
        image, landmarks = sample['image'], sample['landmarks']

        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = skt.rotate(image/255, angle, preserve_range=True)
        image = image*255

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5

        return {'image': image, 'landmarks': new_landmarks}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    # fixme are these supposed to change landmarks
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        #print(h,w)
        #print(self.output_size)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # landmarks["x"] = (new_w / w) * landmarks["x"]
        # landmarks["y"] = (new_h / h) * landmarks["y"]
        landmarks["x"] *= new_h/h
        landmarks["y"] *= new_w/w
        #print(img.shape)

        #reader.show_landmarks(landmarks, img)
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        #print(image.shape)

        h, w = image.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks["x"]-= left
        landmarks["y"]-=top
        #reader.show_landmarks(landmarks, image)
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))

        # image = Image.fromarray(image)
        # image = TF.to_tensor(image)
        # image = TF.normalize(image, [0.5], [0.5])
        # return {'image': image,
        #         'landmarks': torch.from_numpy(landmarks.to_numpy())}

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks.to_numpy())}

        #alternatively
        # return {'image': torch.from_numpy(image),
        #         'landmarks': torch.from_numpy(landmark_points(landmarks))}

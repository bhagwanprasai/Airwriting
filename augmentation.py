"""
Augmentation Module
Contains all image augmentation functions for training
"""

import numpy as np
import cv2


def elastic_transform(image, alpha=30, sigma=4, random_state=None):
    """Elastic distortion"""
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y+dy).astype(np.float32), (x+dx).astype(np.float32)
    return cv2.remap(image, indices[1], indices[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def cutout_augmentation(image, min_scale=0.02, max_scale=0.15):
    """Cutout/Random Erasing"""
    h, w = image.shape
    scale = np.random.uniform(min_scale, max_scale)
    mask_h = int(h * scale)
    mask_w = int(w * scale)
    
    y = np.random.randint(0, h - mask_h + 1) if h > mask_h else 0
    x = np.random.randint(0, w - mask_w + 1) if w > mask_w else 0
    
    image = image.copy()
    image[y:y+mask_h, x:x+mask_w] = 0
    return image


def add_gaussian_noise(image, mean=0, sigma=0.05):
    """Gaussian noise"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def apply_heavy_aug(image):
    """Enhanced augmentation pipeline"""
    from config import Config
    
    h, w = image.shape
    
    if np.random.random() < Config.ELASTIC_PROB:
        image = elastic_transform(image, alpha=h*0.4, sigma=h*0.08)
    
    if np.random.random() < 0.5:
        angle = np.random.uniform(-7, 7)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    if np.random.random() < 0.5:
        shear_factor = np.random.uniform(-0.25, 0.25)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        new_w = w + int(abs(shear_factor*h))
        image = cv2.warpAffine(image, M, (new_w, h), borderMode=cv2.BORDER_REPLICATE)
    
    if np.random.random() < 0.2:
        kernel = np.ones((2,2), np.uint8)
        if np.random.random() < 0.5:
            image = cv2.erode(image, kernel, iterations=1)
        else:
            image = cv2.dilate(image, kernel, iterations=1)
    
    return image

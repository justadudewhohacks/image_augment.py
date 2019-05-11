import cv2
import random
import math
import numpy as np
from augment.augment import augment, abs_coords

img = cv2.imread('./face.jpg')

key = -1
while key != 32:

  #roi = [0.35, 0.25, 0.65, 0.8]
  roi = [0.1, 0.1, 0.9, 0.9]

  blur_config = { 'kernel_size':  random.choice([0, 3, 5, 7, 11]), 'std_dev': random.uniform(0.5, 1.5) }
  blur_prob = 0.5
  gray_prob = 0.1

  res = augment(
    img,
    flip = random.random() < 0.5,
    rotation_angle = random.uniform(-15, 15),
    blur = blur_config if random.random() < blur_prob else None,
    intensity = { 'alpha': random.uniform(0.5, 1.5), 'beta': random.uniform(-20, 20) },
    hsv = [random.uniform(-5, 5), random.uniform(-15, 15), random.uniform(-20, 20)],
    to_gray = random.random() < gray_prob,
    shear = [random.uniform(0.0, 0.2), random.uniform(0.0, 0.2)],
    random_crop = roi,
    stretch = { 'stretch_x': random.uniform(1.0, 1.4), 'stretch_y': random.uniform(1.0, 1.4) }
  )

  abs_roi = abs_coords(roi, img)
  cv2.imshow('roi', img[abs_roi[1]:abs_roi[3], abs_roi[0]:abs_roi[2]])
  cv2.imshow('res', res)
  key = cv2.waitKey(0)

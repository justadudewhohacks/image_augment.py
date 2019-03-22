import cv2
import random
import math
import numpy as np
from augment import augment, abs_coords

img = cv2.imread('./lenna.jpg')

key = -1
while key != 32:

  roi = [0.35, 0.25, 0.65, 0.8]

  sx = random.uniform(0.0, 0.2)
  sy = random.uniform(0.0, 0.2)
  res = augment(
    img,
    flip = random.random() < 0.5,
    rotation_angle = random.uniform(-5, 5),
    blur = { 'kernel_size':  random.choice([3, 5, 7]), 'std_dev': random.uniform(0.8, 1.2) },
    intensity = { 'alpha': random.uniform(0.8, 1.2), 'beta': random.uniform(-10, 10) },
    to_gray = random.random() < 0.1,
    shear = [sx, sy],
    random_crop = roi,
    stretch = { 'stretch_x': random.uniform(1.0, 1.4), 'stretch_y': random.uniform(1.0, 1.4) }
  )

  abs_roi = abs_coords(roi, img)
  cv2.imshow('roi', img[abs_roi[1]:abs_roi[3], abs_roi[0]:abs_roi[2]])
  cv2.imshow('res', res)
  key = cv2.waitKey(0)

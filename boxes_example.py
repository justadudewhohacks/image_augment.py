import cv2
import random
import math
import numpy as np
from augment.augment import augment, abs_coords

def draw_box(img, box):
  x, y, w, h = abs_coords(box, img)

  cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
  cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
  cv2.circle(img, (x, y + h), 2, (0, 0, 255), -1)
  cv2.circle(img, (x + w, y), 2, (0, 0, 255), -1)
  cv2.circle(img, (x + w, y + h), 2, (0, 0, 255), -1)

img = cv2.imread('./lenna.jpg')

key = -1
while key != 32:

  boxes = [(0.05, 0.2, 0.25, 0.35), (0.45, 0.4, 0.15, 0.3), (0.55, 0.85, 0.1, 0.1)]
  #roi = [0.35, 0.25, 0.65, 0.8]
  roi = [0.35, 0.2, 0.65 - 0.35, 0.8 - 0.2]
  roi = [0, 0, 1.0, 1.0]

  blur_config = { 'kernel_size':  random.choice([0, 3, 5, 7, 11]), 'std_dev': random.uniform(0.5, 1.5) }
  blur_prob = 0.5
  gray_prob = 0.1

  res, res_boxes = augment(
    img,
    boxes = boxes,
    random_crop = roi,
    flip = random.random() < 0.5,
    stretch = { 'stretch_x': random.uniform(1.0, 1.4), 'stretch_y': random.uniform(1.0, 1.4) },
    shear = [random.uniform(0.0, 0.2), random.uniform(0.0, 0.2)],
    rotation_angle = random.uniform(-15, 15),

    pad_to_square = True,
    resize = 400,

    blur = blur_config if random.random() < blur_prob else None,
    intensity = { 'alpha': random.uniform(0.5, 1.5), 'beta': random.uniform(-20, 20) },
    hsv = [random.uniform(-5, 5), random.uniform(-15, 15), random.uniform(-20, 20)],
    to_gray = random.random() < gray_prob
  )

  draw_img = img.copy()
  for box in boxes:
    draw_box(draw_img, box)
  cv2.imshow('img', draw_img)

  for box in res_boxes:
    draw_box(res, box)
  cv2.imshow('res', res)

  abs_roi = abs_coords(roi, img)
  cv2.imshow('roi', img[abs_roi[1]:(abs_roi[1] + abs_roi[3]), abs_roi[0]:(abs_roi[0] + abs_roi[2])])

  key = cv2.waitKey(0)

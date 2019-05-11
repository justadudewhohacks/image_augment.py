import cv2
import math
import random
import numpy as np

def num_in_range(val, min_val, max_val):
  return min(max(min_val, val), max_val)

def abs_coords(bbox, img):
  height, width = img.shape[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [int(round(min_x * width)), int(round(min_y * height)), int(round(max_x_or_w * width)), int(round(max_y_or_h * height))]

def rel_coords(bbox, img):
  height, width = img.shape[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [min_x / width, min_y / height, max_x_or_w / width, max_y_or_h / height]

def default_box(img):
  height, width = img.shape[:2]
  return [0, 0, width, height]

def apply_intensity_adjustment(img, params):
  has_alpha = 'alpha' in params
  has_beta = 'beta' in params

  if not has_alpha and not has_beta:
    print('warning: intensity dict neither has alpha nor beta')

  alpha = params['alpha'] if has_alpha else 1.0
  beta = params['beta'] if has_alpha else 0.0

  return np.clip((img * [alpha, alpha, alpha] + [beta, beta, beta]), 0, 255).astype(img.dtype)

def apply_hsv_adjustment(img, hsv):
  if not len(hsv) == 3:
    raise Exception('hsv must contain 3 values')

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = np.clip(img + hsv, 0, 255).astype(img.dtype)
  img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

  return img

def apply_to_gray(img):
  return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def apply_blur(img, params):
  has_kernel_size = 'kernel_size' in params
  has_std_dev = 'std_dev' in params

  if not has_kernel_size and not has_std_dev:
    print('warning: blur dict neither has kernel_size nor std_dev')

  kernel_size = params['kernel_size'] if has_kernel_size else 3
  std_dev = params['std_dev'] if has_std_dev else 1.0

  return cv2.GaussianBlur(img, (kernel_size, kernel_size), std_dev, std_dev)

def apply_random_crop(img, roi, boxes = None):
  height, width = img.shape[:2]

  x, y, w, h = roi
  min_x, min_y, max_x, max_y = x, y, x + w, y + h
  min_x = int(num_in_range(min_x, 0, 1) * width)
  min_y = int(num_in_range(min_y, 0, 1) * height)
  max_x = int(num_in_range(max_x, 0, 1) * width)
  max_y = int(num_in_range(max_y, 0, 1) * height)
  x0 = random.randint(0, min_x)
  y0 = random.randint(0, min_y)
  x1 = random.randint(0, abs(width - max_x)) + max_x
  y1 = random.randint(0, abs(height - max_y)) + max_y

  cropped_img = img[y0:y1, x0:x1]

  shifted_boxes = None
  if boxes is not None:
    shifted_boxes = []
    for box in boxes:
      x, y, w, h = abs_coords(box, img)
      sx = x - x0
      sy = y - y0
      shifted_boxes.append(rel_coords((sx, sy, w, h), cropped_img))

  return cropped_img, shifted_boxes

def apply_rotate(img, angle):
  height, width = img.shape[:2]
  cx, cy = int(width / 2), int(height / 2)

  M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  new_width = int((height * sin) + (width * cos))
  new_height = int((height * cos) + (width * sin))

  M[0, 2] += (new_width / 2) - cx
  M[1, 2] += (new_height / 2) - cy

  return cv2.warpAffine(img, M, (new_width, new_height))

def apply_stretch(img, params, boxes = None):
  has_x = 'stretch_x' in params
  has_y = 'stretch_y' in params

  height, width = img.shape[:2]

  if not has_x and not has_y:
    print('warning: random_stretch dict neither has x nor y')

  stretch_x = params['stretch_x'] if has_x else 1.0
  stretch_y = params['stretch_y'] if has_y else 1.0

  shape_stretch_x = (int(round(stretch_x * width)), height)
  shape_stretch_y = (width, int(round(stretch_y * height)))

  shape = random.choice([shape_stretch_x, shape_stretch_y]) if has_x and has_y else (shape_stretch_y if not has_x else shape_stretch_x)

  stretched_img = cv2.resize(img, shape)

  stretched_boxes = None
  if boxes is not None:
    stretched_boxes = []
    orig_h, orig_w = img.shape[0:2]
    new_h, new_w = stretched_img.shape[0:2]
    rx = new_w / orig_w
    ry = new_h / orig_h
    for box in boxes:
      x, y, w, h = abs_coords(box, img)
      stretched_boxes.append(rel_coords((x * rx, y * ry, w * rx, h * ry), stretched_img))

  return stretched_img, stretched_boxes

def apply_shear(img, shear):
  height, width = img.shape[:2]

  shear_h, shear_v = shear
  shear_matrix = np.array([[1, shear_h, 0], [shear_v, 1, 0]])

  shape = (width + int(round(height * shear_h)), height + int(round(width * shear_v)))

  return cv2.warpAffine(img, shear_matrix, shape)

def apply_flip(img, is_flip, boxes = None):
  out_img = cv2.flip(img, 1) if is_flip is True else img
  out_boxes = boxes
  if boxes is not None and is_flip:
    out_boxes = []
    for box in boxes:
      x, y, w, h = box
      out_boxes.append((1.0 - (x + w), y, w, h))

  return out_img, out_boxes

def augment(
  img,
  boxes = None,
  intensity = None,
  hsv = None,
  blur = None,
  to_gray = False,
  random_crop = None,
  stretch = None,
  shear = None,
  flip = False,
  rotation_angle = None
):
  img = apply_intensity_adjustment(img, intensity) if intensity is not None else img
  img = apply_hsv_adjustment(img, hsv) if hsv is not None else img
  img = apply_blur(img, blur) if blur is not None else img
  img = apply_to_gray(img) if to_gray is True else img

  # TODO noise


  # TODO roi rotate, shift and shear before random_crop, return transormed roi
  if random_crop:
    img, boxes = apply_random_crop(img, random_crop, boxes)

  img, boxes = apply_stretch(img, stretch, boxes) if stretch is not None else (img, boxes)
  img = apply_shear(img, shear) if shear is not None else img
  img, boxes = apply_flip(img, flip, boxes)
  img = apply_rotate(img, rotation_angle) if rotation_angle is not None else img

  if boxes is not None:
    return img, boxes

  return img
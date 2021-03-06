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

def get_random_crop_options(random_crop):
  roi = random_crop
  crop_range = 0
  apply_before_transform = True
  if isinstance(random_crop, dict):
    roi = random_crop['roi']
    crop_range = random_crop['crop_range']
    if 'apply_before_transform' in random_crop:
      apply_before_transform = random_crop['apply_before_transform']
  return roi, crop_range, apply_before_transform

def apply_random_crop(img, roi, crop_range, boxes = None, pad_to_square = False):
  height, width = img.shape[:2]

  x, y, w, h = roi
  min_x, min_y, max_x, max_y = x, y, x + w, y + h
  min_x = int(num_in_range(min_x, 0, 1) * width)
  min_y = int(num_in_range(min_y, 0, 1) * height)
  max_x = int(num_in_range(max_x, 0, 1) * width)
  max_y = int(num_in_range(max_y, 0, 1) * height)
  x0 = random.randint(round(crop_range * min_x), min_x)
  y0 = random.randint(round(crop_range * min_y), min_y)
  x1 = random.randint(0, round((1.0 - crop_range) * abs(width - max_x))) + max_x
  y1 = random.randint(0, round((1.0 - crop_range) * abs(height - max_y))) + max_y

  # todo pad to square
  if pad_to_square:
    height_new = y1 - y0
    width_new = x1 - x0
    pad = int(abs(height_new - width_new) / 2)
    if height_new > width_new:
      x0 = max(0, x0 - pad)
      x1 = min(width, x1 + pad)
    if width_new > height_new:
      y0 = max(0, y0 - pad)
      y1 = min(height, y1 + pad)

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

def apply_rotate(img, angle, boxes = None):
  height, width = img.shape[:2]
  cx, cy = int(width / 2), int(height / 2)

  rotation_matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
  cos = np.abs(rotation_matrix[0, 0])
  sin = np.abs(rotation_matrix[0, 1])

  new_width = int((height * sin) + (width * cos))
  new_height = int((height * cos) + (width * sin))

  rotation_matrix[0, 2] += (new_width / 2) - cx
  rotation_matrix[1, 2] += (new_height / 2) - cy

  rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

  transform_point = lambda x, y: np.reshape(cv2.transform(np.reshape([x, y],(1, 1, 2)), rotation_matrix), 2)

  rotated_boxes = boxes
  if boxes is not None:
    rotated_boxes = []
    for box in boxes:
      x0, y0, w, h = abs_coords(box, img)
      p0 = transform_point(x0, y0)
      p1 = transform_point(x0 + w, y0)
      p2 = transform_point(x0, y0 + h)
      p3 = transform_point(x0 + w, y0 + h)

      xs = [p0[0], p1[0], p2[0], p3[0]]
      ys = [p0[1], p1[1], p2[1], p3[1]]
      new_x0 = min(xs)
      new_y0 = min(ys)
      new_x1 = max(xs)
      new_y1 = max(ys)
      rotated_boxes.append(rel_coords((new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0), rotated_img))

  return rotated_img, rotated_boxes

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

def apply_shear(img, shear, boxes = None):
  height, width = img.shape[:2]

  shear_h, shear_v = shear
  shear_matrix = np.array([[1, shear_h, 0], [shear_v, 1, 0]])

  shape = (width + int(round(height * shear_h)), height + int(round(width * shear_v)))

  sheared_img = cv2.warpAffine(img, shear_matrix, shape)

  transform_point = lambda x, y: (shear_matrix[0][0] * x + shear_matrix[0][1] * y, shear_matrix[1][0] * x + shear_matrix[1][1] * y)

  sheared_boxes = None
  if boxes is not None:
    sheared_boxes = []
    for box in boxes:
      x0, y0, w, h = abs_coords(box, img)
      tx0, ty0 = transform_point(x0, y0)
      # TODO: only translate x0, y0?
      #x1, y1 = x0 + w, y0 + h
      #tx1, ty1 = transform_point(x1, y1)
      sheared_boxes.append(rel_coords((tx0, ty0, w, h), sheared_img))

  return sheared_img, sheared_boxes

def flip_box(box):
  x, y, w, h = box
  return (1.0 - (x + w), y, w, h)

def apply_flip(img, boxes = None):
  out_img = cv2.flip(img, 1)
  out_boxes = boxes
  if boxes is not None:
    out_boxes = []
    for box in boxes:
      out_boxes.append(flip_box(box))

  return out_img, out_boxes


def apply_resize_preserve_aspect_ratio(img, size):
  height, width = img.shape[:2]
  max_dim = max(height, width)
  ratio = size / float(max_dim)
  resized_img = cv2.resize(img, (int(round(width * ratio)), int(round(height * ratio))))

  return resized_img

def apply_pad_to_square(img, boxes = None):
  if len(img.shape) == 2:
    img = np.expand_dims(img, axis = 2)

  height, width, channels = img.shape
  max_dim = max(height, width)
  out_img = np.zeros([max_dim, max_dim, channels], dtype = img.dtype)

  dx = math.floor(abs(max_dim - width) / 2)
  dy = math.floor(abs(max_dim - height) / 2)
  out_img[dy:dy + height,dx:dx + width] = img

  out_boxes = boxes
  if boxes is not None:
    out_boxes = []
    for box in boxes:
      x, y, w, h = abs_coords(box, img)
      out_boxes.append(rel_coords((x + dx, y + dy, w, h), out_img))

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
  rotation_angle = None,
  flip = False,
  pad_to_square = False,
  resize = None
):
  img = apply_intensity_adjustment(img, intensity) if intensity is not None else img
  img = apply_hsv_adjustment(img, hsv) if hsv is not None else img
  img = apply_blur(img, blur) if blur is not None else img
  img = apply_to_gray(img) if to_gray is True else img

  # TODO noise


  # TODO roi rotate, shift and shear before random_crop, return transormed roi
  if random_crop:
    crop_roi, crop_range, apply_before_transform = get_random_crop_options(random_crop)

  if random_crop and apply_before_transform:
    img, boxes = apply_random_crop(img, crop_roi, crop_range, boxes)

  img, boxes = apply_stretch(img, stretch, boxes) if stretch is not None else (img, boxes)
  img, boxes = apply_shear(img, shear, boxes) if shear is not None else (img, boxes)
  img, boxes = apply_flip(img, boxes) if flip else (img, boxes)
  img, boxes = apply_rotate(img, rotation_angle, boxes) if rotation_angle is not None else (img, boxes)

  if random_crop and not apply_before_transform:
    if flip:
      crop_roi = flip_box(crop_roi)
    img, boxes = apply_random_crop(img, crop_roi, crop_range, boxes, pad_to_square)

  img = apply_resize_preserve_aspect_ratio(img, resize) if resize else img
  img, boxes = apply_pad_to_square(img, boxes) if pad_to_square else (img, boxes)


  if boxes is not None:
    return img, boxes

  return img
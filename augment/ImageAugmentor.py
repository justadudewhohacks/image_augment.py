import json
import random

from .augment import augment

class ImageAugmentor:
  @staticmethod
  def from_json(aug_json):
    aug_config = ImageAugmentor()
    aug_dict = json.loads(aug_json)
    for key in aug_dict:
      setattr(aug_config, key, aug_dict[key])
    return aug_config

  @staticmethod
  def load(json_file_path):
    with open(json_file_path) as json_file:
      return ImageAugmentor.from_json(json.load(json_file))

  def to_json(self):
    return json.dumps(self.__dict__)

  def save(self, json_file_path):
    with open(json_file_path, 'w') as json_file:
      json_file.write(json.dumps(self.to_json()))

  def __init__(
    self,
    flip_prob = 0.0,
    rotation_prob = 0.0,
    rotation_angle_range = None,
    shear_prob = 0.0,
    shear_ranges = None,
    stretch_prob = 0.0,
    stretch_ranges = None,
    intensity_prob = 0.0,
    intensity_alpha_range = None,
    intensity_beta_range  = None,
    hsv_prob = 0.0,
    hsv_ranges = None,
    blur_prob = 0.0,
    blur_kernel_size_opts = None,
    blur_std_dev_range  = None,
    gray_prob = 0.0
  ):
    self.flip_prob = flip_prob
    self.rotation_prob = rotation_prob
    self.rotation_angle_range = rotation_angle_range
    self.shear_prob = shear_prob
    self.shear_ranges = shear_ranges
    self.stretch_prob = stretch_prob
    self.stretch_ranges = stretch_ranges
    self.intensity_prob = intensity_prob
    self.intensity_alpha_range = intensity_alpha_range
    self.intensity_beta_range = intensity_beta_range
    self.hsv_prob = hsv_prob
    self.hsv_ranges = hsv_ranges
    self.blur_prob = blur_prob
    self.blur_kernel_size_opts = blur_kernel_size_opts
    self.blur_std_dev_range = blur_std_dev_range
    self.gray_prob = gray_prob

  def augment(self, img, random_crop = None):

    def prob(p):
      return random.random() < p

    def random_in_range(range_tuple):
      if range_tuple is None:
        return None
      lower, upper = range_tuple
      return random.uniform(lower, upper)

    return augment(
      img,
      random_crop = random_crop,
      flip = prob(self.flip_prob),
      rotation_angle = random_in_range(self.rotation_angle_range) if prob(self.rotation_prob) else None,
      shear = [random_in_range(self.shear_ranges[0]), random_in_range(self.shear_ranges[1])] if prob(self.shear_prob) else None,
      stretch = { 'stretch_x': random_in_range(self.stretch_ranges[0]), 'stretch_y': random_in_range(self.stretch_ranges[1]) } if prob(self.stretch_prob) else None,
      intensity = { 'alpha': random_in_range(self.intensity_alpha_range), 'beta': random_in_range(self.intensity_beta_range) } if prob(self.intensity_prob) else None,
      hsv = [random_in_range(self.hsv_ranges[0]), random_in_range(self.hsv_ranges[1]), random_in_range(self.hsv_ranges[2])] if prob(self.hsv_prob) else None,
      blur = { 'kernel_size':  random.choice(self.blur_kernel_size_opts), 'std_dev': random_in_range(self.blur_std_dev_range) } if prob(self.blur_prob) else None,
      to_gray = prob(self.gray_prob)
    )
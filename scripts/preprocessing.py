
import cv2
import numpy as np
# import defs # for Accessing globals
from export_env_variables import *
import defs
import utils
from PIL import Image
import random
import os
from os import path
import shutil
import glob
from scipy import misc
from scipy import ndimage
import skimage
from skimage import exposure, morphology
import time

# ===================== OPEN CV PRE PROCESSING FUNCTIONS ===========================
def open_cv_histogram_equalization(img):
    img = cv2.imread(img)
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # #Image Resizing
    # img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

# -------------------------------------------------------------------------------------------------------
def open_cv_CLAHE_histogram_equalization(img):

    img = cv2.imread(img, cv2.CV_16UC1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img
# -------------------------------------------------------------------------------------------------------


def open_cv_adaptive_gaussian_thresholding(img):

    img = cv2.imread(img, cv2.CV_8UC1)
    img = cv2.medianBlur(img, 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return img

# -------------------------------------------------------------------------------------------------------
def open_cv_binary_thresholding(img):
    # mode.data_preprocessings.append("open_cv_binary_thresholding")

    img = cv2.imread(img, 0)

    # img_np = cv2.GaussianBlur(img_np, 3)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # returns a tuple

    return img

# -------------------------------------------------------------------------------------------------------
def open_cv_morphology_closing(img):
    # mode.data_preprocessings.append("open_cv_morphology_closing")

    # closes small holes in foregorund objects.

    img = cv2.imread(img, 0)
    kernel = np.ones((5, 5), np.uint8)
    # img_np = cv2.GaussianBlur(img_np, 3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img

# -------------------------------------------------------------------------------------------------------

def gaussian_blur(img):
    sigma = 1
    return skimage.img_as_ubyte(ndimage.gaussian_filter(img, sigma=sigma))
# -------------------------------------------------------------------------------------------------------

# def sharpen(img):
#     alpha=3
#     filter_blurred_f = skimage.img_as_ubyte(ndimage.gaussian_filter(img, 1))
#     return img + alpha * (img - filter_blurred_f)
# # -------------------------------------------------------------------------------------------------------

def brightness(img):
    value = random.randint(-20, 20)

    if not value: value = random.randint(-30, 20)

    if value >= 0:
        return np.where((255 - img) < value,255,img+value)
    else:
        return np.where(img < value ,0,img-value)
# -------------------------------------------------------------------------------------------------------

# # Logarithmic
# def logarithmic(img):
#     return exposure.adjust_log(img, 1)

def contrast(img):
    clip_limit = 0.01
    nbins = 256
    return skimage.img_as_ubyte(exposure.equalize_adapthist(skimage.img_as_float(img), kernel_size=None, clip_limit=clip_limit, nbins=nbins))

# -------------------------------------------------------------------------------------------------------

def get_augment_train_set(train_set, mode_augmented_data_path, augment_targets):
    all_imgs_augmented_plus_original = [path.basename(img) for img in glob.glob(mode_augmented_data_path + "/*.jpg")]

    new_train_set = []
    for target_name_label in train_set:

        target_name, label = target_name_label
        target_name_short = target_name.replace(".jpg","")
        only_target_name = target_name_short.partition("_")[0]

        if augment_targets:
            if only_target_name not in augment_targets:
                continue
        for img in all_imgs_augmented_plus_original:
            if img.count("_") > 1: # augmented_img
                first_i = img.find("_")
                second_i = img.find("_", first_i + 1)
                augmented_img_short = img[:second_i]
                if augmented_img_short == target_name_short:
                    new_train_set.append((img, label))
    extended_set = new_train_set + train_set
    random.shuffle(extended_set)
    return extended_set


# -------------------------------------------------------------------------------------------------------



def off_center_crop_image(image, new_width, new_height, to_the_left):

    was_transposed = False
    if image.shape[0] <= 4:
        # need to transpose it..
        image = image.transpose(1, 2, 0) # (chan, height, width) -> (height, width, chan)
        was_transposed = True

    height, width, chan = image.shape

    width_cut = (width - new_width) // 2
    height_cut = (height - new_height) // 2

    top, bottom = height_cut, -height_cut
    left, right = (width_cut+to_the_left), -(width_cut-to_the_left)

    # could have 1 pixel off.
    height_diff = new_height - (height - (height_cut*2))
    width_diff = new_width - (width - (width_cut*2))

    top -= height_diff
    left -= width_diff

    if was_transposed:
        return image[top:bottom, left:right].transpose(2, 0, 1) # (height, width, chan) -> (chan, height, width)
    else:
        return image[top:bottom, left:right]

# -------------------------------------------------------------------------------------------------------

def crop_off_center_augment_every_image_twice(overwrite_folder=False):
    if os.path.exists(augment_every_image_twice_path) and overwrite_folder:
        shutil.rmtree(augment_every_image_twice_path)

    random.seed(time.time())


    utils.makedirs_ok(augment_every_image_twice_path)
    all_my_model_imgs = [path.basename(img) for img in glob.glob(my_model_data + "/*.jpg")]

    all_full_size_imgs = [path.basename(img) for img in glob.glob(full_size_images_path + "/*.jpg")]

    for full_size_img in all_full_size_imgs:
        if full_size_img in all_my_model_imgs:


            shutil.copy(full_size_images_path + "/" + full_size_img, augment_every_image_twice_path)
            img = augment_every_image_twice_path + "/" + full_size_img
            img_basename = path.basename(img)


            contrast_new_basename = img_basename.partition(".jpg")[0] + "_" + str(1) + ".jpg"
            new_path = path.dirname(img) + "/" + contrast_new_basename
            shutil.copy(img, new_path)
            new_img = misc.imread(new_path)
            new_img = off_center_crop_image(new_img, TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 20)

            new_img = contrast(new_img)
            misc.imsave(new_path, new_img, "JPEG")

            blur_new_basename = img_basename.partition(".jpg")[0] + "_" + str(2) + ".jpg"
            new_path = path.dirname(img) + "/" + blur_new_basename
            shutil.copy(img, new_path)
            new_img = misc.imread(new_path)
            new_img = off_center_crop_image(new_img, TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, -20)
            new_img = gaussian_blur(new_img)
            misc.imsave(new_path, new_img, "JPEG")

            #center crop original image
            new_img = misc.imread(img)
            new_img = off_center_crop_image(new_img, TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 0)

            misc.imsave(img, new_img, "JPEG")


# -------------------------------------------------------------------------------------------------------


def augment_every_image_twice(overwrite_folder=False):
    if os.path.exists(augment_every_image_twice_path) and overwrite_folder:
        shutil.rmtree(augment_every_image_twice_path)

    random.seed(time.time())


    utils.makedirs_ok(augment_every_image_twice_path)
    utils.copy_all_files(my_model_data, augment_every_image_twice_path, "*.jpg")

    healthy_imgs = glob.glob(augment_every_image_twice_path + "/healthy_*.jpg")
    kc_imgs = glob.glob(augment_every_image_twice_path+ "/kc_*.jpg")
    cly_imgs = glob.glob(augment_every_image_twice_path + "/cly_*.jpg")
    sus_imgs = glob.glob(augment_every_image_twice_path + "/sus_*.jpg")


    functions = [contrast, gaussian_blur]

    for target in [healthy_imgs, kc_imgs, cly_imgs, sus_imgs]:
        for img in target:

            img_basename = path.basename(img)

            contrast_new_basename = img_basename.partition(".jpg")[0] + "_" +str(1) + ".jpg"
            new_path = path.dirname(img) + "/" + contrast_new_basename
            shutil.copy(img, new_path)
            new_img = misc.imread(new_path)


            new_img = contrast(new_img)
            misc.imsave(new_path, new_img, "JPEG")

            blur_new_basename = img_basename.partition(".jpg")[0] + "_" + str(2) + ".jpg"
            new_path = path.dirname(img) + "/" + blur_new_basename
            shutil.copy(img, new_path)
            new_img = misc.imread(new_path)

            new_img = gaussian_blur(new_img)
            misc.imsave(new_path, new_img, "JPEG")

# -------------------------------------------------------------------------------------------------------


def brightness_contrast_augment_data(num_of_imgs_to_have=103, overwrite_folder=False):
    if os.path.exists(augmented_data_path) and overwrite_folder:
        shutil.rmtree(augmented_data_path)

    random.seed(time.time())


    utils.makedirs_ok(augmented_data_path)
    utils.copy_all_files(my_model_data, augmented_data_path, "*.jpg")

    original_images = glob.glob(augmented_data_path + "/*.jpg")

    healthy_imgs = glob.glob(augmented_data_path + "/healthy_*.jpg")
    kc_imgs = glob.glob(augmented_data_path+ "/kc_*.jpg")
    cly_imgs = glob.glob(augmented_data_path + "/cly_*.jpg")
    sus_imgs = glob.glob(augmented_data_path + "/sus_*.jpg")


    functions = [contrast, gaussian_blur]

    for target in [healthy_imgs, kc_imgs, cly_imgs, sus_imgs]:
        img_i = 0
        for i in range(num_of_imgs_to_have - len(target)):

            img_basename = path.basename((target[img_i]))

            new_basename = img_basename.partition(".jpg")[0] + "_" +str(i) + ".jpg"
            new_path = path.dirname(target[img_i]) + "/" + new_basename
            shutil.copy(target[img_i], new_path)
            new_img = misc.imread(new_path)
            #
            num_of_functions_to_use = random.randint(1, len(functions))
            # new_img = random.choice(functions)(new_img)
            for function in random.sample(functions, num_of_functions_to_use):
                new_img = function(new_img)

            # new_img = contrast(new_img)
            misc.imsave(new_path, new_img, "JPEG")

            img_i = (img_i +1)%len(target)


    # for img in original_images:
    #     os.remove(img)













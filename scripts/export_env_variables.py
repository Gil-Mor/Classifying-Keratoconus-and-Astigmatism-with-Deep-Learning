import os
from os import path
import sys
import glob
from platform_defs import *


# ========================= GLOBALS =======================

demo_mode = False
def set_uber_demo_mode(b):
    global demo_mode
    demo_mode = b
    print("demo mode", demo_mode)

# --------------- SET MATPLOTLIB STUFF ------------------------
import matplotlib # i,port matplot lib only here so that we can configure it according to PLATFROM
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.style.use('seaborn-bright')

if EC2 not in PLATFORM:
    plt.switch_backend('Qt5Agg')  # solve bug on ubuntu. backend was agg - non interactive - can only save fig.
# plt.rcParams['figure.figsize'] = (10, 10)  # large images
# plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
# plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



TRAINING_IMAGE_SIZE = 256
CLASSIFICATION_IMAGE_SIZE = 227

healthy_label = 0
kc_label = 1

label_a = 0
label_b = 1
label_c = 2
label_d = 3
# third_label=2
# fourth_label=3

sus_label = 2
cly_label = 2


real_world_string_to_presentation_string = {"healthy":"Healthy", "kc":"KC", "sus":"KC_Sus", "cly":"CLY", "dummy":"Dummy"}

label_int_to_presentation_string = {healthy_label: "Healthy", kc_label: "KC"}

healthy_imgs = None
kc_imgs = None
cly_imgs = None
sus_imgs = None
NUM_OF_HEALTHY = None
NUM_OF_KC = None
NUM_OF_SUS = None
NUM_OF_CLY = None
NUM_OF_TOTAL_IMAGES = None
HEALTHY_TRAIN_SIZE_DEFAULT = None
KC_TRAIN_SIZE_DEFAULT = None
SUS_TRAIN_SIZE_DEFAULT = None
CLY_TRAIN_SIZE_DEFAULT = None
DEFAULT_TRAIN_SIZE_FRACTION = 0.8
DEFAULT_SUS_TRAIN_SIZE_FRACTION = 0.5
caffe = None # make global for other modules.

# ========================== PATHS ==========================

if PLATFORM == PC_Platform:
    final_project_root  = r"/home/gil/Caffe/FinalProject"
elif  EC2 in PLATFORM:
    final_project_root  = r"/home/ubuntu/FinalProject"

pycharm_logs        = path.join(final_project_root, "pycharm_logs")
last_pycharm_log    = path.join(pycharm_logs, "log.log")

if PLATFORM == PC_Platform:
    caffe_root = r"/home/gil/Caffe/caffe"
    ec2_caffe_root = r"/home/gil/Caffe/ec2_caffe"
elif PLATFORM == EC2_GPU_Platform:
    caffe_root = r"/home/ubuntu/caffe_gpu/caffe"
elif PLATFORM == EC2_CPU_Platform:
    caffe_root = r"/home/ubuntu/caffe_cpu/caffe"

caffe_tools         = path.join(caffe_root, "build/tools")
pycaffe_module_path = path.join(caffe_root, "python")


model            = r"bvlc_reference_caffenet"
my_model         = r"kc"
combined_model   = my_model + "_" + model    # kc_bvlc_reference_caffenet

my_model_path    = path.join(final_project_root, "models", combined_model)



my_model_all_data             = path.join(final_project_root, "data")
data_info_files               = path.join(my_model_all_data, "data_info_files")

my_model_data                 = path.join(my_model_all_data, my_model)
my_model_preprocessed_data    = path.join(final_project_root, "data", my_model + "_preprocessed")
kc_suspects_and_predictions_path = path.join(final_project_root, "data", "kc_suspects_with_predctions")

train_txt_basename        = "train.txt"
# train_txt                 = path.join(my_model_data, train_txt_basename)
val_txt_basename          = "val.txt"
# val_txt                   = path.join(my_model_data, val_txt_basename)

train_lmdb_basename          = "train_lmdb"
# train_lmdb                   = path.join(my_model_data, train_lmdb_basename)
val_lmdb_basename            = "val_lmdb"
# val_lmdb                     = path.join(my_model_data, val_lmdb_basename)

full_size_images_path = path.join(final_project_root, "data/full_size_images")
new_data_in_pdf = path.join(final_project_root, "data/new_data_in_pdf")
data_copy_for_changes = path.join(final_project_root, "data/data_copy_for_changes")
augmented_data_path  = path.join(final_project_root, "data/kc_brightness_contrast_augmented")
augment_cly_path = path.join(final_project_root, "data/augment_cly")
augment_every_image_twice_path = path.join(final_project_root, "data/augment_every_image_twice")

caffenet_weights = path.join(my_model_path, model + ".caffemodel")
caffenet_deploy_prototxt = path.join(my_model_path, model + "_deploy.prototxt")

# ---------------- DATA MEAN BINARY PROTO ---------------------
my_model_mean_binaryproto_basename  = my_model + "_mean.binaryproto"
# my_model_mean_binaryproto           = path.join(my_model_data, my_model_mean_binaryproto_basename)

# ----------------------- MODES -------------------
healthy_kc_mode                             = "healthy_kc"
kc_sus_mode                                 = "kc_sus"
healthy_sus_mode                            = "healthy_sus"
healthy_kc_sus_2_classes_sus_as_kc_mode     = "healthy_kc_sus_2_classes_sus_as_kc"
healthy_kc_sus_3_classes_mode               = "healthy_kc_sus_3_classes"
kc_cly                                      = "kc_cly"
modes = [healthy_kc_mode, kc_sus_mode, healthy_sus_mode, healthy_kc_sus_2_classes_sus_as_kc_mode, healthy_kc_sus_3_classes_mode, kc_cly]

modes_path = path.join(my_model_path, "modes")

demo_modes_path = path.join(modes_path, "demo_modes")

# ---------------------- more stuff -------------------------
kc_synset_words = path.join(my_model_path,  "kc_synset_words.txt")

my_model_reports_path = path.join(my_model_path, "reports")

final_project_scripts = final_project_root + "/scripts"

saved_logs_path = path.join(final_project_root, "saved_logs")

# --- demo ---
misclassified_images_file = path.join(demo_modes_path, "misclassified.txt")

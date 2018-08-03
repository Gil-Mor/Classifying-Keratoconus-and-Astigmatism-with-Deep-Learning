# import tempfile

# from scipy.stats._discrete_distns import logser_gen

import export_env_variables
from export_env_variables import *
# from write_prototxts import *
# from visualizations import *
from preprocessing import *


from defs import *
import os
from os import path
from os.path import join as pathjoin
import sys
import plot_learning_curve
import subprocess

import numpy as np
from PIL import Image
import PIL
import random
import shutil
from time import time
import cv2
import signal
from datetime import datetime
import solve
from sys import exit
import webbrowser # for import webbrowser.open(img) which displays the image in a viewer with the image filename as the title.
import re
from skimage import io, img_as_float
from natsort import natsorted, ns # natural sort alphanumerical strings with natsort(list)
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import fnmatch



def clean_mode(mode):
    rmdirs = []
    for root, dirnames, filenames in os.walk(mode.mode_path):
        if root != mode.mode_path:
            rmdirs.append(root)
    
    for dir in rmdirs:
        try:
            shutil.rmtree(dir)
        except:
            pass
    
# ---------------------------------------------------------------------

def get_imagename__true_label__pred_label__and_max_prob_from_line(line):
    
    stripped_line = line.strip().replace("\t", " ") # save original line for writing to a file.
    stripped_line = " ".join(stripped_line.split())
    linesplit = stripped_line.split()
        
    img_filename_col = 0
    true_label_col = 1
    predicted_label_col = 2
    probs_cols = 4
    
    
    image_name      = linesplit[img_filename_col]
    true_label      = int(linesplit[true_label_col])
    predicted_label      = int(linesplit[predicted_label_col])
    probs = [float(x) for x in re.findall(r"\d\.\d+", str(linesplit[probs_cols:]))]
    max_prob = max(probs)
    
    return image_name, true_label, predicted_label, max_prob
# -------------------------------------------------------------------------------------------------------
    
def get_train_epoch(iter, train_batch_size, train_set_size):
    return int(math.ceil(float(iter) / (float(train_set_size) / float(train_batch_size))))
# ---------------------------------------------------------------------

def prepare_globals_in_dir(mode):
    """
    Read the data_dir and prepare info about the data. Not in use so much anymore since now I just read the data with glob.
    :param mode: Use the mode.data_dir
    :return:
    """
    global healthy_imgs, kc_imgs, cly_imgs, sus_imgs, NUM_OF_HEALTHY, NUM_OF_KC, NUM_OF_SUS, NUM_OF_CLY, NUM_OF_TOTAL_IMAGES, \
        HEALTHY_TRAIN_SIZE_DEFAULT, KC_TRAIN_SIZE_DEFAULT, SUS_TRAIN_SIZE_DEFAULT, CLY_TRAIN_SIZE_DEFAULT

    healthy_imgs = natsorted([os.path.basename(x) for x in glob.glob(mode.data_dir + "/healthy_*.jpg")])
    kc_imgs = natsorted([os.path.basename(x) for x in glob.glob(mode.data_dir + "/kc_*.jpg")])
    cly_imgs = natsorted([os.path.basename(x) for x in glob.glob(mode.data_dir + "/cly_*.jpg")])
    sus_imgs = natsorted([os.path.basename(x) for x in glob.glob(mode.data_dir + "/sus_*.jpg")])



    all_images = natsorted([os.path.basename(x) for x in glob.glob(mode.data_dir + "/*.jpg")])

    NUM_OF_HEALTHY = len(healthy_imgs)
    NUM_OF_KC = len(kc_imgs)
    NUM_OF_CLY = len(cly_imgs)
    NUM_OF_SUS = len(sus_imgs)
    NUM_OF_TOTAL_IMAGES = len(all_images)

    HEALTHY_TRAIN_SIZE_DEFAULT = int(NUM_OF_HEALTHY * DEFAULT_TRAIN_SIZE_FRACTION)
    KC_TRAIN_SIZE_DEFAULT = int(NUM_OF_KC * DEFAULT_TRAIN_SIZE_FRACTION)
    SUS_TRAIN_SIZE_DEFAULT = int(NUM_OF_SUS * DEFAULT_SUS_TRAIN_SIZE_FRACTION)
    CLY_TRAIN_SIZE_DEFAULT = int(NUM_OF_CLY * DEFAULT_TRAIN_SIZE_FRACTION)

    print(
    "num of healthy {}\nnum of kc {}\nnum of sus {}\nnum of cly {}".format(NUM_OF_HEALTHY, NUM_OF_KC, NUM_OF_SUS,
                                                                           NUM_OF_CLY))

    if len(healthy_imgs + kc_imgs + cly_imgs + sus_imgs) != len(all_images):
        print("ERROR")
        sys.exit(0)

# ---------------------------------------------------------------------

def get_shuffled_images_and_labels(data_dir, real_world_target_names_and_labels):

    imgs_and_labels = []
    for target_name, label in real_world_target_names_and_labels.items():
        imgs_and_labels += natsorted((img, label) for img in [os.path.basename(x) for x in glob.glob(data_dir + "/" + target_name +"_*.jpg")])

    random.shuffle(imgs_and_labels)

    return imgs_and_labels
# -------------------------------------------------------------------------------------------------------

def write_train_val_txts(mode):
    """
    Write train.txt and val.txt with Txts_data class instance. Gives more control over the process 
    but not comfortable to use with cross validation.
    """
    if path.exists(mode.train_txt) or path.exists(mode.val_txt):
        raw_input("You're about to overwrite txts in {}.\n".format(mode.name) +
                  "There's no coming back from this in terms of re testing mode on those specific train/val again.")

    try:
        os.remove(path.join(mode.data_path, train_txt_basename))
        os.remove(path.join(mode.data_path, val_txt_basename))
    except OSError:
        pass

    train_f = open(path.join(mode.data_path, train_txt_basename), "w")
    val_f = open(path.join(mode.data_path, val_txt_basename), "w")

    prepare_globals_in_dir(mode)
    train_images = []
    test_images = []


    # ----------- training -------------
    if mode.txts_data.train_healthy:
        if mode.txts_data.shuffle_healthy:
            random.shuffle(healthy_imgs)
        train_healthy_imgs = healthy_imgs[: mode.txts_data.train_healthy]
        train_images += (write_to_file_from_list(train_f, train_healthy_imgs, healthy_label))
    else:
        train_healthy_imgs = []

    if mode.txts_data.train_kc:
        if mode.txts_data.shuffle_kc:
            random.shuffle(kc_imgs)
        train_kc_imgs = kc_imgs[: mode.txts_data.train_kc]
        train_images += (write_to_file_from_list(train_f, train_kc_imgs, kc_label))
    else:
        train_kc_imgs = []

    if mode.txts_data.train_sus:
        if mode.txts_data.shuffle_sus:
            random.shuffle(sus_imgs)
        train_sus_imgs = sus_imgs[: mode.txts_data.train_sus]
        train_images += (write_to_file_from_list(train_f, train_sus_imgs, mode.txts_data.sus_label))
    else:
        train_sus_imgs = []

    if mode.txts_data.train_cly:
        if mode.txts_data.shuffle_cly:
            random.shuffle(cly_imgs)
        train_cly_imgs = cly_imgs[: mode.txts_data.train_cly]
        train_images += (write_to_file_from_list(train_f, train_cly_imgs, mode.txts_data.cly_label))
    else:
        train_cly_imgs = []

    # -------------- test ----------

    if mode.txts_data.val_healthy:
        val_healthy_imgs = list(set(healthy_imgs) - set(train_healthy_imgs))[: mode.txts_data.val_healthy]
        test_images += (write_to_file_from_list(val_f, val_healthy_imgs, healthy_label))

    if mode.txts_data.val_kc:
        val_kc_imgs = list(set(kc_imgs) - set(train_kc_imgs))[: mode.txts_data.val_kc]
        test_images += (write_to_file_from_list(val_f, val_kc_imgs, kc_label))

    if mode.txts_data.val_sus:
        val_sus_imgs = list(set(sus_imgs) - set(train_sus_imgs))[: mode.txts_data.val_sus]
        test_images += (write_to_file_from_list(val_f, val_sus_imgs, mode.txts_data.sus_label))

    if mode.txts_data.val_cly:
        val_cly_imgs = list(set(cly_imgs) - set(train_cly_imgs))[: mode.txts_data.val_cly]
        test_images += (write_to_file_from_list(val_f, val_cly_imgs, mode.txts_data.cly_label))

    train_f.close()
    val_f.close()

    return train_images, test_images


# -----------------------------------------------------------------------


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                return filename
# ------------------------------------------------------------------------------


def makedirs_ok(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
# ------------------------------------------------------

def copy_all_files(src_dir, dst_dir, pattern):
    files = glob.glob(src_dir + "/" + pattern)
    for f in files:
        shutil.copy2(f, dst_dir)
# --------------------------------------------

# -- signal handler
def cleanup(sig, frame):

    print("Do nothing in signal handler..")
    #
    #
    #
    # print('Received signal {signal}'.format(signal=sig))
    #
    # if mode.state == "pycaffe":
    #     try:
    #         shutil.move(last_pycharm_log, mode.get_next_log().replace(".log", "_pycharm_aborted.log"))
    #     except:
    #         print("!!!! No pycharm log in pycharm logs")
    #
    #     try:
    #         if g_log is not None:
    #             g_log.close()
    #     except:
    #         pass
    #
    # else:
    #     try:
    #         shutil.move(mode.log, mode.log.replace(".log", "_aborted.log"))
    #     except:
    #         print("!!!! No log in logs path")
    #
    # try:
    #     os.remove(last_pycharm_log)
    # except:
    #     pass

    sys.exit(0)
# -------------------------------------------------------------------------------------------------------


def center_crop_image(image, new_width, new_height):
    """
    :param image: type: ndarray
    :param new_width:
    :param new_height:
    :return:
    """
    was_transposed = False
    if image.shape[0] <= 4:
        # need to transpose it..
        image = image.transpose(1, 2, 0) # (chan, height, width) -> (height, width, chan)
        was_transposed = True

    height, width, chan = image.shape

    width_cut = (width - new_width) // 2
    height_cut = (height - new_height) // 2

    top, bottom = height_cut, -height_cut
    left, right = width_cut, -width_cut

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

def mean_binary_proto_to_np_array(caffe, mean_binproto):
    """

    :param caffe: caffe instances from import_caffe() method
    :param mean_binproto: full path to the mode's image-mean .binaryproto created from train.lmdb.
    :return:
    """
    # I don't have my image mean in .npy file but in binaryproto. I'm converting it to a numpy array.
    # Took me some time to figure this out.
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_binproto, 'rb').read()
    blob.ParseFromString(data)
    mu = np.array(caffe.io.blobproto_to_array(blob))
    mu = mu.squeeze()  # The output array had one redundant dimension.
    return mu
# -------------------------------------------------------------------------------------------------------

def get_caffenet_transformer(caffe, net, mean_binproto):
    """
    Get caffe.transformer for preprocessing an image before real world prediction.
    :param caffe: caffe instance
    :param net: caffe.Net - usually created from deploy.prototxt for prediction
    :param mean_binproto: full path to the mode's image-mean .binaryproto created from train.lmdb.
    :return:
    """
    mu = mean_binary_proto_to_np_array(caffe, mean_binproto)

    # if mean is 256x256 (train size) but need it in test size - crop it
    if mu.shape[-1] != net.blobs['data'].data.shape[-1]:
        mu = center_crop_image(mu, net.blobs['data'].data.shape[-1], net.blobs['data'].data.shape[-1])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255] - needed for caffenet\Alexnet.
    # transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    return transformer
# -------------------------------------------------------------------------------------------------------


def write_to_file_from_list(f, images, label):
    res = []
    for img in images:
        f.write(img + " " + str(label) + "\n")
        res.append(img + " " + str(label))
    return res
# -----------------------------------------------------------------------

def write_to_file_from_list_of_img_label_tuples(f, img_label_tups):
    for img, label in img_label_tups:
        f.write(img + " " + str(label) + "\n")
# -----------------------------------------------------------------------


def cleanup_from_previous(dir):
    try:
        os.remove(path.join(dir, "train.txt"))
        os.remove(path.join(dir, "val.txt"))
        os.remove(path.join(dir, "train_lmdb"))
        os.remove(path.join(dir, "val_lmdb"))
    except OSError:
        pass


# -----------------------------------------------------------------------

def open_files(data_dir):
    train_f = open(path.join(data_dir, "train.txt"), "w")
    val_f = open(path.join(data_dir, "val.txt"), "w")

    return train_f, val_f


# ---------------------------------------------------------------------


# ============================ FUNCTIONS ===============================

def import_caffe():
    sys.path.insert(0, caffe_root + '/python')
    global caffe
    import caffe

    if PLATFORM == EC2_GPU_Platform:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        # * Set Caffe to CPU mode and load the net from disk.
        caffe.set_mode_cpu()
    return caffe
# ---------------------------------------------------------------------

def crop_to(n):
    cmd = r"mogrify -gravity Center -extent " + str(n)  + "x" + str(n)  + " " + my_model_data + "/*.jpg"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
# ---------------------------------------------------------------------------------------

def resize_to(path, n):
    print("REMEMBER THAT RESIZING DIDN'T WORK!!!")
    cmd = r"mogrify -resize " + str(n) + "x" + str(n) + " " + path + "/*.jpg"
    print(cmd)
    subprocess.call(cmd, shell=True)

# ---------------------------------------------------------------------------------------

def make_lmdb(data_dir, resize, mode_data_info_path_to_copy_to):
    """
    Create train and val lmdbs from train and val txts
    :param data_dir: folder to take images from.
    :param resize: Size to resize the images to.
    :param mode_data_info_path_to_copy_to: where to output.
    :return:
    """
    assert os.path.exists(mode_data_info_path_to_copy_to + "/" + train_txt_basename), "no train txt in mode"
    assert os.path.exists(mode_data_info_path_to_copy_to + "/" + val_txt_basename), "no val txt in mode"
    try:
        shutil.rmtree(path.join(mode_data_info_path_to_copy_to, train_lmdb_basename))
        shutil.rmtree(path.join(mode_data_info_path_to_copy_to, val_lmdb_basename))
    except:
        pass
    # curr_dir = os.getcwd()
    # os.chdir(data_dir)

    subprocess.call(r"rm -r -f train_lmdb", shell=True)
    subprocess.call(r"rm -r -f val_lmdb", shell=True)


    cmd = r"GLOG_logtostderr=1 " + caffe_tools + \
          "/convert_imageset --resize_height={size} --resize_width={size} --shuffle {data_dir}/ {dst_dir}/{type}.txt {dst_dir}/{type}_lmdb".format(
              data_dir=data_dir, dst_dir=mode_data_info_path_to_copy_to, type="train", size=resize)

    subprocess.call(cmd, shell=True)

    cmd = r"GLOG_logtostderr=1 " + caffe_tools + \
          "/convert_imageset --resize_height={size} --resize_width={size} --shuffle {data_dir}/ {dst_dir}/{type}.txt {dst_dir}/{type}_lmdb".format(
              data_dir=data_dir, dst_dir=mode_data_info_path_to_copy_to, type="val", size=resize)

    subprocess.call(cmd, shell=True)

    # os.chdir(curr_dir)

# ---------------------------------------------------------------------------------------

def change_mode_name():
    """
    If for some reason you decided to change the format of modes names.
    Be careful..
    """
    for i in range(6):
        mode = Mode([healthy_kc_mode, "cross_validation_5", "set_"+str(i)],
                    solver_net_parameters=solver_net_parameters,
                    txts_data=dummy_txts_data_for_num_of_classes,
                    dummy=True,
                    data_dir=my_model_data)

        for root, dirs, files in os.walk(mode.mode_path):
            for filename in files:
                if "cross_validation_5_".format(i) in filename:
                    shutil.move(pathjoin(root, filename), pathjoin(root,
                                            filename.replace("cross_validation_5_", "cross_validation_5_set_"+str(i))))
        continue
        for root, dirs, files in os.walk(mode.mode_path):
            for filename in files:
                if filename.endswith(".prototxt") or filename.endswith(".txt") or filename.endswith(".log"):
                    change = False
                    with open(pathjoin(root, filename)) as f:
                        fstr = f.read()
                        if "cross_validation_5_" in fstr:
                            print(filename)
                            change = True

                    if change:
                        with open(pathjoin(root, filename), "w") as f:
                            f.write(fstr.replace("cross_validation_5_", "cross_validation_5_set_{}_".format(i)))
# -------------------------------------------------------------------------------------------------------


def make_image_mean_binaryproto(data_dir, mode_data_info_path_to_copy_to):
    """
    Create image-mean binaryproto from train.lmdb
    """
    assert os.path.exists(mode_data_info_path_to_copy_to + "/" + train_lmdb_basename), "no train lmdb in mode"
    assert os.path.exists(mode_data_info_path_to_copy_to + "/" + val_lmdb_basename), "no val lmdb in mode"
    try:
        os.remove(path.join(mode_data_info_path_to_copy_to, my_model_mean_binaryproto_basename))
    except:
        pass
    subprocess.call(r"rm -r -f " + data_dir + "/" + my_model_mean_binaryproto_basename, shell=True)
    cmd = caffe_tools + "/compute_image_mean -backend=lmdb " + mode_data_info_path_to_copy_to + "/train_lmdb " + mode_data_info_path_to_copy_to + "/" + my_model_mean_binaryproto_basename
    subprocess.call(cmd, shell=True)


    # shutil.move(data_dir + "/" + my_model_mean_binaryproto_basename, mode_data_info_path_to_copy_to)


# ---------------------------------------------------------------------------------------


def confusion_matrix_to_str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    s = ""
    s += "    " + empty_cell
    for label in labels + ["Total"]:
        s += "%{0}s".format(columnwidth) % label
    s += "\n"
    # Print rows

    Total = 0
    for i, label1 in enumerate(labels):
        s += "    %{0}s".format(columnwidth) % label1

        instances_of_class = 0
        for j in range(len(labels)):
            instances_of_class += int(cm[i, j])
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell

            s += cell

        s += "%{0}d".format(columnwidth) % instances_of_class

        Total += instances_of_class

        s += "\n"

    s += "    " + empty_cell*(len(labels)+1) + "%{0}d".format(columnwidth) % Total
    return s
# -------------------------------------------------------------------------------------------------------

def get_confusion_matrix_and_report_str(true_labels, predicted_labels, target_names_and_labels_for_presentation):

    target_names_and_labels_for_presentation_copy = target_names_and_labels_for_presentation[:]
    # Filter out labels and target names that are not in true_labels or in predicted labels because scikit classification report
    # doesn't know how to handle it.
    labels_enums = [tup[1] for tup in target_names_and_labels_for_presentation_copy]

    labels_not_in_validation = set(labels_enums) - set(np.unique((true_labels, predicted_labels)))

    for label in labels_not_in_validation:
        index_to_remove = labels_enums.index(label)
        del target_names_and_labels_for_presentation_copy[index_to_remove]

    labels_strs = [tup[0] for tup in target_names_and_labels_for_presentation_copy]
    labels_enums = [tup[1] for tup in target_names_and_labels_for_presentation_copy]
    
    s = "Confusion matrix:\n"
    try:
        s += confusion_matrix_to_str(confusion_matrix(true_labels, predicted_labels), labels=labels_strs) + "\n\n"
    except:
        print("Error in confusion matrix")
    s += "\n\nClassification Report:\n"

    try:
        s += classification_report(true_labels, predicted_labels, labels=labels_enums, target_names=labels_strs) + "\n"
    except:
        print("Error in classification report")
    return s
# -------------------------------------------------------------------------------------------------------

def list_images_for_probability_assignment():
    """
    Not important
    :return:
    """
    imgs_files = glob.glob(data_copy_for_changes + "/kc_*.jpg")
    imgs_files.extend(glob.glob(data_copy_for_changes + "/healthy_*.jpg"))
    imgs_files.extend(glob.glob(data_copy_for_changes + "/sus_*.jpg"))
    # imgs_files = natural_sort(imgs_files)
    random.shuffle(imgs_files)

    names_to_anonymous_indexes = open(data_copy_for_changes + "/names_to_anonymous_indexes_for_manual_probability_assignment.txt", "w")
    anonymous_names_f = open(data_copy_for_changes + "/anonymous_names_for_manual_probability_assignment.txt", "w")

    anonymous_names_f.write("{:<10} {:}\n".format("image", "Disease-Probability"))


    for i, img_f in enumerate(imgs_files):
        names_to_anonymous_indexes.write(path.basename(img_f) + " -> " + str(i) + "\n")

        shutil.move(img_f, path.dirname(img_f) + "/" + str(i) + ".jpg")
        anonymous_names_f.write(str(i) + "\n")


    anonymous_names_f.close()
    names_to_anonymous_indexes.close()

    # imgs_files = natural_sort(glob.glob(my_model_data + "/*.jpg"))
    # data_files_names_f = open(my_model_data + "/images_names_for_manual_probability_assignment.txt", "w")
    # data_files_names_f.write("{:<10} {:}\n".format("image", "KC probability"))
    #
    # print(imgs_files)
    # for img_f in imgs_files:
    #     data_files_names_f.write("{}\n".format(os.path.basename(img_f).replace(".jpg", "")))
    # data_files_names_f.close()
# -------------------------------------------------------------------------------------------------------


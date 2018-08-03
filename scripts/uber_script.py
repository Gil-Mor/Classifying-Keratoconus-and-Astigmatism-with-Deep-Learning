import tempfile

from astropy._erfa.core import num00a
from scipy.stats._discrete_distns import logser_gen

import make_train_val_txt as txts
import export_env_variables
from export_env_variables import *
from write_prototxts import *
from visualizations import *
from preprocessing import *

import save_logs
import defs
from defs import *
import utils
from utils import *
import demo_modes
from demo_modes import *
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
from sklearn.metrics import precision_recall_fscore_support
import copy
import shutil



# =========================== GLOBALS ==============================
# mode = None
# g_log = None # maybe manually writing during solve is better then saving the log from the console..
             # global for closing in signal handler.
SIMPLE_TRAIN_PREDICT=0
CROSS_VALIDATION=1
INCREASING_TRAIN_SET_SIZE=2

# ===================== UBER FUNCTIONS ===========================

# ---------------------------------------------------------------------------------------

def run_transfer_learning_caffe_cmd(mode, from_scratch, snapshot_or_weights, weights_or_solverstate):
    """
    Activatate training via cmdline.
    :param snapshot_or_weights: string 'snapshot' or 'weights'. 
    - snapshot means we're resuming the training of a net **WE** already started to train. 
      if 'snapshot' is chosen then weights_or_solverstate has to be a full path of solverstate file (saved with snapshots. ends with .solverstate).
    - weights means we're resuming the training of caffenet. 
      if 'weights' is chosen then weights_or_solverstate has to be a full path of caffenet weights.
    """
    if from_scratch:
        print("Training From Scratch")
        transfer_learning_cmd = caffe_tools + "/caffe" + " train --solver=" + mode.solver_prototxt + " 2>&1 | tee " + mode.log
    else:
        print("snapshot or weights: ", snapshot_or_weights)
        print("Using weights/solverstate: ", weights_or_solverstate)
        transfer_learning_cmd = caffe_tools + "/caffe" + " train --solver=" + mode.solver_prototxt + " --" + snapshot_or_weights + " " + weights_or_solverstate + " 2>&1 | tee " + mode.log

    subprocess.call(transfer_learning_cmd, shell=True)

# ---------------------------------------------------------------------------------------

def solve(mode):
    """
    Manually train the net (not via cmdline. Not so much in use.
    Call external solve if you want the caffe log to be saved to a file.
    """

    iterations=solve_parameters.max_iter,
    solver_prototxt=mode.solver_prototxt,
    weights=mode.weights,
    display_iter=solve_parameters.display_iter,
    test_interval=solve_parameters.test_interval,
    val_txt = mode.val_txt,
    mean_binaryproto=mode.mean_binaryproto,
    data_dir=mode.data_dir,
    log=mode.log
    
    
    sys.path.append(pycaffe_module_path)
    caffe = import_caffe()

    solver = caffe.get_solver(olver_prototxt)
    if solverstate is not None:
        solver.restore(solverstate)

    print("using weights ", os.path.basename(weights))
    solver.net.copy_from(weights)
    print("starting from iteration ", solver.iter)

    train_loss, val_loss, acc = [], [], []


    for _ in range(iterations):

        if solver.iter % display_iter == 0:
            train_loss.append(solver.net.blobs['loss'].data.copy())



        image, label = get_image_n_label_from_blob(solver.net, 0)
        print("image from data\n",image)
        print(label)

        if solver.iter % test_iter == 0:
            # solver.test_nets[0].forward()
            # solver.net.forward()
            # out = solver.test_nets[0].forward()
            # print(out)
            # print(out['prob'])

            val_labels = list(solver.test_nets[0].blobs['label'].data.copy().astype(np.int))
            val_propabilities = solver.test_nets[0].blobs['prob'].data.copy()
            predicted = [tup.argmax() for tup in val_propabilities]
            print("labels      ", val_labels)
            print("predictions ", predicted)

            val_loss.append(solver.test_nets[0].blobs['loss'].data.copy())
            acc.append(solver.test_nets[0].blobs['accuracy'].data.copy())
            #
            # pred_label = np.array(solver.test_nets[0].blobs['loss'], dtype=np.int32)[0]
            #
            # image, label = get_image_n_label_from_blob(solver.test_nets[0], 0)

            # predict_for_one_image_using_test_net(caffe, solver.test_nets[0], image, label, ['healthy', 'kc'], num_of_classes=2)



            ### visualize

            # filters = solver.net.params['conv1'][0].data
            # show_blobs(filters.transpose(0, 2, 3, 1))
            # feat = solver.net.blobs['conv2'].data[0, :36]
            # feat = solver.net.blobs['data'].data[0]
            # show_blobs(feat)
            # check_for_overfitting(loss, acc)

        # step here because we want to test in iteration 0 as well!!
        solver.step(1)  # run a single SGD step in Caffe


# -------------------------------------------------------------------------------------------------------



def prediction_from_txt_with_deploy(mode,
                                    weights,
                                    iter,
                                    imgs_txt_file,
                                    net=None,
                                    display_images=False,
                                    target_names_and_labels_for_presentation=None,
                                    out_path=None,
                                    out_file_name_addition="",
                                    called_via_call_predict=False):
    """
    Use saved weights to predict all images in imgs_txt_file (usually train.txt or val.txt).

    :param file: full path to a file with img file names

    :type mode: Mode

    :param model: path to deploy.prototxt. comes with caffenet.
                  You need to change the name of the output layer to the same name you gave your output layer in the train_val.prototxt
                  This is how you make it take the weights of the trained model.
                  and also change the num_output to 2.

    :param weights: path to the binaryproto. snapshot of the fine-tuned model. define how to save it in your solver.prototxt.

    :param out_file_name_addition: in case all classification files with the same name are going to the same output folder.

    """

    if PLATFORM == PC_Platform:
        print("Why are you caliing predict? don't mess gpu prediction logs")
        return


    if not called_via_call_predict:
        print("Call via call predict")
        return

    if out_path is None:
        out_path = mode.mode_logs_path



    # * Load `caffe`.
    # The caffe module needs to be on the Python path;
    #  we'll add it here explicitly.

    caffe = import_caffe()

    net = caffe.Net(mode.deploy_prototxt,  # defines the structure of the model
                    caffe.TEST,
                    weights=weights
                     )  # use test mode (e.g., don't perform dropout)

    # * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this, but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
    #
    #     Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected as the first (_outermost_) dimension.
    #
    #     As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the _innermost_ dimension, we are arranging for the needed transformations here.



    # create transformer for the input called 'data'
    transformer = get_caffenet_transformer(caffe, net, mode.mean_binaryproto)


    imgs_set = os.path.basename(imgs_txt_file).replace(".txt", "")


    with open(imgs_txt_file) as f:
        val_images = f.readlines()

    # You don't need to shuffle.. This is how I want to see the output.
    # random.shuffle(val_images)

    # I'm saving mis-classified imgs_txt_filenames in a imgs_txt_file.
    # misclassified = open(val_txt_folder + "/{}_misclassified.txt".format(imgs_set), "w")
    if out_file_name_addition:
        out_file_name_addition = "_" + out_file_name_addition

    classification_file = open(get_classification_path_from_imgs_txt_file(out_path, imgs_txt_file, str(iter) + out_file_name_addition), "w")

    classification_file.write("{:<20} {:<10} {:<10} {:<10} {:}\n".format("Image", "Label", "Predicted", "Status", "Probs"))

    imgs_and_labels = set()
    for image_name_n_label in val_images:
        if len(image_name_n_label.split(' ')) != 2:
            continue

        #  ALWAYS PREDICT ON ORIGINAL DATA
        image_basename, label = image_name_n_label.split(' ')[0], int(image_name_n_label.split(' ')[1])
        if image_basename.count("_") > 1: # It's a preprocessed file.

            image_basename = "_".join(image_basename.split("_")[:2]) + ".jpg"

        imgs_and_labels.add((image_basename, label))

    imgs_and_labels = natsorted(imgs_and_labels, key=lambda tup:tup[0]) # sort by img name

    true_labels = []
    predicted_labels = []

    correct = 0
    count = 1
    
    misclassified_images_s = ""
        
    for image_basename, label in imgs_and_labels:

        image_file = os.path.join(my_model_data, image_basename) # ALWAYS PREDICT ON ORIGINAL IMAGES
    
        true_labels.append(label)
        if iter > 0 and imgs_set != "train":
            print(str(count) +  ". image: " + os.path.basename(image_file) + " label: " + str(label))
        image = caffe.io.load_image(image_file)

        # image shape is (3, 256, 256). we want it (3, 227, 227) for caffenet.
        # asking about shape[0] and shape[1] because I can't know if the image is (chan, h, w) or (h, w, chan)
        if image.shape[0] == TRAINING_IMAGE_SIZE or image.shape[1] == TRAINING_IMAGE_SIZE or image.shape[2] == TRAINING_IMAGE_SIZE:
            # I'm cropping the numpy array on the fly so that I don't have to mess with resizing
            # the actual images in a separate folder each time.
            image = center_crop_image(image, CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE)


        try:
            transformed_image = transformer.preprocess('data', image)
        except:
            # try to transpose and again
            image = image.transpose(2,0,1) # (height, width, chan) -> (chan, height, width)
            transformed_image = transformer.preprocess('data', image)


        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward(start='conv1')

        # Extract convolutionzed image of kc_6.jpg - classic example of KC.
        '''
        try:
            if "kc_6.jpg" in image_basename:

                makedirs_ok(mode.data_path + "/kc6_vizualizations")
                for conv in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                    makedirs_ok(mode.data_path + "/vizualizations")
                    feat = net.blobs[conv].data[0, :36]
                    vis_square(feat, filename=mode.data_path + "/kc6_vizualizations/" + image_basename.replace(".jpg","") + "_" + conv + "_" + str(iter) + ".png")
        except Exception,e:
            print(str(e))
            sys.exit(0)
        '''    
        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        max_prob = max(output_prob)
        predicted_label = output_prob.argmax()
        predicted_labels.append(predicted_label)

        s = "{:<20} {:<10} {:<10} {:} ".format(os.path.basename(image_file) , label, predicted_label, str(label == predicted_label))
        s += " " * (60 - len(s))
        classification_file.write(s)
        for i, prob in enumerate(output_prob):
            classification_file.write(str(i) + ": " + "{:.5f}   ".format(output_prob[i]))
        classification_file.write("\n")

        if predicted_label == label:
            correct += 1
        elif iter > 0 and imgs_set != "train":
            print("!!!!!!!!!!!!!!! misclassified !!!!!!!!!!!!!!! ")
            # misclassified.write(os.path.basename(image_file) + " " + str(label) + " " + str(predicted_label) + "\n")
            # display misclassified images.
            if display_images and count - correct < 3:
                webbrowser.open(image_file)
                # image = PIL.Image.open(image_file)
                # image.show()



        accuracy = ((100. * correct) / (count))

        if iter > 0 and imgs_set != "train":
            print(str(count) + '. predicted class is: ' + str(output_prob.argmax()))
            print(str(count) + ". accuracy: " + str(accuracy))
            print("")

        count += 1


    classification_file.write("\n" + get_confusion_matrix_and_report_str(true_labels, predicted_labels, target_names_and_labels_for_presentation) + "\n")

    

    classification_file.close()
    


    return true_labels, predicted_labels

# -------------------------------------------------------------------------------------------------------


def visualize_net_layers(model_path, weights_path, snapshot_iter, out_dir_path):
    """
    :param: model_path is the full path of a mode.deploy_prototxt
    :param: weights_path is the full path of caffenet weights or some snapshot
    :type mode: Mode
    """
    makedirs_ok(out_dir_path)
    caffe = import_caffe()

    net = caffe.Net(model_path,  # defines the structure of the model (deploy....)
                    caffe.TEST,
                    weights=weights_path
                    )



    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1), filename=path.join(out_dir_path, 'conv1_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat, padval=1, filename=path.join(out_dir_path, 'conv1_squares_closeup_{}.png'.format(snapshot_iter)))
    
            #plt.show()
    
    filters = net.params['conv2'][0].data
    vis_square(filters[:48].reshape(48**2, 5, 5), filename=path.join(out_dir_path, 'conv2_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['conv2'].data[0, 100:136]
    vis_square(feat, padval=1, filename=path.join(out_dir_path, 'conv2_squares_closeup_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['conv3'].data[0]
    vis_square(feat, padval=0.5, filename=path.join(out_dir_path, 'conv3_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['conv4'].data[0]
    vis_square(feat, padval=0.5, filename=path.join(out_dir_path, 'conv4_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['conv5'].data[0]
    vis_square(feat, padval=0.5, filename=path.join(out_dir_path, 'conv5_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    feat = net.blobs['pool5'].data[0]
    vis_square(feat, padval=1, filename=path.join(out_dir_path, 'pool5_squares_{}.png'.format(snapshot_iter)))
    
    #plt.show()
    
    plt.figure(figsize=(10,10))
    plt.axis('off')
    
    feat = net.blobs['fc6'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)

    plt.savefig(path.join(out_dir_path, 'fc6_{}.png'.format(snapshot_iter)), bbox_inches='tight', pad_inches=0)
    plt.close()
    #plt.show()
    
    plt.figure(figsize=(10,10))
    plt.axis('off')

    feat = net.blobs['fc7'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)


    plt.savefig(path.join(out_dir_path, 'fc7_{}.png'.format(snapshot_iter)), bbox_inches='tight', pad_inches=0)
    plt.close()

# -------------------------------------------------------------------------------------------------------

def visualize_layers_of_all_snapshots(mode):
    """
    Overkill. They all look the same. one is enough.
    """
    makedirs_ok(mode.visualize_layers_path)
    for iter in mode.get_existing_snapshots_iters()[4:5]: # take the 4th snapshot
        weights = mode.get_snapshot(iter)
        visualize_net_layers(mode.deploy_prototxt, weights, iter, mode.visualize_layers_path)


# -------------------------------------------------------------------------------------------------------

def call_external_solve(mode, solve_parameters):
    """
    Call manual solve via cmd line so that we can redirect the GLOG to a file as usual.
    If we don't do this the log won't be saved to a file.
    """
    import_caffe()

    cmd = "~/anaconda2/bin/python solve.py {iterations} {solver_prototxt} {weights} {display_iter} {test_interval} {val_txt} {mean_binaryproto} {data_dir} 2>&1 | tee {log}".format(
        iterations=solve_parameters.max_iter,
        solver_prototxt=mode.solver_prototxt,
        weights=mode.weights,
        display_iter=solve_parameters.display_iter,
        test_interval=solve_parameters.test_interval,
        val_txt = mode.val_txt,
        mean_binaryproto=mode.mean_binaryproto,
        data_dir=mode.data_dir,
        log=mode.log)

    subprocess.call(cmd, shell=True)

# -------------------------------------------------------------------------------------------------------

def average_all_classifications(classification_files_root_path, classification_filename_pattern, display=False, target_names_and_labels_for_presentation=None, outpath=None,
                                predicted_probabilities_of_misclassified=None,
                                write_output_file=True):
    """
    Collects all misclassified images from specific classifications iterations in a specific mode.
    :param classification_filename_pattern: substring - not regex. used for out file name pattern - should
                                            be composed with get_classification_basename_from_imgs_txt_file(imgs_txt_file, snapshot_iter)

    :param write_output_file: False in case we just want predictions.
    """

    if outpath is None:
        outpath = classification_files_root_path

    processed_images_lines = {}

    img_filename_col = None
    label_col = None
    predicted_label = None
    probs_cols = None

    for root, dirnames, filenames in os.walk(classification_files_root_path):
        for filename in filenames:

            if filename.endswith(".log") and classification_filename_pattern in filename:

                with open(root + "/" + filename) as f:

                    for i, line in enumerate(f.readlines()):
                        if line == "":
                            continue

                        stripped_line = line.strip().replace("\t", " ") # save original line for writing to a file.
                        stripped_line = " ".join(stripped_line.split())
                        linesplit = stripped_line.split()

                        if i == 0: # get info columns
                            if "Image" not in line:
                                break
                            img_filename_col = linesplit.index("Image")
                            label_col = linesplit.index("Label")
                            predicted_label_col = linesplit.index("Predicted")
                            probs_cols = linesplit.index("Probs")
                            continue


                        if line.startswith("!@#"): # for debug
                            continue

                        if ".jpg" in line:
                            img_filename = linesplit[img_filename_col]
                            if img_filename in processed_images_lines.keys(): # already processed False predicted instance of image - Don't need another false.
                                if "False" in processed_images_lines[img_filename]:
                                    continue

                            processed_images_lines[img_filename] = line # save original line for readibility when writeing it to a file later!



    misclassified_str = ""


    target_names = []
    true_labels = []
    predicted_labels = []
    predicted_probs_by_img_filename = {}


    if processed_images_lines:

        for img_filename, line in processed_images_lines.items():

            stripped_line   = line.strip().replace("\t", " ")  # save original line for writing to a file.
            stripped_line   = " ".join(stripped_line.split())
            linesplit       = stripped_line.split()

            # save all labels - true and false predictions for classification report
            true_label      = int(linesplit[label_col])
            predicted_label = int(linesplit[predicted_label_col])

            target_names.append(img_filename.partition("_")[0])
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            labels = [int(x[0]) for x in re.findall(r"\d:", str(linesplit[probs_cols:]))]
            probs = [float(x) for x in re.findall(r"\d\.\d+", str(linesplit[probs_cols:]))]
            predicted_probs_by_img_filename[img_filename] = (list([(l, p) for l, p in zip(labels, probs)]))

            if true_label != predicted_label:
                misclassified_str += line
                if predicted_probabilities_of_misclassified is not None:

                    labels = [int(x[0]) for x in re.findall(r"\d:", str(linesplit[probs_cols:]))]
                    probs = [float(x) for x in re.findall(r"\d\.\d+", str(linesplit[probs_cols:]))]

                    predicted_probabilities_of_misclassified[img_filename] = list([(l, p) for l, p in zip(labels, probs)])



        new_str = "{:<20} {:<10} {:<10} {:<10} {:}\n".format("Image", "Label", "Predicted", "Status", "Probs") + \
                                  "\n".join(natsorted(misclassified_str.split("\n"), key=lambda line: line.split(" ")[0])) + "\n\n\n" + \
                                  get_confusion_matrix_and_report_str(true_labels, predicted_labels, target_names_and_labels_for_presentation) + "\n"
                
                
        average_c_filename = ""
        if write_output_file:
            average_c_filename = average_all_classifications_filename_from_classification_file_pattern(outpath, classification_filename_pattern)
            with open(average_c_filename, "w") as f:
                f.write(new_str)
                
        if demo_mode and "val" in classification_filename_pattern:
            try:
                os.remove(misclassified_images_file)
            except:
                pass
            try:
                shutil.copy2(average_c_filename, misclassified_images_file)
            except Exception as e:
                print(str(e))

        # print("Misclassified:\n" + misclassified_str + "\n")

        if display:
            print("Displaying val misclassified:")
            print("Using " + my_model_data + " for display")
            for line in misclassified_str.split('\n'):
                if not line:
                    continue
                filename = my_model_data + "/" + re.search(r"\w+_\d+\.jpg", line).group(0)
                # img = plt.imread(filename)
                # img -= imgs_mean
                # plt.imshow(img)
                # plt.show()

                # img = io.imread(filename, as_grey=False) # open like caffe.io.load_image
                # img -= imgs_mean
                # io.imshow(img)
                # io.show()

                webbrowser.open(filename)

    return target_names, true_labels, predicted_labels, predicted_probs_by_img_filename
# -------------------------------------------------------------------------------------------------------

def call_predict_on_train_and_val_txts(mode, iter, display_images=False, target_names_and_labels_for_presentation=None):
    """
    Call prediction on the mode's validation set and training set.
    """
    call_predict(mode, iter, mode.train_txt, display_images, target_names_and_labels_for_presentation)
    call_predict(mode, iter, mode.val_txt, display_images, target_names_and_labels_for_presentation)

# -------------------------------------------------------------------------------------------------------

def call_predict_on_val_txts(mode, iter, display_images=False, target_names_and_labels_for_presentation=None):
    """
    Call prediction on the mode's validation set.
    """
    call_predict(mode, iter, mode.val_txt, display_images, target_names_and_labels_for_presentation)

# -------------------------------------------------------------------------------------------------------

def call_predict(mode, iter, imgs_txt_file, display_images=False, target_names_and_labels_for_presentation=None, out_path=None, out_file_name_addition=""):
    """
    Call to predict imgs imgs_txt_file param with specific snapshot of a specific mode.
    """

    weights = mode.resume_from_iter(iter) if iter else mode.caffenet_weights
    return prediction_from_txt_with_deploy(mode=mode,
                                           weights=weights,
                                           iter=iter,
                                           imgs_txt_file=imgs_txt_file,
                                           display_images=display_images,
                                           target_names_and_labels_for_presentation=target_names_and_labels_for_presentation,
                                           out_path=None,
                                           out_file_name_addition="",
                                           called_via_call_predict=True)
# -------------------------------------------------------------------------------------------------------


def train_predict(root_mode, first_set_i=0, last_set_i=-1, mode_of_operation=CROSS_VALIDATION, from_scratch=False):
    """
    Preform cross-validation or a simple, single iteration, training.
    Can be used for testing on increasing set size by calling with different val_set_fractions and different root_mode.extra_info.

    :type root_mode: Mode
    :param root_mode: The root for submodes (CV iteration modes). Use it's mode_path, data_dir, txts_data (only for classes and labels).

    :param first_set_i: From which CV iteration to start

    :param last_set_i: Last cross validation iteration (absolute - not like number of iterations).
                       -1 means run on all sets.
                       Can be used for skipping training and going straight to
                       average_all_classifications and plotting averages by setting to 0.

    :param mode_of_operation: CROSS_VALIDATION to do cross-validation.
                              SIMPLE_TRAIN_PREDICT to do a single iteration train and predict.
                              INCREASING_TRAIN_SET_SIZE to do increasing training set size learning evaluation.

    :param from_scratch: Whether to train from scratch or to fine-tune caffenet weights.
    """

    target_names_and_labels_for_getting_imgs = root_mode.get_target_names_and_labels_tuples_for_real_world()
    
    makedirs_ok(root_mode.mode_path)

    # I'm writing all the images to a single file so that I can always resume or repeat the cross-validaion
    # with the same train and validation sets.

    # if the file exists build a list from it.
    if path.exists(path.join(root_mode.mode_path, "shuffled_imgs_list_order.txt")):

        imgs_and_labels = []
        with open(path.join(root_mode.mode_path, "shuffled_imgs_list_order.txt"), "r") as imgs_and_labels_f:
            for line in imgs_and_labels_f.readlines():
                img = str(line.split(" ")[0])
                label = int(line.split(" ")[1])
                imgs_and_labels.append((img, label))

    else: # Doesn't exist yet - write it

        # if we're using augmented data than the mode.data_dir is the augmented folder
        # but for this list we want the original images.
        if root_mode.augmented_data:
            # my_model_data - always points to original images folder.don't let it read duplicates from augmented folder
            imgs_and_labels = get_shuffled_images_and_labels(my_model_data,
                                                             target_names_and_labels_for_getting_imgs)
        else:
            imgs_and_labels = get_shuffled_images_and_labels(root_mode.data_dir, target_names_and_labels_for_getting_imgs)

        with open(path.join(root_mode.mode_path, "shuffled_imgs_list_order.txt"), "w") as imgs_and_labels_f:
            imgs_and_labels_f.write("\n".join("{} {}".format(img, label) for img, label in imgs_and_labels))


    num_of_imgs = len(imgs_and_labels)

    val_set_size = int(math.ceil(num_of_imgs * root_mode.val_set_fraction))

    if mode_of_operation==CROSS_VALIDATION:
        if val_set_size:
            num_of_iterations = int(math.ceil(float(num_of_imgs) / float(val_set_size)))
        else:
            num_of_iterations = 1 # train on all data - I honestly can't remember why it's here.
    else:
        num_of_iterations=last_set_i

    train_set_size = num_of_imgs - val_set_size

    root_mode.solver_net_parameters.train_set_size = train_set_size
    root_mode.solver_net_parameters.val_set_size = val_set_size

    root_mode.snapshot_iters = root_mode.get_snapshots_iters_by_solver_params()

    for set_i in range(first_set_i, num_of_iterations):


        if last_set_i != -1 and set_i >= last_set_i:
            break

        # build train and val sets.
        if mode_of_operation==CROSS_VALIDATION:
            val_set = imgs_and_labels[set_i * val_set_size: (set_i * val_set_size) + val_set_size] # not cyclic. The last val_set_will contain only the last imgs
            train_set = list(set(imgs_and_labels) - set(val_set))[: train_set_size]

            # If we're using augmented data - get the augmented versions of the current training set.
            if root_mode.augmented_data:
                train_set = get_augment_train_set(train_set, root_mode.data_dir, root_mode.augment_targets)
                root_mode.solver_net_parameters.train_set_size = len(train_set)
            
        else: # In case of increase train set size or simple train predict we may want to train again - shuffle.
            random.shuffle(imgs_and_labels)
            val_set = imgs_and_labels[:val_set_size]  # not cyclic. The last val_set_will contain only the last imgs
            train_set = list(set(imgs_and_labels) - set(val_set))[: train_set_size]

            # make sure there's at least one instace of every class.
            # This is a problem with very small training sets.
            for target in target_names_and_labels_for_getting_imgs.keys():
                if target not in [t[0].partition("_")[0]  for t in train_set]:
                    i = [t[0].partition("_")[0] for t in val_set].index(target)
                    if i > -1:
                        missing_tup = copy.deepcopy(val_set[i])
                        train_set.append(missing_tup)
                        del val_set[i]

        # deprecated - balance the amount of images in all classes.
        if root_mode.balance_target_names is not None:
            train_targets = np.array([target_and_label[0].partition("_")[0] for target_and_label in train_set])
            keep_indices, = np.where(train_targets == root_mode.balance_target_names["keep"])
            throw_from_indices, = np.where(train_targets != root_mode.balance_target_names["keep"])
            throw_how_many = len(throw_from_indices) - len(keep_indices)
            throw_from_this_set, = np.where(train_targets == root_mode.balance_target_names["throw"])
            removes = []
            for i in range(throw_how_many): # can't remove during iteration.. indices are changing. save all objects to remove.
                removes.append(train_set[throw_from_this_set[i]])
            for rem in removes:
                train_set.remove(rem)

        # update the train set and val set sizes.
        solver_net_parameters = update_dynamic_members_in_solver(root_mode.solver_net_parameters, len(train_set), len(val_set))

        # create sub mode in CV.
        if mode_of_operation == CROSS_VALIDATION or  (val_set_size > 0 and last_set_i > 1):
            sub_mode_arg = root_mode.mode + ["set_" + str(set_i)]
            mode = clone_root_mode(root_mode, sub_mode_arg, dummy=False)

        else: # Either we're doing simple train on one set or we're training on all data. Don't create submode.
            mode = root_mode


        # write train.txt
        with open(mode.train_txt, "w") as train_txt:
            write_to_file_from_list_of_img_label_tuples(train_txt, train_set)

        # write val.txt
        with open(mode.val_txt, "w") as val_txt:
            write_to_file_from_list_of_img_label_tuples(val_txt, val_set)


        make_lmdb(root_mode.data_dir, TRAINING_IMAGE_SIZE, mode.data_path) # before make image mean
        make_image_mean_binaryproto(root_mode.data_dir, mode.data_path)
        write_prototxts(mode, solver_net_parameters)
        mode.finalize()  # mandatory step!! after preprocessing

        if PLATFORM == PC_Platform:
            print("don't.. he suffered enough")
            continue
        run_transfer_learning_caffe_cmd(mode, from_scratch=from_scratch, snapshot_or_weights="weights", weights_or_solverstate=mode.weights)


        if val_set_size: #predict if we didn't train on all data.
            for snapshot_iter in root_mode.snapshot_iters:
                
                if demo_mode and snapshot_iter == 0: # don't predict with caffenet weights in demo mode.
                    continue
                elif demo_mode:
                    # for the demo flow I only predicted the validation set (The 'normal flow').
                    call_predict_on_val_txts(mode, snapshot_iter, display_images=False, target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation())
                else:
                    # When not in demo mode I also predict the training set to get more insight (which images are mis classified during training).
                    call_predict_on_train_and_val_txts(mode, snapshot_iter, display_images=False, target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation())

        if not root_mode.save_snapshots:
            mode.delete_snapshots_and_solverstates() # save space


    # Average results
    if not demo_mode:
        average_results_plot_averages_plot_by_classes(root_mode, root_mode.snapshot_iters, include_val_set=val_set_size>0)

# -------------------------------------------------------------------------------------------------------

def average_results_plot_averages_plot_by_classes(mode, iters, include_val_set=True):
    """
    Averages all the mode and its sub-modes predictions. Plot regular learning graphs and also learning graphs of specific classes.
    :param mode: 
    :param iters: snapshot iters.
    :param include_val_set: Weteher or not the mode was trained with validation set. (some modes were trained on all data for prediction on other classes.)
    :return: 
    """
    target_names_and_labels_by_file = {"val":{}, "train":{}}
    makedirs_ok(mode.mode_logs_path)
    makedirs_ok(mode.mode_plots_path)

    for iter in iters:
        train_epoch = get_train_epoch(iter, mode.solver_net_parameters.train_batch_size, mode.solver_net_parameters.train_set_size)

        if include_val_set:
             tn, tl, pl, _ = average_all_classifications(mode.mode_path,
                                                            classification_filename_pattern=get_classification_basename_from_imgs_txt_file(mode.val_txt,iter),
                                                            display=False,
                                                            target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation(),
                                                            outpath=mode.mode_logs_path
                                                            )

             target_names_and_labels_by_file["val"][iter] = tn, tl, pl


        tn, tl, pl, _= average_all_classifications(mode.mode_path,
                                                    classification_filename_pattern=get_classification_basename_from_imgs_txt_file( mode.train_txt, iter),
                                                    display=False,
                                                    target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation(),
                                                     outpath=mode.mode_logs_path
                                                    )
        target_names_and_labels_by_file["train"][iter] = tn, tl, pl

    # Plot from averages.
    if include_val_set: # If there were no tests so there were no predictions
        last_iter = iters[-1]

        plot_learning_curve.average_classifications(logs_path_recursive=mode.mode_path,
                                     out_path=get_classification_average_plot_file_path(out_path=mode.mode_plots_path, mode_name=mode.name, iter=last_iter),
                                     title=mode.plot_title,
                                     x_label= str(last_iter) + " Iterations - " + str(train_epoch) + " epochs")


        plot_learning_curve.plot_accuracy_for_each_class(out_path=mode.mode_plots_path,
                                                     mode_name=mode.name,
                                                     dict_of_labels_by_iteration = target_names_and_labels_by_file,
                                                     mode_target_names_and_labels=mode.get_target_names_and_labels_tuples_for_real_world() ,
                                                     title = mode.plot_title,
                                                     x_label = str(last_iter) + " Iterations - " + str(mode.solver_net_parameters.train_epochs) + " epochs"
                                                     )
# -------------------------------------------------------------------------------------------------------

def demo_average_results_plot_averages(mode, iters):
    """
    Demo flow. Do a light weight version of average_results_plot_averages_plot_by_classes.
    :param mode: 
    :param iters: list of snapshot iters.
    :return: 
    """
    makedirs_ok(mode.mode_logs_path)
    makedirs_ok(mode.mode_plots_path)

    for iter in iters:
        train_epoch = get_train_epoch(iter, mode.solver_net_parameters.train_batch_size, mode.solver_net_parameters.train_set_size)


        average_all_classifications(mode.mode_path,
                                      classification_filename_pattern=get_classification_basename_from_imgs_txt_file(mode.val_txt,iter),
                                      display=False,                                                      
                                     target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation(),
                                      outpath=mode.mode_logs_path
                                                            )



    # Plot from averages.

    last_iter = iters[-1]

    plot_learning_curve.average_classifications(logs_path_recursive=mode.mode_path,
                                     out_path=get_classification_average_plot_file_path(out_path=mode.mode_plots_path, mode_name=mode.name, iter=last_iter),
                                     title=mode.plot_title,
                                     x_label= str(last_iter) + " Iterations - " + str(train_epoch) + " epochs")

# -------------------------------------------------------------------------------------------------------

def save_logs_recursively(logs_root, dst_folder_name):
    """
    Call the save_logs module. This is used to call it via uber_script from the cmdline.
    """
    save_logs.save_logs_recursively(logs_root, dst_folder_name)
# -------------------------------------------------------------------------------------------------------


def recursive_predict_all_submodes_on_their_train_val_txts(root_mode):
    """
    Re predict train and val imgs with saved snapshots from cross validation
    :type root_mode: Mode
    """
    iter = []
    for mode in root_mode.get_sub_modes():
        iters = mode.get_existing_snapshots_iters()
        for iter in iters:
            call_predict_on_train_and_val_txts(mode,
                                                iter,
                                                display_images=False,
                                                target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation()
                                                )

    for iter in iters:
        average_all_classifications(root_mode.mode_path,
                                    classification_filename_pattern=get_classification_basename_from_imgs_txt_file(root_mode.val_txt,iter),
                                    display=False,
                                    target_names_and_labels_for_presentation=root_mode.get_target_names_and_labels_tuples_for_presentation())

        average_all_classifications(root_mode.mode_path,
                                   classification_filename_pattern=get_classification_basename_from_imgs_txt_file( root_mode.train_txt, iter),
                                   display=False,
                                   target_names_and_labels_for_presentation=root_mode.get_target_names_and_labels_tuples_for_presentation())
# -------------------------------------------------------------------------------------------------------

def predict_custom_imgs_file_with_one_mode(mode, imgs_txt_file, classification_files_outpath=None, out_filename_addition=""):
    """
    Called from recursive_predict_all_submodes_on_custom_file.
    """
    iters = mode.get_existing_snapshots_iters()
    for iter in iters:
        call_predict(mode,
                     iter,
                     imgs_txt_file=imgs_txt_file,
                     display_images=False,
                     target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation(),
                     out_path=classification_files_outpath, # if None classification file will be saved in sub mode logs path
                     out_file_name_addition=out_filename_addition)
# -------------------------------------------------------------------------------------------------------

def predict_target(root_mode, target_name, label=kc_label):   
    """
    Use pre saved snapshots from cross validation to predict a certain class (target). 
    I don't really remember why I used it.
    :type root_mode: Mode
    """
    classification_path = root_mode.mode_path + "/predict_" + target_name
    makedirs_ok(classification_path)
    with open(root_mode.data_dir + "/{}.txt".format(target_name), "w") as f:
        for img in glob.glob(root_mode.data_dir + "/{}_*.jpg".format(target_name)):
            f.write(path.basename(img) + " " + str(label) + "\n")
    try:
        shutil.copy(root_mode.data_dir + "/{}.txt".format(target_name), classification_path)
    except:
        print("Couldn't copy file")
        return
    imgs_txt_file = classification_path + "/{}.txt".format(target_name)
    recursive_predict_all_submodes_on_custom_file(root_mode, imgs_txt_file, classification_path, classification_path)
#-------------------------------------------------------------------------------------------

def recursive_predict_all_submodes_on_custom_file(root_mode, imgs_txt_file, classification_files_outpath=None, classification_average_outpath=None):
    """
    Use pre saved snapshots from cross validation to predict on a custom set of images. 
    I don't really remember why I used it.
    imgs_txt_file - full path to val.txt or train.txt or cly.txt or whatever.
    :type root_mode: Mode
    """

    # create classification files


    iters = []
    for mode in root_mode.get_sub_modes():
        print(mode.name)
        iters = mode.get_existing_snapshots_iters()
        # If all classification files are going to the same folder we need a name addition to each classification file.
        out_filename_addition = "" if not classification_files_outpath else mode.mode[-1]

        predict_custom_imgs_file_with_one_mode(mode, imgs_txt_file, classification_files_outpath, out_filename_addition)

    # Average results from classification files
    if classification_files_outpath is None:
        classification_files_outpath = root_mode.mode_path


    for iter in iters:
        classification_file_pattern = get_classification_basename_from_imgs_txt_file(imgs_txt_file, iter)
        average_all_classifications(classification_files_root_path=classification_files_outpath,
                              classification_filename_pattern=classification_file_pattern,
                              display=False,
                              target_names_and_labels_for_presentation=root_mode.get_target_names_and_labels_tuples_for_presentation(),
                              outpath=classification_average_outpath)

# -------------------------------------------------------------------------------------------------------

def simple_run_mode(mode, manual_solve_with_intermidiate_steps=False):
    """
    Run the net with out cross validation. 
    This flow wasn't used recently so it's kind of depracated.
    For simple run just do call train_predict with mode_of_operation=SIMPLE_TRAIN_PREDICT
    :type root_mode: Mode
    """

    write_train_val_txts(mode) # before make lmdb
    make_lmdb(mode.data_dir, TRAINING_IMAGE_SIZE, mode.data_path) # before make image mean
    make_image_mean_binaryproto(mode.data_dir, mode.data_path)
    write_prototxts(mode, mode.solver_net_parameters)
    mode.finalize() # mandatory step!! Noneafter preprocessing

    if manual_solve_with_intermidiate_steps:
        call_external_solve(mode, mode.solver_net_parameters)
    else:
        run_transfer_learning_caffe_cmd(mode, snapshot_or_weights="weights", weights_or_solverstate=mode.weights)
    #
    #
    for iter in mode.get_existing_snapshots_iters():
        call_predict(mode, iter, custom_files="val_cly", display_images=False, target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation()())
    #
    #
    plot_learning_curve.plot_log(mode.mode_logs_path, mode.log)


# -------------------------------------------------------------------------------------------------------

def increasing_train_set_size(root_mode, val_set_fractions):
    """
    Evaluate learning of Healthy vs KC with increasing training set sizes.
    Run some cross-validation iteration on increasing training set size.
    Average results of each
    :type root_mode: Mode
    """
    target_names_and_labels_for_getting_imgs = root_mode.get_target_names_and_labels_tuples_for_real_world()
    imgs_and_labels = get_shuffled_images_and_labels(root_mode.data_dir, target_names_and_labels_for_getting_imgs)

    num_of_imgs = len(imgs_and_labels)
    val_set_size = int(math.ceil(num_of_imgs * root_mode.val_set_fraction))

    train_set_size = num_of_imgs - val_set_size

    for val_set_fraction in val_set_fractions:
        train_set_fraction = 1 - val_set_fraction
        val_set_fraction_str = "{:.2f}".format(train_set_fraction).replace(".","_")
        
        submode_mode_arg = root_mode.mode + [val_set_fraction_str + "_train_set_fraction"]
        sub_mode = clone_root_mode(root_mode, submode_mode_arg)
        sub_mode.val_set_fraction = val_set_fraction

        val_set_size = int(math.ceil(num_of_imgs * sub_mode.val_set_fraction))

        train_set_size = max(2, num_of_imgs - val_set_size)

        sub_mode.plot_title = "Healthy vs. KC With Train set of size " + str(train_set_size)

        if PLATFORM== PC_Platform:
            last_set_i = 0
        else:
            last_set_i = 5
        train_predict(sub_mode, first_set_i=0, last_set_i=last_set_i, mode_of_operation=INCREASING_TRAIN_SET_SIZE)
# -------------------------------------------------------------------------------------------------------

def plot_suspects_predictions(mode, iter, suspects_predicted_with_net_that_wasnt_trained_on_suspects=False):
    """
    Compare the mode's suspects predictions with my subjective evaluation. 
    Treat differently if the mode was trained on suspects or not.
    """
    dir = os.path.join(mode.mode_logs_path, "evaluate_misclassified_probabilites")
    makedirs_ok(dir)

    if suspects_predicted_with_net_that_wasnt_trained_on_suspects:
        classification_file_basename = get_classification_basename_from_imgs_txt_file("sus.txt", iter)
        average_filename_basename = classification_file_basename  # not "_average" on custom predictions

    else:
        classification_file_basename = get_classification_basename_from_imgs_txt_file("val.txt", iter)
        average_filename_basename = path.basename(average_all_classifications_filename_from_classification_file_pattern(mode.mode_path, classification_file_basename))

    _, _, _, predicted_probs_by_img_filename = average_all_classifications(mode.mode_path,
                                                                                            classification_filename_pattern=average_filename_basename,
                                                                                            display=False,
                                                                                            target_names_and_labels_for_presentation=mode.get_target_names_and_labels_tuples_for_presentation(),
                                                                                            outpath=dir)

    ks_sus_subjective_probabilities = {}
    with open(my_model_data + "/kc_sus_kc_probabilities.txt") as f:
        for line in f.readlines():
            if line != "" and "sus" in line:
                img_filename = line.split(" ")[0]
                kc_prob = float(line.split(" ")[1])
                healthy_prob = 1 - kc_prob
                ks_sus_subjective_probabilities[img_filename] = list(
                    [(mode.targets_and_labels["healthy"], healthy_prob),
                     (mode.targets_and_labels["kc"], kc_prob)])

    subjective_probs_scatter_dict = PlotDict(x=[], y=[], color='blue', marker='o', markersize=35, legend = "subjective KC probability")
    predicted_probs_scatter_dict = PlotDict(x=[], y=[], color='orange', marker='o', markersize=40, legend = "predicted KC probability")
    net_predicted_right_scatter_plot = PlotDict(x=[], y=[], color='green', marker='^', markersize=50, legend = "net predicted right")
    net_predicted_wrong_scatter_plot = PlotDict(x=[], y=[], color='red', marker='v', markersize=50, legend = "net predicted wrong")
    xticks = []
    net_right_prediction = []
    subjective_right_prediction = []

    i = 0
    for img, subjective_labels_and_probs in natsorted(ks_sus_subjective_probabilities.items(), key=lambda img_label_tup: img_label_tup[0]):
        subjective_labels, subjective_probs = zip(*subjective_labels_and_probs)
        sujective_kc_prob =  subjective_probs[1]

        subjective_label = subjective_probs.index(max(subjective_probs))
        subjective_right_prediction.append(subjective_label == mode.targets_and_labels['kc'])


        if img in predicted_probs_by_img_filename.keys():

            predicted_labels_and_probs = predicted_probs_by_img_filename[img]

            predicted_labels, predicted_probs = zip(*predicted_labels_and_probs)

            max_predicted_prob = max(predicted_probs)


            predicted_label = predicted_labels[predicted_probs.index(max_predicted_prob)]

            predicted_kc_prob = predicted_probs[mode.targets_and_labels['kc']]

            subjective_probs_scatter_dict.x.append(i)
            subjective_probs_scatter_dict.y.append(sujective_kc_prob)
            predicted_probs_scatter_dict.x.append(i)
            predicted_probs_scatter_dict.y.append(predicted_kc_prob)
            net_right_prediction.append(predicted_label == mode.targets_and_labels['kc'])

            xticks.append(img.replace(".jpg", ""))
            #
            # if predicted_label != mode.targets_and_labels['kc']:
            #     net_predicted_wrong_scatter_plot.x.append(i)
            #     net_predicted_wrong_scatter_plot.y.append(1.1)
            # else:
            #     net_predicted_right_scatter_plot.x.append(i)
            #     net_predicted_right_scatter_plot.y.append(1.1)
            #     net_right_prediction.append(True)


            i += 1

    plot_learning_curve.scatter_suspects_probabilities(
                                        [
                                        subjective_probs_scatter_dict,
                                         predicted_probs_scatter_dict,
                                         # net_predicted_right_scatter_plot,
                                         # net_predicted_wrong_scatter_plot
                                         ],
                                        x_ticks=xticks,
                                        net_right_prediction=net_right_prediction,
                                        subjective_right_prediction=subjective_right_prediction,
                                        out=dir + "/subjective_vs_net_kc_predictions " + str(iter) + "_iter",
                                        title='Net Prediction vs. Subjective Predicion\n' + mode.name + " " + str(iter) + " iteration",
                                        x_label="KC Suspects")


# -------------------------------------------------------------------------------------------------------

def main():

    # main
    if PLATFORM == EC2_GPU_Platform:
        
        mode = dummy_healthy_vs_kc_vs_cly_best_iter_cv  

        if os.path.exists(mode.solver_prototxt):
            raw_input(mode.name + " exists. Are you Sure?")
            
        train_predict(mode, first_set_i=1,last_set_i=-1)


        # predict special targets with the mode's saved snapshots.
        # Requires to save the mode's snapshots after we're done with the mode.
        # predict_target(mode, "cly")
        # predict_target(mode, "sus")
        return
    else:


        crop_off_center_augment_every_image_twice(overwrite_folder=True)
        return
        mode = hksc_4_classes_cross_validation_balanced_augmented_data_adam

        # plot_suspects_predictions(mode, 40, suspects_predicted_with_net_that_wasnt_trained_on_suspects=True)
        # return
        if glob.glob(mode.mode_path + "/set_0*"):
            raw_input(mode.name + " exists. Are you Sure?")
        train_predict(mode, first_set_i=0, last_set_i=1)
        return

        # evalute the predictions of suspects.
        # different if the mode was trained on suspects or not.
        if mode.target_in_val("sus"):
            evaluate_misclassified_predicted_probabilities(mode)
        else:
            evaluate_misclassified_predicted_probabilities(mode, suspects_predicted_with_net_that_wasnt_trained_on_suspects=True)


    ### test minimum required train set size to train on Healthy vs. KC.
    # increasing_train_set_size(mode, np.arange(0.05, 1, 0.1))
    # return

    ### save logs
    #save_logs_recursively(mode.mode_path, mode.name)
    #return


    # visualize_net_layers(caffenet_deploy_prototxt, mode.caffenet_weights, my_model_path + "/visualize_caffenet_layers")



# --------------------------------------------------------------------------------------
if __name__=="__main__":

    # call uber_script.py with save_logs <mode name> to save logs to /home/saved_logs directory.
    if "save_logs" in sys.argv:
        src_path = modes_path
        if len(sys.argv) > 2:
            src_path += "/" + sys.argv[2]
        save_logs_recursively(src_path, sys.argv[2])

    # call uber_script from run_demos notebook.
    # demo train predict.
    elif "demo_train_predict" in sys.argv:
        if "from_scratch" in sys.argv:
            mode = recycle_healthy_vs_kc_vs_cly_from_scratch
            from_scratch = True
        else:
            mode = recycle_healthy_vs_kc_vs_cly_best_iter_cv 
            from_scratch = False
        if "last_set_i" in sys.argv:
            last_set_i = int(sys.argv[sys.argv.index("last_set_i")+1]) # the last set will be after "last_set_i" string
        else:
            # by default - do only one cross-validation iteration in demo.
            last_set_i = 1
        train_predict(mode, first_set_i=0,last_set_i=last_set_i, from_scratch=from_scratch)

    else:
        main()





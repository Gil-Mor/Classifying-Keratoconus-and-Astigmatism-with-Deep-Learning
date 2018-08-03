'''
Title           :plot_learning_curve.py
Description     :This script generates learning curves for caffe models
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160619
version         :0.1
usage           :python plot_learning_curve.py model_1_train.log ./caffe_model_1_learning_curve.png
python_version  :2.7.11
'''

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import fnmatch

from sympy.plotting import plot



import matplotlib.pylab as plt
from pylab import MaxNLocator
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patches as mpatches

from export_env_variables import *
import defs
# from defs import *
from utils import *
import re
from sklearn.metrics import precision_recall_fscore_support
# plt.style.use('ggplot')


import textwrap
def wrap_title(title):
    return "\n".join(textwrap.wrap(title, 50))
# --------------------------------------------------------------------------------------------
def get_accuracy_plot_filename_by_target_name(out_path, set):
    return os.path.join(out_path, set + "_accuracy_by_classes") #.png is added by plt.savefig
# --------------------------------------------



def plot_accuracy_for_each_class(out_path, mode_name, dict_of_labels_by_iteration, mode_target_names_and_labels, title, x_label):
    """
        :type out_path: str - full dirname path for output file
        :type dict: dict
        :type title: str
        :type x_label: str

    """


    recall_and_support_by_file_iter_target = {} # When all true labels are positive recall = accuracy and precision has no meaning.

    iters = None

    all_target_names = set() # first collect all target names
    all_labels = set()

    for file in dict_of_labels_by_iteration.keys():

        iters = sorted(dict_of_labels_by_iteration[file].keys())

        for iter in iters:

            target_names, true_labels ,predicted_labels = dict_of_labels_by_iteration[file][iter]

            all_target_names.update(set(target_names))
            all_labels.update(set(np.unique((true_labels, predicted_labels))))

    all_target_names = sorted(list(all_target_names))
    all_labels       = sorted(list(all_labels))

    for file in dict_of_labels_by_iteration.keys():

        if file not in recall_and_support_by_file_iter_target.keys():
            recall_and_support_by_file_iter_target[file] = {}

        iters = sorted(dict_of_labels_by_iteration[file].keys())

        for iter in iters:

            target_names, true_labels, predicted_labels = dict_of_labels_by_iteration[file][iter]

            for target_name in sorted(list(all_target_names)):

                if target_name not in recall_and_support_by_file_iter_target[file].keys():
                    recall_and_support_by_file_iter_target[file][target_name] = []

                label = mode_target_names_and_labels[target_name]

                np_target_names = np.asanyarray(target_names)

                target_indices = np.where(np_target_names == target_name)
                if not target_indices:
                    empty_p_r_f_s = [0 for _ in range(len(all_labels))]
                    p_r_f_s_list = [empty_p_r_f_s, empty_p_r_f_s, empty_p_r_f_s, empty_p_r_f_s]
                    recall_and_support_by_file_iter_target[file][target_name].append(p_r_f_s_list)
                    continue

                true_labels_by_target = np.array(true_labels)[target_indices]
                predicted_labels_by_target = np.array(predicted_labels)[target_indices]

                p, r, f1, s = precision_recall_fscore_support(true_labels_by_target,predicted_labels_by_target, labels=all_labels) # make sure result array are always in the length of all results

                r_s = [r, s]
                recall_and_support_by_file_iter_target[file][target_name].append(r_s)



    colors = {"healthy":"green", "kc":"red", "sus":"orange", "cly":"blue"}
    for file in recall_and_support_by_file_iter_target.keys():
        plot_dicts = []

        for i, target_name in enumerate(sorted(list(all_target_names))):
            label = mode_target_names_and_labels[target_name]

            presentation_target_name = real_world_string_to_presentation_string[target_name]

            recall_non_existent_label_dict = defs.PlotDict(plot=False, x=iters, y=[], color=colors[target_name], legend="No samples of " + presentation_target_name + " in batch", marker='X', markersize=10, markevery=[])
            recall_plot_dict = defs.PlotDict(x=iters, y=[], color=colors[target_name], legend=presentation_target_name + " accuracy")


            for i_plot_point, iter, r_s in zip(range(len(iters)), iters, recall_and_support_by_file_iter_target[file][target_name]):

                r, s = r_s

                if s[label] == 0: # no instances of this label in this iteration
                    recall_non_existent_label_dict.plot = True
                    recall_non_existent_label_dict.markevery.append(i_plot_point)
                    if iter > 0: # assign previous value if exists
                        recall_plot_dict.y.append(recall_plot_dict.y[-1])
                    else:
                        recall_plot_dict.y.append(0)

                else:
                    recall_plot_dict.y.append(r[label])


            recall_non_existent_label_dict.y = recall_plot_dict.y
            plot_dicts.append(recall_plot_dict)
            plot_dicts.append(recall_non_existent_label_dict)
            # plot_dicts = [recall_non_existent_label_dict, recall_plot_dict]

            # out = get_accuracy_plot_filename_by_target_name(out_path, file, target_name)

            # plot_from_dicts(plot_dicts, out, title + " " + presentation_target_name + " accuracy", x_label)

        out = get_accuracy_plot_filename_by_target_name(out_path, file)

        plot_from_dicts(plot_dicts, out, title + " accuracy by class on " + file + " set", x_label)


# --------------------------------------------------------------------------------------------

def average_classifications(logs_path_recursive, out_path, title, x_label="Iterations"):



    logs = []
    train_average = {'#Iters': 0, "TrainingLoss":np.array([], dtype=np.float32)}
    test_average = {"TestLoss": np.array([], dtype=np.float32), "TestAccuracy":np.array([], dtype=np.float32)}

    for root, dirnames, filenames in os.walk(logs_path_recursive):
        for filename in filenames:

            if filename.endswith(".caffe.log"):
                logs.append(path.join(root, filename))

    for i, log in enumerate(logs):
        # Get directory where the model logs is saved, and move to it
        model_log_dir_path = os.path.dirname(log)
        os.chdir(model_log_dir_path)
        train_log_path = log + '.train'
        test_log_path = log + '.test'
        try:
            caffe_root_custom =  ec2_caffe_root if "ec2_modes" in logs_path_recursive else caffe_root # parse newer caffe log format from ec2 with new caffe tools
            command = os.path.join(caffe_root_custom,'tools/extra/parse_log.sh ') + log
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()

            # Read training and test logs

            train_log = pd.read_csv(train_log_path, delim_whitespace=True)
            test_log = pd.read_csv(test_log_path, delim_whitespace=True)

            if i == 0:
                train_average["#Iters"] = train_log["#Iters"]
                test_average["#Iters"] = test_log["#Iters"]
                train_average["TrainingLoss"] = np.zeros(shape=np.array(train_log["TrainingLoss"], dtype=np.float32).shape)
                test_average["TestLoss"] = np.zeros(shape=np.array(test_log["TestLoss"], dtype=np.float32).shape)
                test_average["TestAccuracy"] = np.zeros(shape=np.array(test_log["TestAccuracy"], dtype=np.float32).shape)


            train_average["TrainingLoss"] += np.array(train_log["TrainingLoss"], dtype=np.float32)
            test_average["TestLoss"] += np.array(test_log["TestLoss"], dtype=np.float32)
            test_average["TestAccuracy"] += np.array(test_log["TestAccuracy"], dtype=np.float32)


        except Exception, e:
            print(e.message)

            print("failed with log " + log)
            return

        finally:
            os.remove(train_log_path)
            os.remove(test_log_path)

    train_average["TrainingLoss"] /= len(logs)
    test_average["TestLoss"]  /= len(logs)
    test_average["TestAccuracy"] /= len(logs)

    plot_dicts = [
        defs.PlotDict(x=train_average['#Iters'], y=train_average['TrainingLoss'], color='blue', legend='Training Loss'),
        defs.PlotDict(x=test_average['#Iters'], y=test_average['TestLoss'], color='red', legend='Test Loss'),
        defs.PlotDict(x=test_average['#Iters'], y=test_average['TestAccuracy'], color='green', legend='Test Accuracy')
    ]
    plot_from_dicts(plot_dicts, out=out_path, title=title, x_label=x_label)
        
        
# --------------------------------------------------------------------------------------------

def ord_to_char(v, p=None):
    return chr(int(v))


def scatter_suspects_probabilities(scatter_dicts, x_ticks,  net_right_prediction, subjective_right_prediction, out, title='Training Curve', x_label="Iterations", legend_box_pos=(1, 1)):

    fig, ax1 = plt.subplots()

    fig.set_size_inches(10, 10)

    # ax1.xaxis.grid(True)

    # plt.grid()

    # else:
    #     ax1.xaxis.set_major_formatter(FuncFormatter(ord_to_char))
    #     ax1.xaxis.set_major_locator(MultipleLocator(1))

    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel(x_label, fontsize=15)
    ax1.set_ylabel("KC Probability", fontsize=15)
    ax1.tick_params(labelsize=10)

    legends_plots = []
    legends_strs = []
    for plot_dict in scatter_dicts:


        tmp = ax1.scatter(plot_dict.x, plot_dict.y, color=plot_dict.color, marker=plot_dict.marker, s=plot_dict.markersize, label=plot_dict.legend)

        legends_plots.append(tmp)
        legends_strs.append(plot_dict.legend)


    ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.1))

    ax1.set_xticks(range(len(x_ticks)))
    ax1.set_xticklabels(x_ticks, rotation='vertical', fontsize=10)

    plt.axhline(y=0, xmin=0, xmax=3, linewidth=1, color='grey')
    plt.axhline(y=1, xmin=0, xmax=3, linewidth=1, color='grey')

    plt.axhline(y=0.5, xmin=0, xmax=3, linewidth=1.5, zorder=0, color='green', linestyle='dashed', label="0.5 propability. above: KC. below: Healthy")
    plt.axhspan(0.5, 1 , facecolor='0.2', alpha=0.2)
    plt.axhspan(0, 0.5 , facecolor='0.5', alpha=0.2)

    for i, right_prediction in enumerate(net_right_prediction):
        if right_prediction:
            if subjective_right_prediction[i]:
                ax1.axvspan(i-0.5, i+0.5, alpha=0.1, color='green')
            else:
                ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')

        else:

            if subjective_right_prediction[i]:
                ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
            else:
                ax1.axvspan(i-0.5, i+0.5, alpha=0.1, color='red')



    # Adding legend

    a = mpatches.Patch( alpha=0.1, color='green', label='both predicted kc')
    b = mpatches.Patch( alpha=0.2, color='green', label='net: kc. subjective: healthy')
    c = mpatches.Patch(alpha=0.1, color='red', label='both predicted healthy')
    d = mpatches.Patch(alpha=0.2, color='red', label='subjective: kc. net: healthy')

    handles, labels = ax1.get_legend_handles_labels()
    handles += [a,b,c,d]
    lgd = plt.legend(handles=handles,
                     bbox_to_anchor=legend_box_pos)


    plt.title(wrap_title(title), fontsize=18)

    if out.endswith(".png"):
        out = out.replace(".png", "")


    fig.set_size_inches(15, 10)

    plt.savefig(out, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.5)

    plt.close()


# --------------------------------------------------------------------------------------------

def plot_from_dicts(plot_dicts, out, title='Training Curve', x_label="Iterations", legend_box_pos=(1, 1)):
    '''
    Making learning curve
    '''


    max_y = max([max(plot_dict.y) for plot_dict in plot_dicts if plot_dict.y != []])

    if max_y > 2.0: # If Loss is too high make two plot - one with original loss and one with y_max = 2 so that Accuracy is clear
        y_maxes = [2.0, max_y]
    else:
        y_maxes = [2.0] # Always make minimum y_max 2.0 for legend box

    for y_max in y_maxes:


        fig, ax1 = plt.subplots()


        fig.set_size_inches(10,10)

        ax1.xaxis.grid(True)

        plt.grid()

        ax1.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        ax1.set_ylim(0, y_max)
        ax1.set_xlabel(x_label, fontsize=15)
        ax1.tick_params(labelsize=10)



        legends_plots = []
        legends_strs = []
        x_ticks = []
        for plot_dict in plot_dicts:
            if plot_dict.plot:

                if len(plot_dict.x) > len(x_ticks):
                    x_ticks = plot_dict.x
                tmp, = ax1.plot(plot_dict.x, plot_dict.y, linewidth=2, color=plot_dict.color, marker=plot_dict.marker, markersize=plot_dict.markersize, markevery=plot_dict.markevery)

                legends_plots.append(tmp)
                legends_strs.append(plot_dict.legend)



        if y_max < 3.5:
            y_ticks_steps = 0.1
        elif y_max < 5:
            y_ticks_steps = 0.5
        elif y_max < 20:
            y_ticks_steps = 1
        else:
            y_ticks_steps = 5

        y_ticks_until_1 = np.arange(0, 1.05, 0.1)
        if y_max > 5:
            yticks = np.arange(0, y_max + (float(y_ticks_steps) / 2), y_ticks_steps)
        elif y_max > 1:
            yticks = np.concatenate((y_ticks_until_1, np.arange(1,  y_max + (float(y_ticks_steps)/2), y_ticks_steps)))
        else:
            yticks = y_ticks_until_1

        ax1.yaxis.set_ticks(yticks)


        ax1.xaxis.set_ticks(x_ticks)

        max_accuracy_line = plt.axhline(y=1, xmin=0, xmax=3, linewidth=1.5, zorder=0, color='green', linestyle='dashed')
        min_loss_line = plt.axhline(y=0, xmin=0, xmax=3, linewidth=1.2, zorder=0, color='m', linestyle='dashed')


        # Adding legend
        lgd = plt.legend(legends_plots + [min_loss_line, max_accuracy_line],
                   legends_strs + ["Min Loss/Accuracy = 0", "Max Accuracy = 1"],
                   bbox_to_anchor=legend_box_pos)

        plt.title(wrap_title(title), fontsize=18)
        # plt.tight_layout() # not good
        # Saving learning curve

        if out.endswith(".png"):
            out = out.replace(".png", "")

        if len(y_maxes) > 1:
            if y_max > 2.0:
                fig.set_size_inches(15, 10)

                plt.savefig(out, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.5)
            else:
                plt.savefig(out + "_closeup", bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.5)
        else:
            plt.savefig(out, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.5)

        if demo_mode:
            plt.show()
        if not demo_mode:
            plt.close()

# --------------------------------------------------------------------------------------------


def plot_from_log_txt(log, out, title='Training Curve'):
    model_log_path = log
    learning_curve_path = out
    # Get directory where the model logs is saved, and move to it
    model_log_dir_path = os.path.dirname(model_log_path)
    os.chdir(model_log_dir_path)
    train_log_path = model_log_path + '.train'
    test_log_path = model_log_path + '.test'
    try:

        '''
        Generating training and test logs
        '''
        #Parsing training/validation logs

        caffe_root = r"/home/gil/Caffe/caffe/" # update variable from export_env_variables.py

        command = caffe_root + 'tools/extra/parse_log.sh ' + model_log_path
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        #Read training and test logs
        train_log_path = model_log_path + '.train'
        test_log_path = model_log_path + '.test'
        train_log = pd.read_csv(train_log_path, delim_whitespace=True)
        val_log = pd.read_csv(test_log_path, delim_whitespace=True)


        plot_from_dicts(train_log, val_log, out, title)

    except:
        print("Failed plot " + log)

    '''
    Deleting training and test logs
    '''
    command = 'rm ' + train_log_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    command = command = 'rm ' + test_log_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

#--------------------------------------------------------------------------------

def make_out_and_title_from_log(log_path):
    out = log_path.replace(caffe_log_postfix, ".png")
    title = os.path.basename(log_path).replace(".log", "").replace("_", " ")
    return out, title
#--------------------------------------------------------------------------------

def plot_all_logs_in_subfolders(folders):
    for folder in folders:
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name.endswith(".log.test") or  name.endswith(".log.train") :
                    os.remove(os.path.join(path, name))

                if name.endswith(".caffe.log"):
                    log = os.path.join(path, name)
                    out, title = make_out_and_title_from_log(log)
                    plot_from_log_txt(log, out, title)


# ------------------------------------------------------------------------------


def plot_log(dir_to_recursive_search, log_name):
    log = find_files(dir_to_recursive_search, log_name)
    out, title = make_out_and_title_from_log(log)
    plot(log, out, title)
# ------------------------------------------------------------------------------




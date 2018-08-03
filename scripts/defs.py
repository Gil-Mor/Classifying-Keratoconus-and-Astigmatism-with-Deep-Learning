import export_env_variables
from export_env_variables import *
import cv2
import shutil
import PIL
import errno
import glob
import random
import numpy as np
from natsort import natsorted, ns # natural sort alphanumerical strings with natsort(list)
import pickle
import re

import utils
from utils import *
import copy



class Txts_data:
    """
    Use to manually define the content of train/val txts. Currently not used because it's not so useful for cross-validation.
    Currently I do CV iteration by splitting the data according to percentages.
    Use in accordence with the write_train_val_txts(mode) in utils.py
    """
    def __init__(self, train_healthy=0, train_kc=0, train_sus=0, train_cly=0, val_healthy=0,
                 val_kc=0, val_sus=0, val_cly=0, val_dummy=0,
                 targets_and_labels=(("healthy", label_a), ("kc", label_b)),
                 h_label=label_a, k_label=label_b, s_label=label_c, c_label=label_c, dummy_label=label_c):
        self.train_healthy = train_healthy
        self.train_kc = train_kc
        self.train_sus = train_sus
        self.train_cly = train_cly
        self.val_healthy = val_healthy
        self.val_kc = val_kc
        self.val_sus = val_sus
        self.val_cly = val_cly
        self.val_dummy = val_dummy

        self.healthy_label = h_label
        self.kc_label = k_label
        self.cly_label = c_label
        self.sus_label = s_label
        self.dummy_label = dummy_label


        self.shuffle_healthy = True
        self.shuffle_kc = True
        self.shuffle_sus = True
        self.shuffle_cly = True

# -----------------------------------------------------------------------


BEST_KNOWN_MOMENTUM=0.5
BEST_KNOWN_WEIGHT_DECAY=0.001
BEST_KNOWN_LR=0.001

VANILLA_SGD_SOLVER_TYPE="vanilla_sgd"
ADAM_SOLVER_TYPE="Adam"


platform_max_batch_size = 256

class Solver_Net_parameters:
    """
    The net configurations.
    """
    def __init__(self,
                 solver_type=VANILLA_SGD_SOLVER_TYPE,
                 train_epochs=-1,
                 train_set_size=-1,
                 val_set_size=-1,
                 max_batch_size=platform_max_batch_size,
                 train_batch_size=-1,
                 val_batch_size=-1,
                 max_iter=-1,
                 display_iter=-1,
                 test_interval=-1,
                 test_iter=-1,
                 snapshot_iter=-1,
                 momentum=BEST_KNOWN_MOMENTUM,
                 weight_decay=BEST_KNOWN_WEIGHT_DECAY,
                 lr=BEST_KNOWN_LR,
                 lr_stepsize=-1):
        """

        :param solver_type:
        :param train_epochs: For how many epochs to train.
        :param train_set_size: size of training set.
        :param val_set_size: Size of validation set.
        :param max_batch_size: Maximum batch size.
        :param train_batch_size: Batch size during training.
                                 Calculated dynamically from train_set_size.
                                 If train_set_size >= max_batch_size then use max_batch_size.
        :param val_batch_size: same as train_batch_size.
        For other parameters check: https://github.com/BVLC/caffe/wiki/Solver-Prototxt
        """

        self.solver_type            = solver_type
        self.train_epochs           = train_epochs
        self.train_set_size         = train_set_size
        self.val_set_size           = val_set_size
        self.max_batch_size         = max_batch_size
        self.max_iter               = max_iter # don't let max_iter be dynamically calculated - let it be a nice round hard coded number.
        self.display_iter           = display_iter
        self.test_interval          = test_interval
        self.snapshot_iter          = snapshot_iter
        self.momentum               = momentum
        self.weight_decay           = weight_decay
        self.lr                     = lr
        self.lr_stepsize            = lr_stepsize

        # Dynamic stuff - give the option to give hard coded values for flexibility
        self._train_batch_size       = train_batch_size
        self._val_batch_size         = val_batch_size
        self._test_iter              = test_iter

    # -------------------------------------------




    @property
    def train_batch_size(self):
        """
        Update the train batch size dynamically
        :return:
        """
        self._train_batch_size = min(self.max_batch_size, self.train_set_size)
        return self._train_batch_size

    # -------------------------------------------

    @property
    def val_batch_size(self):
        self._val_batch_size = min(self.max_batch_size, self.val_set_size)
        return self._val_batch_size

    # -------------------------------------------

    # @property # No - Want nice round hard coded values.. not dynamically calculated
    # def max_iter(self):
    #     if self._max_iter != -1: # give the option to set
    #         return self._max_iter
    #
    #     if self.train_epochs != -1 and self.train_set_size == -1 and self.train_batch_size == -1:
    #         self._max_iter = int(math.ceil(float(self.train_set_size) / float(self.train_batch_size)) * self.train_epochs)
    #
    #     return self._max_iter
    # # -------------------------------------------

    @property
    def test_iter(self):
        if self.val_set_size == 0:
            self._test_iter = 0
        elif self.val_set_size != -1 and self.val_batch_size != -1:
            self._test_iter = int(math.ceil(float(self.val_set_size) / float(self.val_batch_size)))
        return self._test_iter
    # -------------------------------------------


    def __str__(self):
        s = ""
        key_val = vars(self).items() # is OK vars gives hidden members and doesn't give methods
        key_val.sort()
        for key, val in key_val:
            line = '{:<25}  {:}'.format(str(key), str(val)) + "\n"
            s += line

        return s
    # -------------------------------------------

# -----------------------------------------------------------------------

def update_dynamic_members_in_solver(src_solver, train_set_size, val_set_size):
    """
    Copy a solver but update train set size and val set size.
    """
    cpy = copy.deepcopy(src_solver)
    cpy.train_set_size = train_set_size
    cpy.val_set_size = val_set_size
    return cpy

# -----------------------------------------------------------------------


# ================= MODES STUFF ===============================
# ----------------------- MODE RELATED FILES ---------------
train_val_prototxt_postfix = "_train_val.prototxt"
train_net_prototxt_postfix = "_train_net.prototxt"
val_net_prototxt_postfix = "_val_net.prototxt"
solver_prototxt_postfix = "_solver.prototxt"
deploy_prototxt_postfix = "_deploy.prototxt"
mode_pickle_filename = "mode_pickle.pkl"
caffe_log_postfix = ".caffe.log"
snapshot_postfix = "_snapshot"
caffe_snapshot_postfix = "_iter_{iter}.caffemodel"
solverstate_postfix = "_snapshot"
caffe_solverstate_postfix = "_iter_{iter}.solverstate"


def get_sub_modes_from_sub_dirs(root_path, solver_net_params, txts_data):
    """ 
    Get sub-mode instances from root_mode subdirectories (get cross validation iterations submodes like set_0, set_1 etc..).
    Use if you want to re predict or work with the saved statistics of all cross-validation iterations.
    """
    modes = []
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if "_solver.prototxt" in filename:

                full_mode = root.partition("modes/")[2]

                mode = full_mode.split("/")

                modes.append(Mode(mode, solver_net_params, txts_data, dummy=True))

    return modes

# --------------------------------------------

def get_sub_modes_from_pickles(root_path):
    """ 
    Get sub-mode instances from pickles. I started saving a pickle of every mode in the mode's path at some point.
    """
    modes = []
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == mode_pickle_filename:
                modes.append(pickle.load(open(os.path.join(root, filename), "rb")))

    return modes
# --------------------------------------------

def get_classification_path_from_imgs_txt_file(path, imgs_txt_file, iter):
    """
    Build a classification file name for predictions.
    """
    return os.path.join(path, get_classification_basename_from_imgs_txt_file(imgs_txt_file, iter))
# --------------------------------------------

def get_classification_basename_from_imgs_txt_file(imgs_txt_file, iter):
    filename = ""
    if imgs_txt_file:
        filename += str(os.path.basename(imgs_txt_file.replace(".txt",""))) + "_"

        filename += "classification"

    if str(iter):
        filename += "_" + str(iter)

        filename += ".log"
    return filename
# --------------------------------------------

def average_all_classifications_filename_from_classification_file_pattern(path, classification_file_pattern):
    """
    Build a file name for the average classifications file.
    """
    return os.path.join(path, classification_file_pattern.replace(".log","") + "_classifications_average.log")
# --------------------------------------------

def get_classification_average_plot_file_path(out_path, mode_name, iter):
    """
    Build a file name for the average plot file.
    """
    return os.path.join(out_path, mode_name + "_{}_iter_average".format(iter)) #.png is added by plt.savefig
# --------------------------------------------

def get_precision_recall_plot_filename_by_target_name(out_path, set, target_name):
    return os.path.join(out_path, set + "_" + target_name + "_precision_recall") #.png is added by plt.savefig
# --------------------------------------------

def clone_root_mode(src_mode, new_mode, dummy=False):
    """
    :type src_mode: Mode
    """
    return Mode(mode=new_mode,
                solver_net_parameters=copy.deepcopy(src_mode.solver_net_parameters),
                txts_data=copy.deepcopy(src_mode.txts_data),
                dummy=dummy,
                data_dir=src_mode.data_dir,
                plot_title=src_mode.plot_title,
                save_snapshots=src_mode.save_snapshots,
                val_set_fraction=src_mode.val_set_fraction,
                balance_target_names=copy.deepcopy(src_mode.balance_target_names),
                targets_and_labels=src_mode.targets_and_labels.copy(),
                augmented_data=src_mode.augmented_data,
                augment_targets=src_mode.augment_targets
                )
# --------------------------------------------



class Mode:
    def __init__(self, mode, solver_net_parameters=None, txts_data=None, dummy=False, override_log=True,
                 override_snapshot=False, data_dir=my_model_data, plot_title="", save_snapshots=True,
                 val_set_fraction=0.2, balance_target_names=None,
                 targets_and_labels=(("healthy",label_a), ("kc",label_b)),
                 augmented_data=False,
                 augment_targets=[]):
        """

        :param mode: A list that defines the mode's name and path (they're the same).

        :param solver_net_parameters: The mode's Net configurations.

        :param txts_data: Use to manually specify number of images in train/val sets. Now I use percentages. 
                          Gives more control over the process but not comfortable to use with cross validation.

        :param dummy: Whether it's an existing mode or not. dummy means the mode exits and we're just creting a 'dummy' to 
                      use the existing mode info.

        :param override_log: Whether to override existing caffe log.

        :param override_snapshot: Whether to override existing snapshots.

        :param data_dir: The images directory to use with this mode. Regular/ / augmented etc...

        :param plot_title: Title of learning graph plot.

        :param save_snapshots: whether or not to save snapshots after predictions. Only enable if you have a purpose. 
                               They take a lot of space.

        :param val_set_fraction: The fraction of data that'll be used for the validation set.

        :param balance_target_names: dictionary with "keep" and "throw" keys and real world target names as values ("healthy", "kc", "sus", "cly").
                                    Use if you want to have the same number of targets. only works in train_predict method. I didn't find this helpful.

        :param targets_and_labels: dict list of tuples of targets and their labels for this experiment.

        :param augmented_data: Are we using augmented data or original data.

        :param augment_targets: Which targets did we augment.
        """

        assert type(mode) is list, "mode argument is a list now"


        mode = [x for x in mode if x] # remove dummy parameters

        self.mode = mode

        self.mode_path = path.join(modes_path, *mode) # unpack mode list

        self.mode_logs_path = path.join(self.mode_path, "logs")
        self.mode_plots_path = path.join(self.mode_logs_path, "plots")
        self.visualize_layers_path = path.join(self.mode_logs_path, "visualize_layers")

        self.mode_snapshots_path = path.join(self.mode_path, "snapshots")

        self.data_path = path.join(self.mode_path, "data")

        if not dummy:
            makedirs_ok(self.mode_path)
            makedirs_ok(self.mode_logs_path)
            makedirs_ok(self.mode_plots_path)
            makedirs_ok(self.mode_snapshots_path)
            makedirs_ok(self.data_path)

        self.mean_binaryproto = path.join(self.data_path, my_model_mean_binaryproto_basename)
        self.train_lmdb = path.join(self.data_path, train_lmdb_basename)
        self.val_lmdb = path.join(self.data_path, val_lmdb_basename)
        self.train_txt = path.join(self.data_path, train_txt_basename)
        self.val_txt = path.join(self.data_path, val_txt_basename)


        self.name = '_'.join([x for x in mode if x])

        self.train_val_prototxt = path.join(self.mode_path, self.name + train_val_prototxt_postfix)
        self.solver_prototxt = path.join(self.mode_path, self.name + solver_prototxt_postfix)
        self.deploy_prototxt = path.join(self.mode_path, self.name + deploy_prototxt_postfix)

        self.log = path.join(self.mode_logs_path,
                             self.get_next_filename(self.name, self.mode_logs_path, caffe_log_postfix, override_log))

        self.snapshot_prefix_forsolver_prototxt = path.join(self.mode_snapshots_path,
                                                            self.get_next_filename(self.name, self.mode_snapshots_path, snapshot_postfix, override_snapshot))

        if balance_target_names is not None:
            assert len(balance_target_names.keys()) == 2 and "keep" in balance_target_names.keys() and "throw" in balance_target_names.keys()

        if type(targets_and_labels) is dict:
            self.targets_and_labels = targets_and_labels
        else:
            self.targets_and_labels = {}
            for target_label in targets_and_labels:
                self.targets_and_labels[target_label[0]] = target_label[1]

        self.augmented_data = augmented_data
        self.augment_targets = augment_targets
        self.balance_target_names = balance_target_names
        self.val_set_fraction = val_set_fraction
        self.save_snapshots = save_snapshots
        self.snapshot_resume = None
        self.solverstate_resume = None
        self.caffenet_weights = caffenet_weights
        self.weights = caffenet_weights
        self.state = None  # changes to pycaffe in solve. check in signal handler to know about logs
        self.data_dir = data_dir
        self.pickle_path = os.path.join(self.mode_path, mode_pickle_filename)

        self.plot_title = plot_title
        self.data_preprocessings = []

        self.solver_net_parameters = solver_net_parameters
        self.txts_data = txts_data



    # -------------------------------------------------------------------------------

    def get_train_size(self):
        with open(self.train_txt) as f:
            return len([line for line in f.readlines() if ".jpg" in line])
    # ---------------------------------------------------------------------

    def target_in_val(self, target):
        return target in self.targets_and_labels.keys()

    def get_num_of_classes(self):
        return len(set(self.targets_and_labels.values()))
    # --------------------------------------------------


    def get_target_names_and_labels_tuples_for_real_world(self):
        return self.targets_and_labels
    # --------------------------------------------------

    def get_target_names_and_labels_tuples_for_presentation(self):
        """
        For getting imgs files from folders.
        :return: 
        """

        tuples = []
        if "healthy" in self.targets_and_labels.keys():
            tuples.append(("Healthy", self.targets_and_labels["healthy"]))


        if "kc" in self.targets_and_labels.keys():

            # if KC suspects are labeled as KC then write KC+Sus
            if "sus" in self.targets_and_labels.keys() and self.targets_and_labels["sus"] == kc_label:
                tuples.append(("KC+Sus", self.targets_and_labels["kc"]))

            # if cly are labeled as KC then write KC+CLY
            elif "cly" in self.targets_and_labels.keys() and self.targets_and_labels["cly"] == kc_label:
                tuples.append(("KC+CLY", self.targets_and_labels["kc"]))

            else:
                tuples.append(("KC", self.targets_and_labels["kc"]))

        if "sus" in self.targets_and_labels.keys():

            # If suspects are not labeled as KC then write KC_Sus
            if "kc" not in self.targets_and_labels.keys() or self.targets_and_labels["sus"] != self.targets_and_labels["kc"]:
                tuples.append(("KC_Sus", self.targets_and_labels["sus"]))

        if "cly" in self.targets_and_labels.keys():
            if "kc" not in self.targets_and_labels.keys() or self.targets_and_labels["cly"] != self.targets_and_labels["kc"]:
               tuples.append(("CLY", self.targets_and_labels["cly"]))


        return tuples
    # --------------------------------------------------


    def get_next_filename(self, name, file_path, postfix, override):
        """
        You can choose not to overwrite files like logs and snapshots when re running an existing mode.
        In this case the files will be post-fixed with an index.
        To get the next file name, get the one with the biggest postfix and return the next index.

        I don't always use it.
        """
        if override:
            return name + postfix

        next = name + postfix
        i = 1
        while os.path.exists(path.join(file_path, next)):
            next = name + "_" + str(i) + postfix
            i += 1
        return path.join(file_path, next)

    # --------------------------------------------

    def get_next_prefix_forsolver_prototxt(self):
        self.snapshot_prefix_forsolver_prototxt = self.get_next_filename(self.name, self.mode_snapshots_path,
                                                                         snapshot_postfix, override=False)
        return self.snapshot_prefix_forsolver_prototxt

    # --------------------------------------------

    def get_next_log(self):
        """
        Overwrite logs. Just a mess to keep multiple logs.
        """
        #self.log = self.get_next_filename(self.name, self.mode_logs_path, caffe_log_postfix, override=False)
        return self.log

    # --------------------------------------------


    def get_existing_snapshot(self):
        """
        Get paths of all snapshots file '.caffemodel' of the mode.
        """
        snapshots = glob.glob(self.mode_snapshots_path + "/*.caffemodel")
        if snapshots:
            print("Found Snapshots")
            for s in snapshots:
                print(path.basename(s))
        return snapshots

    # --------------------------------------------

    def delete_snapshots_and_solverstates(self):
        for snapshot in glob.glob(self.mode_snapshots_path + "/*.caffemodel"):
            try:
                os.remove(snapshot)
            except:
                pass

        for solverstate in glob.glob(self.mode_snapshots_path + "/*.solverstate"):
            try:
                os.remove(solverstate)
            except:
                pass
    # --------------------------------------------

    def get_existing_snapshots_iters(self):
        """
        Get a list of snapshot iterations (iterations where we saved a snapshot).
        :return: list of integers.
        """
        iters = []
        for snapshot in glob.glob(self.mode_snapshots_path + "/*.caffemodel"):
            iters.append(int(re.search(r"iter_(\d+)", snapshot).group(1)))
        iters.sort()
        return iters
    # --------------------------------------------

    def get_snapshots_iters_by_solver_params(self, include_zero=True):
        """
        Get a list of snapshot iterations (iterations where we saved a snapshot).
        This method builds the list according to the net parameters and not according to existing files.
        Use this method if you deleted te snapshots but still need the list.
        :return: list of integers.
        """
        start = 0 if include_zero else self.solver_net_parameters.snapshot_iter
        l = list(range(start, self.solver_net_parameters.max_iter + self.solver_net_parameters.snapshot_iter, self.solver_net_parameters.snapshot_iter))
        
        return l
    # --------------------------------------------
        
    def get_existing_snapshots_iters_plus_caffe_weights(self):
        return [0] + self.get_existing_snapshots_iters()
    # --------------------------------------------

    def get_sub_modes(self):
        """
        Try to get submodes from pickles. If the list is empty then get it from sub directories.
        """
        modes = get_sub_modes_from_pickles(self.mode_path)
        if not modes:
            modes = get_sub_modes_from_sub_dirs(self.mode_path, self.solver_net_parameters, self.txts_data)
        return modes
    # --------------------------------------------

    def get_sub_mode_existing_snapshots_iters(self, submode_index=0):
        """
        Get snapshot iterations of a specific sub-mode.
        :param submode_index:
        :return:
        """
        return self.get_sub_modes()[submode_index].get_existing_snapshots_iters
    # --------------------------------------------


    def __str__(self):
        s = ""
        key_val = vars(self).items()
        key_val.sort()
        for key, val in key_val:
            line = '{:<25}  {:}'.format(str(key), str(val)) + "\n"
            s += line

        return s

    # --------------------------------------------

    def get_full_solverstate_basename(self, iter):
        return self.name + solverstate_postfix + caffe_solverstate_postfix.format(iter=iter)

    # --------------------------------------------

    def get_full_snapshot_basename(self, iter):
        return self.name + snapshot_postfix + caffe_snapshot_postfix.format(iter=iter)

    # --------------------------------------------



    def get_solverstate(self, iter):
        if not os.path.exists(path.join(self.mode_snapshots_path, self.get_full_solverstate_basename(iter))):
            print(self.name + solverstate_postfix + caffe_solverstate_postfix.format(iter=iter) + " doesn't exist")
            return None

        return path.join(self.mode_snapshots_path, self.get_full_solverstate_basename(iter))

    # --------------------------------------------

    def get_snapshot(self, iter):
        assert os.path.exists(path.join(self.mode_snapshots_path, self.get_full_snapshot_basename(iter))), self.name + snapshot_postfix + caffe_snapshot_postfix.format(iter=iter) + " doesn't exist"
        if not os.path.exists(path.join(self.mode_snapshots_path, self.get_full_snapshot_basename(iter))):
            raise(self.name + snapshot_postfix + caffe_snapshot_postfix.format(iter=iter) + " doesn't exist")

        return path.join(self.mode_snapshots_path, self.get_full_snapshot_basename(iter))

    # --------------------------------------------



    def resume_from_iter(self, iter):
        self.solverstate_resume = self.get_solverstate(iter)
        self.snapshot_resume = self.get_snapshot(iter)
        return self.snapshot_resume

    # -------------------------------------------------------------------------------------------------------


    def preprocess_all_data(self,open_cv_transformation_functions, delete_and_recopy, display_transformation):
        """
        Deprecated. Given a list of pre-processing functions from pre-processing.py - pre-process the images before training.
        :param open_cv_transformation_functions:
        :param delete_and_recopy:
        :param display_transformation:
        :return:
        """
        self.data_dir = my_model_preprocessed_data
        if os.path.exists(my_model_preprocessed_data) and delete_and_recopy:
            shutil.rmtree(my_model_preprocessed_data)

        if not os.path.exists(my_model_preprocessed_data):
            os.mkdir(my_model_preprocessed_data)

            copy_all_files(my_model_data, my_model_preprocessed_data, "*.jpg")


        list_preprocessing_functions_in_file = open(os.path.join(my_model_preprocessed_data, "a_transformations.txt"), "a")
        self.data_preprocessings.append(open_cv_transformation_functions.__name__)
        list_preprocessing_functions_in_file.write(open_cv_transformation_functions.__name__ + "\n")

        imgs_files = glob.glob(my_model_preprocessed_data + "/*.jpg")
        for img_f in imgs_files:

            preprocessed = open_cv_transformation_functions(img_f)  # each transformation might need to open the file differently.


            if display_transformation:
                PIL.Image.open(img_f).show()
                PIL.Image.fromarray(preprocessed).show()
                raw_input("wait")

            os.remove(img_f)
            PIL.Image.fromarray(preprocessed).save(img_f, "JPEG")

    # ------------------------------------------------------------------
    def finalize(self):
        self.write_info_file()
        pickle.dump(self, open(self.pickle_path, "wb"))
    # ------------------------------------------------------------------

    def write_info_file(self):
        """
        Write the mode's details to a file.
        :return:
        """
        info = open(os.path.join(self.mode_path, "info.txt"), "w")

        info.write(PLATFORM + "\n")

        info.write("mode:\n")
        info.write("\n".join(self.mode) + "\n\n")


        info.write("Solver Net Params:\n")
        info.write(str(self.solver_net_parameters) + "\n\n")
        info.write("Data dir:\n")
        info.write(self.data_dir + "\n\n")
        if len(self.data_preprocessings) > 0:
            info.write("Data Preprocessing:\n")
            info.write("\n".join(self.data_preprocessings))

        info.close()
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------------

class PlotDict:
    """
    Generic data structure for the plotting methods.
    """
    def __init__(self, x, y, color, legend, marker='o', markersize=5, markevery=None, plot=True):
        self.x = x
        self.y = y
        self.color = color
        self.legend = legend
        self.marker = marker
        self.markersize = markersize
        self.markevery = markevery # Only plot certain points on the plot.
        self.plot = plot
# --------------------------------------------------------------------------------------------


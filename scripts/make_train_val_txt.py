#from env_variables import *
import os
from os import path
import random
import sys
from export_env_variables import *
# from defs import *
import glob




def write_normal_healthy_kc_train_val(train_f, val_f, train_set_size, **kwargs):

    random.shuffle(healthy_imgs)
    random.shuffle(kc_imgs)

    train_images = []
    test_images = []

    train_images += (write_to_file_from_list(train_f, healthy_imgs[: train_set_size], healthy_label))
    train_images += (write_to_file_from_list(train_f, kc_imgs[: train_set_size], kc_label))


    test_images += (write_to_file_from_list(val_f,  healthy_imgs[train_set_size:], healthy_label))
    test_images += (write_to_file_from_list(val_f,  kc_imgs[train_set_size:], kc_label))

    return train_images, test_images

#-----------------------------------------------------------------------

def write_normal_healthy_kc_and_sus_train_val(train_f, val_f, train_set_size, **kwargs):
    '''
     Write suspect with kc label. write half of suspects images in each set because there are only 29.
    :param train_f:
    :param val_f:
    :param train_set_size:
    :return:
    '''

    random.shuffle(healthy_imgs)
    random.shuffle(kc_imgs)
    random.shuffle(sus_imgs)

    suspect_to_train = int(NUM_OF_SUS * 0.5)
    train_images = []
    test_images = []

    if "sus_label" in kwargs.keys():
        sus_label = kwargs.get("sus_label")




    train_images += (write_to_file_from_list(train_f,  healthy_imgs[: train_set_size], healthy_label))
    train_images += (write_to_file_from_list(train_f,kc_imgs[: train_set_size], kc_label))
    train_images += (write_to_file_from_list(train_f,  sus_imgs[: suspect_to_train], sus_label))


    test_images += (write_to_file_from_list(val_f,  healthy_imgs[train_set_size:], healthy_label))
    test_images += (write_to_file_from_list(val_f, kc_imgs[train_set_size:], kc_label))
    test_images += (write_to_file_from_list(val_f, sus_imgs[suspect_to_train: ], sus_label))

    return train_images, test_images


# -----------------------------------------------------------------------


def sanity_check_half_train_labeled_wrong(train_f, val_f):

    # Confused train.txt
    write_to_file_by_index_in_filename(train_f, "healthy", range(1,26), kc_label)
    write_to_file_by_index_in_filename(train_f, "healthy", range(26,51), healthy_label)

    write_to_file_by_index_in_filename(train_f, "kc", range(1,13), healthy_label)
    write_to_file_by_index_in_filename(train_f, "kc", range(65,77), healthy_label)
    write_to_file_by_index_in_filename(train_f, "kc", range(13,37), kc_label)


    write_to_file_by_index_in_filename(val_f, "healthy", range(51, 105), healthy_label)
    write_to_file_by_index_in_filename(val_f, "kc", range(37, 65), kc_label)
#-----------------------------------------------------------------------


def sanity_check_half_train_labeled_wrong_random(train_f, val_f, train_set_size):

    random_for_healthy = random.sample(range(1, NUM_OF_HEALTHY+1), NUM_OF_HEALTHY)
    random_for_kc = random.sample(range(1, NUM_OF_KC+1), NUM_OF_KC)

    write_to_file_by_index_in_filename(train_f, "healthy", random_for_healthy[:26], healthy_label)
    write_to_file_by_index_in_filename(train_f, "healthy", random_for_healthy[26:51], kc_label)

    write_to_file_by_index_in_filename(train_f, "kc", random_for_kc[:26], kc_label)
    write_to_file_by_index_in_filename(train_f, "kc", random_for_kc[26:51], healthy_label)

    write_to_file_by_index_in_filename(val_f, "healthy", random_for_healthy[51:], healthy_label)
    write_to_file_by_index_in_filename(val_f, "kc", random_for_kc[51:], kc_label)


# -----------------------------------------------------------------------

def sanity_check_half_of_test_set_labeled_wrong_random(train_f, val_f, train_set_size):

    # switch val txt and train txt
    #sanity_check_half_train_labeled_wrong_random(val_f, train_f, train_set_size)

    random_for_healthy = random.sample(range(1, NUM_OF_HEALTHY+1), NUM_OF_HEALTHY)
    random_for_kc = random.sample(range(1, NUM_OF_KC+1), NUM_OF_KC)

    write_to_file_by_index_in_filename(train_f, "healthy", random_for_healthy[ : train_set_size ], healthy_label)
    write_to_file_by_index_in_filename(train_f, "kc",      random_for_kc[ : train_set_size ], kc_label)

    write_to_file_by_index_in_filename(val_f, "kc", random_for_kc[ train_set_size : train_set_size + NUM_OF_KC//4 ], kc_label)
    write_to_file_by_index_in_filename(val_f, "kc", random_for_kc[ train_set_size + NUM_OF_KC//4 : ], healthy_label)

    write_to_file_by_index_in_filename(val_f, "healthy", random_for_healthy[ train_set_size : train_set_size + NUM_OF_HEALTHY//4 ], healthy_label)
    write_to_file_by_index_in_filename(val_f, "healthy", random_for_healthy[ train_set_size + NUM_OF_HEALTHY//4 : ], kc_label)

# -----------------------------------------------------------------------

def sanity_check_test_set_labeled_wrong(train_f, val_f, train_set_size):

    random_for_healthy = random.sample(range(1, NUM_OF_HEALTHY+1), NUM_OF_HEALTHY)
    random_for_kc = random.sample(range(1, NUM_OF_KC+1), NUM_OF_KC)
    
    
    write_to_file_by_index_in_filename(train_f, "healthy", random_for_healthy[ : train_set_size], healthy_label)
    write_to_file_by_index_in_filename(train_f, "kc", random_for_kc[ : train_set_size], kc_label)

    write_to_file_by_index_in_filename(val_f, "healthy",  random_for_healthy[train_set_size : ], kc_label)
    write_to_file_by_index_in_filename(val_f, "kc", random_for_kc[train_set_size  : ], healthy_label)
# ---------------------------------------------------------------------


def write_to_file_by_index_in_filename(f, category, nums, label):

    # also write to file and also return in list
    res = []
    for n in nums:
        image = category + "_" + str(n) + ".jpg"
        if path.exists(path.join(my_model_data, image)):
            f.write(image + " " + str(label) + "\n")
            res.append(image + " " + str(label))
    return res

#-----------------------------------------------------------------------



# def main(**kwargs):
#
#     prepare_globals()
#
#     cleanup_from_previous()
#
#     train_f, val_f = open_files()
#
#
#     if "size" in kwargs:
#         print("Use \"train_set_size\" and not \"size\"")
#
#     # try getting train set size from kwargs
#
#     train_set_size = kwargs.get("train_set_size", TRAIN_SET_SIZE_DEFAULT)
#
#     function = kwargs.get("function", write_normal_healthy_kc_and_sus_train_val)
#     function(train_f, val_f, train_set_size, **kwargs)
#
#
#     train_f.close()
#     val_f.close()
#     #return train_set, test_set
# # ----------------------------------------------------------------------------

if __name__=="__main__":

    if (len(sys.argv) > 1):
        train_set_size = sys.argv[1]
    else:
        train_set_size = TRAIN_SET_SIZE_DEFAULT


    main(train_set_size=TRAIN_SET_SIZE_DEFAULT)




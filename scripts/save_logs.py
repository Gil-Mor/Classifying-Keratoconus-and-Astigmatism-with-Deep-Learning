from export_env_variables import *
import os
import sys
from os import path
from utils import *
from defs import *
import shutil


def save_logs_recursively(logs_root, dst_folder_name):

    if not path.exists(logs_root):
        print(logs_root + " doesn't exist")
        return

    logs_root_basename = path.basename(logs_root)

    if path.exists(path.join(saved_logs_path, dst_folder_name)):
        ans = raw_input("{} logs backup folder already exist. Merge 'm' or Replace 'r'? ".format(dst_folder_name))
        if ans.lower() == 'r':
            shutil.rmtree(path.join(saved_logs_path, dst_folder_name))

    makedirs_ok(path.join(saved_logs_path, dst_folder_name))

    logs_to_save_dst_path = []
    logs_to_save_src_path = []
    for root, dirnames, filenames in os.walk(logs_root):

        for filename in filenames:
            if filename.endswith(".log") or filename.endswith(".png") or filename == "val.txt" or filename == "train.txt" or filename == "shuffled_imgs_list_order.txt":
                curr_dir = root[root.find(logs_root_basename) + len(logs_root_basename) + 1:]
                logs_to_save_dst_path.append(path.join(saved_logs_path, dst_folder_name, curr_dir, filename))
                logs_to_save_src_path.append(path.join(root, filename))

                # makedirs_ok(path.join(saved_logs_path, dst_folder_name, curr_dir))
                # shutil.copy2(path.join(root, filename), path.join(saved_logs_path, dst_folder_name, curr_dir, filename))


    existing_logs = []
    for root, dirnames, filenames in os.walk(logs_root):

        for filename in filenames:
            if filename.endswith(".log") or filename.endswith(".png"):
                existing_logs.append(path.join(root, filename))

    logs_that_will_be_lost = set(existing_logs) - set(logs_to_save_src_path)

    for log in logs_that_will_be_lost:
        print(log + " will be lost")

    if len(logs_that_will_be_lost) > 0:
        raw_input("\n\n\n", len(logs_that_will_be_lost), " Logs Will Be lost. Are you sure?")

    for src_log, dst_log in zip(logs_to_save_src_path, logs_to_save_dst_path):
        makedirs_ok(path.dirname(dst_log))

        shutil.copy2(src_log, dst_log)
# -------------------------------------------------------------------------------------------------------


if __name__=="__main__":
    src_path = modes_path
    out_path = "modes"
    if len(sys.argv) > 1:
        src_path += "/" + sys.argv[1]
        out_path = sys.argv[1]

    save_logs_recursively(src_path, out_path)

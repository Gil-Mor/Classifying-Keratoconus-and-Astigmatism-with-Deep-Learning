from __future__ import print_function
import os
import fnmatch
import re
from subprocess import call

import subprocess
from export_env_variables import *
import glob
import shutil
import xlrd

from skimage import io, img_as_float

import numpy as np

INVALID_FILENAME = "A_INVALID"


def delete_new_files_with_new_names():
    try:
        new_files_file = open("new_names.txt", "r")
    except:
        return


    try:
        for file in new_files_file.readlines():
            file = file.strip()
            os.remove(file)
    except Exception, e:
        print(e.message)

    new_files_file.close()
    os.remove("new_names.txt")

# ------------------------------------------------------------------------------

def delete_new_files_with_new_names_by_file_prefix(pdfs):

    try:
        for pdf in pdfs:
            if pdf.start_with("_"):
                os.remove(pdf)
    except Exception, e:
        print(e.message)



# ------------------------------------------------------------------------------
def filename_contains_patient_name(filename):
    return re.match(r"^\w\w", filename) and not (filename.startswith("kc_")
                                               or filename.startswith("sus_")
                                               or filename.startswith("healthy_")
                                               or filename.startswith("cly_"))

# ------------------------------------------------------------------------------


def prepend_patient_names_from_txt_file_to_files(pdfs, delete_former_files=True):

    # if delete_former_files:
    #     delete_new_files_with_new_names()

    serial_to_patient_name = open("serial_to_patient_name.txt", "r")

    # new_names_file = open("new_names.txt", "w")

    for line in serial_to_patient_name.readlines():
        line = line.strip()
        if not line:
            continue
        try:
            serial = line.split(' ')[0]
            patient_name = line.split(' ')[1]
            for pdf in pdfs:
                if (serial in pdf)and (not filename_contains_patient_name(pdf)):
                    change_file_name(pdf, patient_name +  "_" + pdf)


            # new_names_file.write(new_name + "\n")
        except:
            pass


    # new_names_file.close()
    serial_to_patient_name.close()
# ----------------------------------------------------------------------------

def info_to_name(info):
    name = info["name"] + "__" if info["name"] else ""

    return name + info["category"] + "_" + info["side"] + "_" + info["serial"]
# ----------------------------------------------------------------------------

def get_all_info_from_filename(pdf, print_stuff=True ,print_erros=True):

    if print_stuff:
        print(pdf)

    info = {"serial": "", "category": "", "side": "", "name": "", "invalid": ""}

    patient_name = pdf[:pdf.find("__")]
    info["name"] = patient_name


    if len(re.findall(r"P\d+", pdf)) > 1:
        if print_erros:
            print("----------------------- MULTIPLE SERIALS")
        info["invalid"] = INVALID_FILENAME
    else:
        try:
            info["serial"] = re.search(r"P\d+", pdf).group(0)
        except:
            return

    if not info["serial"]:
        if print_erros:
            print("---------------------------------- NO SERIAL")
        info["invalid"] = INVALID_FILENAME

    right_seperator = r"[^a-zA-Z0-9]"
    left_seperator = r"(^|_)"
    os = re.search(r"({ls}os{rs}|{ls}left{rs}|{ls}l{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower())
    if os is not None:
        info["side"] = "os"
    if re.search(r"({ls}od{rs}|{ls}right{rs}|{ls}r{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower()) is not None:
        if info["side"] == "os":
            if print_erros:
                print("-------------------------------------- TWO SIDES IN FILE NAME")
            info["invalid"] = INVALID_FILENAME
        info["side"] = "od"
    if info["side"] == "":
        if print_erros:
            print("------------------------------------------------ NO SIDE")
        info["invalid"] = INVALID_FILENAME


    kc_match = re.search(r"({ls}kc{rs}|{ls}kcf{rs}|{ls}sick{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower())
    healthy_match = re.search(r"({ls}normal{rs}|{ls}healthy{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower())
    sus_match = re.search(r"({ls}sus{rs}|{ls}suspect{rs}|{ls}l{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower())
    cly_match = re.search(r"({ls}cly{rs})".format(rs=right_seperator, ls=left_seperator), pdf.lower())

    if kc_match:
        info["category"] = "kc"
    if healthy_match:
        if info["category"]:
            if print_erros:
                print("--------------------------- TWO CATEGORIES")
            info["invalid"] = INVALID_FILENAME
        info["category"] = "healthy"
    if sus_match:
        if info["category"]:
            if print_erros:
                print("--------------------------- TWO CATEGORIES")
            info["invalid"] = INVALID_FILENAME
        info["category"] = "sus"
    if cly_match:
        if info["category"]:
            if print_erros:
                print("--------------------------- TWO CATEGORIES")
            info["invalid"] = INVALID_FILENAME
        info["category"] = "cly"
    if info["category"] == "":
        if print_erros:
            print("---------------------- ERROR: no category in file " + pdf)

    return info


# -----------------------------------------------------------------------------

def copy_pdf_with_new_name(pdf, info):

    shutil.copy2(pdf, info_to_name(info)+ ".pdf")
# -----------------------------------------------------------------------------

def change_pdf_name_from_info(pdf, info):
    os.rename(pdf, info_to_name(info) + ".pdf")
# -----------------------------------------------------------------------------
def change_file_name(pdf, new):
    os.rename(pdf, new)
# -----------------------------------------------------------------------------

def enumerate_catagories_write_to_txt_file(serials_and_sides):
    pdfs = [os.path.basename(x) for x in glob.glob(new_data_in_pdf + "/*.pdf")]

    duplicates = set()



    serial_to_new_name_file = open(new_data_in_pdf + "/serial_to_new_filename.txt", "w")

    catagory_indexes = {"kc": 60, "healthy": 104, "sus": 1, "cly": 1}

    for i, pdf in enumerate(pdfs):


        info = get_all_info_from_filename(pdf)
        copy_pdf_with_new_name(pdf, info)

        if (info["serial"], info["side"]) in serials_and_sides:
            duplicates.add(info_to_name(info))
            print("!!!!!!!!!!!!! ERROR " +  info["serial"] + "_" + info["side"] + " already exist")
            continue

        serials_and_sides.append((info["serial"], info["side"]))


        filename = info["category"] + "_" + str(catagory_indexes[info["category"]])

        catagory_indexes[info["category"]] += 1

        serial_to_new_name_file.write(info["serial"] + "_" + info["side"] + " -> " + filename + "\n")

        print("--- DUPLICATES")
        for dup in duplicates:
            print(dup)

    serial_to_new_name_file.close()
# ----------------------------------------------------------------------------------------------

def check_duplicates(file_names, ignore_invalid_file_names=False):
    """
    Check for duplicates in the folder. Duplicates are images with same patient serial and same side.
    :param file_names: list of file names
    :param ignore_invalid_file_names: Wether or not to ignore invalid file names (no serial or side).
    :return:
    """
    duplicates = set()

    serials_counter = {}
    serials_and_sides = []

    for i, name in enumerate(file_names):

        try:
            info = get_all_info_from_filename(name, print_stuff=False, print_erros=False)
        except:

            if not ignore_invalid_file_names:
                print("Couldn't get info from file " + name + " aborting")
                return
            else:
                print("skipping file " + name)
                continue

        if ignore_invalid_file_names:
            if info['serial'] == "":
                continue
        else:
            if info['serial'] == "":
                print("No Serial in file ", name)
                return



        if not ignore_invalid_file_names and info['side'] == "":
            print("Invalid file name. No Side ", name)
            return

        # if there's a side - check for serial + side duplicate (better)
        if info['side'] != "":

            if (info["serial"], info["side"]) in serials_and_sides:
                duplicates.add(info_to_name(info))
                print("!!!!!!!!!!!!! ERROR " +  info["serial"] + "_" + info["side"] + " already exist")
                continue

            serials_and_sides.append((info["serial"], info["side"]))

        # if side doesn't exist for checking serial + size. count the serial
        else:
            serial_count = serials_counter.get(info['serial'], 0)
            if serial_count > 2:
                print("Error more than 2 serials for " + name)

            serials_counter[info['serial']] = serial_count+1


    if duplicates:
        print("--- DUPLICATES")
        for dup in duplicates:
            print(dup)
        return True
    return False
# ------------------------------------------------------------------------------------------------


def change_pdfs_names(pdfs):

    for pdf in pdfs:
        info = get_all_info_from_filename(pdf)
        change_pdf_name_from_info(pdf, info)

        

# ------------------------------------------------------------------------------------------------

def add_dunder_score_after_patinet_name_in_filename(pdfs):
    for pdf in pdfs:
        info = get_all_info_from_filename(pdf)
        print(pdf)

        side_index = re.search(r"[^a-zA-Z]" + info["side"] + r"[^a-zA-Z]", pdf).span()[0] # get match start index span() = (begin index, end index)
        catagory_index = re.search(r"[^a-zA-Z]" + info["category"] + r"[^a-zA-Z]", pdf).span()[0]

        if side_index < catagory_index:
            after_name_underscore_index = pdf.find("_" + info["side"] + "_")

        else:
            after_name_underscore_index = pdf.find("_" + info["category"] + "_")

        if after_name_underscore_index == -1:
            print("Error3")
            return

        new_name = pdf[:after_name_underscore_index+1] + "_" + pdf[after_name_underscore_index+1:]

        change_file_name(pdf, new_name)


# ------------------------------------------------------------------------------------------------

def normalize(pdfs):
    for pdf in pdfs:
        newname = re.sub(r" ", "_", pdf)
        newname = re.sub(r"_+", "_", newname)
        newname = re.sub(r"^_+", "", newname)
        newname = re.sub(r"_+\.", ".", newname)
        newname = newname.lower()

        serial = re.findall(r"p\d+", newname)[0]
        newname = newname.replace(serial, 'P' + serial[1:]) # restore capital P in serial
        newname = newname.replace("suspect", "sus").replace("normal", "healthy")




        change_file_name(pdf, newname)

# ------------------------------------------------------------------------------------------------

def create_serial_to_patientname_file(pdfs):

    file = open("serial_to_patientname.txt", "w")

    for pdf in pdfs:
        info = get_all_info_from_filename(pdf)
        print(pdf)
        if not re.match(r"^[a-z]+_[a-z]+.*__", pdf):
            print("ERROR")
            return
        patient_name = pdf[:pdf.find("__")]
        file.write(info["serial"] + " " + patient_name + "\n")
    file.close()

# ------------------------------------------------------------------------------------------------

def normalize_fields_order_in_filename(pdfs):
   for pdf in pdfs:
       info = get_all_info_from_filename(pdf, print_stuff=False)

       if info["invalid"]:
           newname = INVALID_FILENAME + "_" + pdf
       else:
           newname = info["name"] + "__" + info["category"] + "_" + info["side"] + "_" + info['serial'] + ".pdf"
       print(newname)

       change_file_name(pdf, newname)



# ------------------------------------------------------------------------------------------------

def create_csv(pdfs):

    table = open("table.csv", "w")
    table.write("#,serial,category,side,name\n")
    for i,pdf in enumerate(pdfs):
        info = get_all_info_from_filename(pdf)
        table.write(str(i) + "," + info['serial'] + "," + info['category'] + "," + info["side"] + "," + info["name"] + "\n")

    table.close()
# ------------------------------------------------------------------------------------------------
import sys
def pdfs_to_jpgs(pdfs):
    for pdf in pdfs:
        cmd = "convert -quality 100 " + pdf +  " " + pdf.replace(".pdf", ".jpg")
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove(pdf)

# ------------------------------------------------------------------------------------------------
from PIL import Image
def center_crop_image(image, new_width, new_height):

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

    # or
    # bottom += ydiff
    # right += xdiff
    # or any other combination

    if was_transposed:
        return image[top:bottom, left:right].transpose(2, 0, 1) # (height, width, chan) -> (chan, height, width)
    else:
        return image[top:bottom, left:right]

# -------------------------------------------------------------------------------------------------------

def crop_to_256_256(jpgs):
    new_width = new_height = 256
    for jpg in jpgs:

        image = img_as_float(io.imread(jpg, as_grey=False)).astype(np.float32) # open like caffe.io.load_image

        io.imsave(jpg, center_crop_image(image, 256, 256))

# ------------------------------------------------------------------------------------------------

def delete_patient_name_from_filename_after_finished(files):
    for file in files:
        newname = file[file.find("__")+2:]
        print(newname)
        change_file_name(file, newname)

# ------------------------------------------------------------------------------------------------

def get_info_from_xlsx_line(patient_info, print_stuff=True, print_errors=True):

    if print_stuff:
        print(patient_info)

    info = {"serial": "", "side": "", "name": "", "invalid": ""}

    # patient_name = patient_info[:patient_info.find("__")]
    # info["name"] = patient_name
    #

    if len(re.findall(r".*P\d+", patient_info)) > 1:
        if print_errors:
            print("----------------------- MULTIPLE SERIALS")
        info["invalid"] = INVALID_FILENAME
    else:
        try:
            info["serial"] = re.search(r".*(P\d+)", patient_info).group(1)
        except:
            pass

    if not info["serial"]:
        print("---------------------------------- NO SERIAL")
        info["invalid"] = INVALID_FILENAME

    right_seperator = r"[^a-zA-Z0-9]"
    left_seperator = r"(^|_| )"
    os = re.search(r"({ls}os{rs}|{ls}left{rs}|{ls}l{rs})".format(rs=right_seperator, ls=left_seperator), patient_info.lower())
    if os is not None:
        info["side"] = "os"
    if re.search(r"({ls}od{rs}|{ls}right{rs}|{ls}r{rs})".format(rs=right_seperator, ls=left_seperator), patient_info.lower()) is not None:
        if info["side"] == "os":
            if print_errors:
                print("-------------------------------------- TWO SIDES IN FILE NAME")
            info["invalid"] = INVALID_FILENAME
        info["side"] = "od"
    if info["side"] == "":
        if print_errors:
            print("------------------------------------------------ NO SIDE")
        info["invalid"] = INVALID_FILENAME

    if print_stuff:
        print(info)

    return info
# ------------------------------------------------------------------------------------------------


def read_serial_new_num_from_xlsx(fname):
    # Open the workbook
    xl_workbook = xlrd.open_workbook(fname)

    # List sheet names, and pull a sheet by name
    #
    sheet_names = xl_workbook.sheet_names()
    print('Sheet Names', sheet_names)
    out_file = open("old_names_with_num_to_serial_name", "w")
    from xlrd.sheet import ctype_text
    for sheet_name in sheet_names:
        xl_sheet = xl_workbook.sheet_by_name(sheet_name)

        for row_i in range(xl_sheet.nrows):

            if row_i < 2:
                continue

            new_num_cell_obj = xl_sheet.cell(row_i, 2)
            new_num = new_num_cell_obj.value

            if not new_num:
                continue


            patient_info_cell_obj = xl_sheet.cell(row_i, 1)
            patient_info =  patient_info_cell_obj.value
            # print(patient_info, new_num)

            info = get_info_from_xlsx_line(patient_info, print_stuff=False, print_errors=False)

            img_f = sheet_name + "_" + str(new_num) + ".jpg"
            if not os.path.exists(img_f):
                print("No img ", img_f, " skipping")
                continue

            if info['serial'] == "":
                print("No serial. skipping")
                continue

            side = "_" + info['side'] if info['side'] != "" else ""
            new_name = sheet_name + side + "_" + info['serial'] + ".jpg"

            print("img ", img_f, " -> ", new_name)
            out_file.write(img_f + " -> " + new_name + "\n")
            change_file_name(img_f, new_name)





    out_file.close()
# ------------------------------------------------------------------------------------------------

def read_entire_xlsx(fname):

    # Open the workbook
    xl_workbook = xlrd.open_workbook(fname)

    # List sheet names, and pull a sheet by name
    #
    sheet_names = xl_workbook.sheet_names()
    print('Sheet Names', sheet_names)

    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

    # Or grab the first sheet by index
    #  (sheets are zero-indexed)
    #
    xl_sheet = xl_workbook.sheet_by_index(0)
    print('Sheet name: %s' % xl_sheet.name)

    # Pull the first row by index
    #  (rows/columns are also zero-indexed)
    #
    row = xl_sheet.row(0)  # 1st row

    # Print 1st row values and types
    #
    from xlrd.sheet import ctype_text

    print('(Column #) type:value')
    for idx, cell_obj in enumerate(row):
        cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
        print('(%s) %s %s' % (idx, cell_type_str, cell_obj.value))

    # Print all values, iterating through rows and columns
    #
    num_cols = xl_sheet.ncols  # Number of columns
    for row_idx in range(0, xl_sheet.nrows):  # Iterate through rows
        print('-' * 40)
        print('Row: %s' % row_idx)  # Print row number
        for col_idx in range(0, num_cols):  # Iterate through columns
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            print('Column: [%s] cell_obj: [%s]' % (col_idx, cell_obj))

# ------------------------------------------------------------------------------------------------

def info_names_to_index_names():
    # ----------------- part 1 - write to files ----------------

    # to_indexes_for_excel = open("info_names_to_index_names_for_excel.txt", "w")
    # to_indexes_for_internal = open("info_names_to_index_names_for_internal.txt", "w")
    # # old_names = open("old_names_with_num_to_serial_name", "r").read()
    # serial_to_name_lines = open("serial_to_patientname.txt", "r").readlines()
    #
    #
    # jpgs = [os.path.basename(x) for x in glob.glob(data_copy_for_changes + "/*.jpg")]
    # jpgs.sort()

    # resume from current indexes
    # indexes = {"healthy": 104,
    #            "kc": 60,
    #            "sus": 31,
    #            "cly": 1}
    #
    # for jpg in jpgs:
    #     # if jpg in old_names:
    #     #     continue
    #     if jpg.replace(".jpg", "").split("_")[1].isdigit():
    #         continue
    #     else:
    #         info = get_all_info_from_filename(jpg, print_stuff=False, print_erros=False)
    #         new_name = info["category"] + "_" + str(indexes[info["category"]]) + ".jpg"
    #         to_indexes_for_internal.write(jpg + " -> " + new_name + "\n")
    #         for serial_name in serial_to_name_lines:
    #             if info["serial"] in serial_name:
    #                 name = serial_name.split(" ")[1].strip()
    #                 xls_string = name.replace("_", " ").title() + " " + "("+info["serial"]+")"
    #                 xls_string += " R" if info["side"]=="od" else " L"
    #                 xls_string = xls_string.ljust(60)
    #                 xls_string += str(indexes[info["category"]])
    #                 print(xls_string)
    #                 to_indexes_for_excel.write(xls_string + "\n")
    #                 break
    #         indexes[info["category"]] += 1
    # to_indexes_for_excel.close()
    # to_indexes_for_internal.close()


    # ------------- part 2 - change file names ------------------
    #
    # to_indexes_for_internal = open("info_names_to_index_names_for_internal.txt", "r")
    # for line in to_indexes_for_internal.readlines():
    #     old_name, new_name = line.split(" -> ")
    #     old_name = old_name.strip()
    #     new_name = new_name.strip()
    #     if not path.exists(data_copy_for_changes + "/" + old_name):
    #         print("error! " + old_name + " doesn't exist")
    #     shutil.move(data_copy_for_changes + "/" + old_name, data_copy_for_changes + "/" + new_name)
    # to_indexes_for_internal.close()


    # ----------------- part 3 - change all file names in project -----------------
    #
    # with open("info_names_to_index_names_for_internal.txt", "r") as to_indexes_for_internal:
    #     to_indexes_for_internal_lines = to_indexes_for_internal.readlines()
    #
    # for root, dirnames, filenames in os.walk(my_model_path):
    #     for filename in filenames:
    #         if filename.endswith(('.txt', '.log')):
    #             # shutil.copy2(root + "/" + filename, root + "/" + filename + ".bak")
    #             # continue
    #
    #
    #             for line in to_indexes_for_internal_lines:
    #
    #                 old_name, new_name = line.split(" -> ")
    #
    #                 old_name = old_name.strip()
    #                 new_name = new_name.strip()
    #
    #                 with open(root + "/" + filename) as f:
    #                     s = f.read()
    #                     if old_name not in s:
    #                         print
    #                         '"{old_name}" not found in {filename}.'.format(**locals())
    #                         continue
    #
    #
    #                 # Safely write the changed content, if found in the file
    #                 with open(root + "/" + filename, 'w') as f:
    #                     print
    #                     'Changing "{old_name}" to "{new_name}" in {filename}'.format(**locals())
    #
    #                     ws_trail_pad = " " *(len(old_name) - len(new_name)) # don't ruin indentation
    #                     s = s.replace(old_name, new_name + ws_trail_pad)
    #                     f.write(s)

    # ----------------- part 4 - change file names in all pictures --------------
    # with open(data_info_files + "/info_names_to_index_names_for_internal.txt", "r") as to_indexes_for_internal:
    #     to_indexes_for_internal_lines = to_indexes_for_internal.readlines()
    #
    #
    # for line in to_indexes_for_internal_lines:
    #
    #     old_name, new_name = line.split(" -> ")
    #
    #     old_name = old_name.strip()
    #     new_name = new_name.strip()
    #
    #     if not path.exists("/home/gil/Caffe/FinalProject/data/All Images/all/" + old_name):
    #         print("error with " + old_name)
    #         return
    #     shutil.move("/home/gil/Caffe/FinalProject/data/All Images/all/" + old_name, "/home/gil/Caffe/FinalProject/data/All Images/all/" + new_name)
# ------------------------------------------------------------------------------------------------

def main():
    # os.chdir(data_copy_for_changes)
    # info_names_to_index_names()



    # read_serial_new_num_from_xlsx(path.join(data_copy_for_changes, "patients_info_changed_by_me.xlsx"))


    # pdfs = [os.path.basename(x) for x in glob.glob(new_data_in_pdf + "/*.pdf")]
    # jpgs = [os.path.basename(x) for x in glob.glob(data_copy_for_changes + "/*.jpg")]
    # pdfs.sort()
    # jpgs.sort()
    #
    # delete_patient_name_from_filename_after_finished(jpgs)
    # normalize(pdfs)
    # add_dunder_score_after_patinet_name_in_filename(pdfs)
    # create_serial_to_patientname_file(pdfs)


    # normalize_fields_order_in_filename(pdfs)


    # if check_duplicates(jpgs, ignore_invalid_file_names=True):
    #    print("dulicates found")

    # prepend_patient_names_from_txt_file_to_files(pdfs)
# ------------------------------------------------------------------------------------------------







if __name__=="__main__":
    main()




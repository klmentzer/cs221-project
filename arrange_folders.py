'''
@author Kaleigh Mentzer

Script to create test, validation, and training folders and move the data into
those folders.

NOTE: this script will move folders, while I've tried to make it safe, probably
a good idea to try running 'segment_data.py' to make sure the file architecture
you have is similar to mine/this won't break things.

Segments 300 images to test and validation, the remaining 2400 images are moved
to training.
'''

import os
import shutil

parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/')
letter_folders = list(set(os.listdir(parent_dir)))
letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]

# make test directory
test_dir = parent_dir+"test"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# make validation directory
val_dir = parent_dir+"val"
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# make validation directory
train_dir = parent_dir+"train"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)


for letter in letter_folders:
    letter_test = parent_dir+"test/"+letter
    letter_val = parent_dir+"val/"+letter
    letter_train = parent_dir+"train/"+letter

    # make test data set
    if not os.path.exists(letter_test):
        os.makedirs(letter_test)

    for i in range(1,301):
        old_list = [parent_dir,letter,"/",letter+str(i)+'.jpg']
        old_path = "".join([str(s) for s in old_list])
        if not os.path.isfile(old_path):
            break
        new_list = [letter_test, "/",letter+str(i)+'.jpg']
        new_path= "".join([str(s) for s in new_list])
        os.rename(old_path,new_path)

    # make validation data set
    if not os.path.exists(letter_val):
        os.makedirs(letter_val)

    for i in range(301,601):
        old_list = [parent_dir,letter,"/",letter+str(i)+'.jpg']
        old_path = "".join([str(s) for s in old_list])
        if not os.path.isfile(old_path):
            break
        new_list = [letter_val, "/",letter+str(i)+'.jpg']
        new_path= "".join([str(s) for s in new_list])
        os.rename(old_path,new_path)

    # move remaining data to training set
    old_path = parent_dir+letter+'/'
    if os.path.exists(old_path):
        shutil.move(old_path,parent_dir+"train/")

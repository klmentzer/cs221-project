import os
import re
import random

def make_training_and_validation():
    '''
    Method to get training and validation data sets. Takes first 2700 images of
    each letter for training, and last 300 for verification.

    @pre Assumes kaggle data file folder is in the parent directory of
         cs221-project repository.

    @return list of training image file paths, list of verification image file paths
    '''
    filename =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/')

    letter_folders = list(set(os.listdir(filename)))
    letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]
    assert(len(letter_folders)==29)

    training_imgs =[]
    for letter in letter_folders:
        for i in range(1,2701):
            path = [os.getcwd(), '../asl-alphabet/asl_alphabet_train/',letter, \
                "/",letter+str(i)+'.jpg']
            full_path = "".join([str(s) for s in path])
            training_imgs.append(full_path)

    verification_imgs =[]
    for letter in letter_folders:
        for i in range(2700,3001):
            path = [os.getcwd(), '../asl-alphabet/asl_alphabet_train/',letter, \
                "/",letter+str(i)+'.jpg']
            full_path = "".join([str(s) for s in path])
            verification_imgs.append(full_path)

    return training_imgs, verification_imgs

def get_category_from_filename(filename):
    '''
    Method to get image category (i.e. the letter) from an image filename
    '''
    img_name = filename.split("/")[-1]
    letter = re.split('\d+', img_name)[0]
    return letter

t,v = make_training_and_validation()
let = get_category_from_filename(random.choice(t))
print(let)

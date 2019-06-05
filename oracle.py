import matplotlib.pyplot as plt
import random
import os
from segment_data import get_category_from_filename


root =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/train')
letter_folders = list(set(os.listdir(root)))
letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]

results = []

for i in range(20):
    letter = random.choice(letter_folders)
    letter_dir = os.listdir(os.path.join(root,letter))
    img = random.choice(letter_dir)
    img_path = os.path.join(root,letter,img)
    print(img_path)
    fig = plt.figure()
    img_data = plt.imread(img_path)
    plt.imshow(img_data)
    plt.show(block=False)
    a = input()
    results.append([get_category_from_filename(img), a])
    fig.clf()
    plt.close(fig)

print(results)

right = 0
for r in results:
    if r[0].upper() == r[1].upper():
        right+=1
print("your accuracy was {}/20, or {}%".format(right,right/.20))

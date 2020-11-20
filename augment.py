import transforms as tr
import random
from torchvision import transforms
import os
from skimage import io as skio
import skimage.color as skc
import pandas as pd
from dataset import FaceLandmarksDataset
from torch.utils.data import Dataset, DataLoader
import asf_read as reader
import numpy as np
# take a dataset, augment it, and write it back into an asf

train_dataset = FaceLandmarksDataset(root_dir='imm_face_db/training/')
test_dataset = FaceLandmarksDataset(root_dir='imm_face_db/testing/')
batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

available_transformations = {
    'rotate': tr.Rotate(),
    'jitter': tr.Jitter(),
    'horizontal_shift': tr.HorizontalShift(),
    'vertical_shift': tr.VerticalShift()
}

def generate_transforms():
    num_to_apply = random.randint(0,2)
    num_applied = 0
    transformation_list = []

    # create a list of transformations
    while (num_applied < num_to_apply):
        key = random.choice(list(available_transformations))
        transformation = available_transformations[key]
        transformation_list.append(transformation)
        num_applied += 1

    return transformation_list

def transform_sample(sample, transformation_list):
    composed = transforms.Compose(transformation_list)
    sample['image'] = np.clip(sample['image'], 0, 1)
    sample = composed(sample)
    return sample

def all_landmarks(path):
    df_dict = {}
    #fixme need path
    #path = "imm_face_db/training/"
    for file in os.listdir(path):
        name, extension = os.path.splitext(file)
        if (extension == '.asf'):
            image = skio.imread(path+name+".jpg")
            image = skc.rgb2gray(image)
            df = create_df(path+file)
            df_dict[name] = df
    return df_dict

def create_df(file_name):
    data_list = []
    asf = open(file_name, "r")
    lines = asf.readlines()
    num_points = int(lines[9])

    for i in range(16, num_points + 16):
        lines[i].rstrip()
        data = lines[i].rstrip().split("\t")

        # get only col 2 and 3
        data = data[2:4]

        # some files have extra columns
        # if len(data) > 7:
        #     data = data[:7]

        data_list.append(data)

    df = pd.DataFrame.from_records(data_list)
    #df.columns = ["path_num", "type", "x", "y", "point_num", "connect_from", "connect_to"]
    df.columns = ["x","y"]
    df = df.apply(pd.to_numeric)
    return df

def save_sample_asf(num, sample):
    # change path to training or testing depending on where we want to add augmentations
    asf_name = "augmented_db/training/" + str(num*193)+".asf"
    jpg_name = "augmented_db/training/" + str(num*193)+".jpg"

    image = sample['image']
    landmarks = sample['landmarks']

    skio.imsave(jpg_name, image)

    file = open(asf_name, 'w+')
    file.write('######################################################################\n'  +
     '#\n'  + '#    AAM Shape File  -  written: Wednesday March 07 - 2001 [11:00]\n'  +
     '#\n'  + '######################################################################\n'  +
     '\n'  + '#\n'  + '# number of model points\n'  + '#\n'  + '58\n'  +'\n'  + '#\n'  +
     '# model points\n'  + '#\n'  + '# format: <path#> <type> <x rel.> <y rel.> <point#> <connects from> <connects to>\n'  +
     '#\n')

    for index, row in landmarks.iterrows():
        line = [0]*7
        line[2] = row["x"]
        line[3] = row["y"]

        converted_list = [str(element) for element in line]
        joined_string = "\t".join(converted_list)
        file.write(joined_string+"\n")

    file.close()


def main():

    num = 1

    for sample in train_dataset:
        print(num)
        transformation_list = generate_transforms()
        transformation_list.append(tr.BW())
        composed = transforms.Compose([tr.Resize(240, 180)])
        sample = composed(sample)

        sample = transform_sample(sample, transformation_list)
        save_sample_asf(num, sample)

        reader.show_landmarks_sample(sample)
        num+=1

if __name__ == "__main__":
    main()

# later when reading these images in again do tr.BW, tr.Resize, tr.toTensor
# create a dataloader of all the images
# for each image, run generate transforms and save the sample
# FIRST RUN ON TRAINING DATA
# THEN ON TESTING
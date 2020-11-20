import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio, transform
import skimage.color as skc
from torchvision import transforms, utils

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

def get_xy(df):
    # if isinstance(df, np.ndarray):
    #     landmarks = df[:, [2,3]]
    landmarks = df[df.columns[2:4]].values.tolist()
    landmarks = np.array(landmarks)
    return landmarks

def adjust_shape(df, image):
    height, width = image.shape
    df["x"] = width * df["x"]
    df["y"] = height * df["y"]
    return df

# show predicted and actual landmarks
def show_model_landmarks(img_num, prediction, landmarks, image):
    h, w = image.shape
    prediction = np.array(prediction)
    landmarks = np.array(landmarks)

    #print(prediction)

    # predicted: red, actual: green

    plt.imshow(image, cmap='gray')
    plt.scatter(prediction[:, 0] * w, prediction[:, 1] * h, s=100, marker='.', c='r')
    plt.scatter(landmarks[:, 0] * w, landmarks[:, 1] * h, s=100, marker='.', c='g')
    plt.pause(0.001)
    plt.savefig(str(img_num)+".jpg")
    plt.show()


def show_landmarks(df, image):

    h, w = image.shape[:2]
    landmarks = np.array(df)

    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0]*w, landmarks[:, 1]*h, s=70, marker='.', c='lightskyblue')
    plt.pause(0.001)
    plt.show()

def show_landmarks_sample(sample):
    image = skc.rgb2gray(sample['image'])
    landmark = sample['landmarks']
    show_landmarks(landmark, image)

# helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples"""
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)

    for i in range(batch_size):
        show_landmarks(landmarks_batch[i], images_batch[i])


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

def all_noses(path):
    df_dict = {}
    #fixme need path
    #path = "imm_face_db/training/"
    for file in os.listdir(path):
        name, extension = os.path.splitext(file)
        if (extension == '.asf'):
            image = skio.imread(path + name + ".jpg")
            image = skc.rgb2gray(image)

            df = create_df(path + file)
            df = df.iloc[-5].to_frame().T
            #df = adjust_shape(df, image)
            df_dict[name] = df
    return df_dict
from torch.utils.data import DataLoader
from dataset import FaceLandmarksDataset
from torchvision import transforms
import transforms as tr
import sys
import numpy as np
import networks
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import asf_read as reader

'''
augmented_db contains 2 copies of each image: 1 is the original and the other has 
randomly applied augmentations (rotation, shifting, color jittering)
'''
transformed_dataset = FaceLandmarksDataset(root_dir='augmented_db/training/',
                                     transform=transforms.Compose([tr.BW(),
                                         tr.Resize(160, 120),
                                         tr.ToTensor()
                                     ]))

train_loader = DataLoader(transformed_dataset, batch_size=8, shuffle=True, num_workers=0)

test_dataset = FaceLandmarksDataset(root_dir='augmented_db/testing/',
                              transform=transforms.Compose([tr.BW(),
                                  tr.Resize(160, 120),
                                  tr.ToTensor()
                              ]))

valid_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)
num_epochs = 25

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()

def plot_loss(training, validation):
    epoch = [*range(1, (num_epochs+1), 1)]
    plt.plot(epoch, training, label = "training")
    plt.plot(epoch, validation, label="validation")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_graph.png')
    plt.show()

def landmark_train():
    sample = next(iter(train_loader))
    images = sample['image']
    landmarks = sample['landmarks']

    # shape: batch_size, height, width
    # print(images.shape)
    # print(landmarks.shape)

    torch.autograd.set_detect_anomaly(True)
    network = networks.Net2()
    # network.cpu()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    loss_min = np.inf

    training_loss_list = []
    valid_loss_list = []

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step in range(1, len(train_loader) + 1):
            sample = next(iter(train_loader))
            images = sample['image']
            landmarks = sample['landmarks']

            # images = images.cuda()
            # landmarks = landmarks.view(landmarks.size(0), -1).cuda()
            landmarks = landmarks.view(landmarks.size(0), -1)

            images = images.type(torch.FloatTensor)
            landmarks = landmarks.type(torch.FloatTensor)

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():

            for step in range(1, len(valid_loader) + 1):
                sample = next(iter(valid_loader))
                images = sample['image']
                landmarks = sample['landmarks']

                landmarks = landmarks.view(landmarks.size(0), -1)

                images = images.type(torch.FloatTensor)
                landmarks = landmarks.type(torch.FloatTensor)

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        training_loss_list.append(loss_train)
        valid_loss_list.append(loss_valid)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), 'saved_models/landmarks.pt')
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time() - start_time))
    # uncomment to get loss plots
    plot_loss(training_loss_list, valid_loss_list)

def landmark_test():
    start_time = time.time()

    with torch.no_grad():
        best_network = networks.Net2()
        # best_network.cuda()
        best_network.load_state_dict(torch.load('saved_models/landmarks.pt'))
        best_network.eval()

        #print(len(valid_loader))
        for step in range(len(valid_loader)):
            sample = next(iter(valid_loader))
            images = sample['image']
            landmarks = sample['landmarks']

            # images = images.cuda()
            # landmarks = (landmarks + 0.5) * 224

            # predictions = (best_network(images).cpu() + 0.5) * 224
            #predictions = best_network(images).cpu()
            predictions = best_network(images)
            predictions = predictions.view(-1, 58, 2)

            for img_num in range(len(images)):
                # print(predictions[img_num])
                # print(len(predictions[img_num]))
                im = images[img_num].numpy().squeeze()
                reader.show_model_landmarks(img_num, predictions[img_num], landmarks[img_num], im)

    print('Total number of saved_models images: {}'.format(len(test_dataset)))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))




def part2():
    landmark_train()
    landmark_test()

def main():
    part2()
    #compose_transforms()
    #test_dataloader()

if __name__ == "__main__":
    main()
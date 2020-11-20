Methodology and output are presented on this site: https://inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/proj4/cs194-26-aeg/

Performs nose tip detection and facial landmark detection using convolutional neural networks on the IMM Face Database.

Part 1: nose tip detector 
Part 2: facial landmark detector 
Part 3: training with a larger dataset, and using ResNet18 

main.py runs parts 1 and 2 when their corresponding booleans are set to true in main.py.
part3.ipynb contains all the code for part 3, and can also be found at the google colab link:
https://colab.research.google.com/drive/10E7cDiMcMgP1sEH0fyrmKUAHDXmuAtbi?usp=sharing

The model used on predictions for part 3 is titled part3model.pth

Other notes: 
- dataset.py contains custom data loaders to load the images and nose tip points, or images and facial landmark points into a dataset. Transformations are applied to convert images to B&W, convert pixels to normalized float values between -0.5 and 0.5, and resize images.  
- networks.py contain neural networks used (Net1 was used for nose tips, Net2 was used for all facial landmarks) 
- augment.py was used to apply transformations and augment the dataset for part 2, which were used to created augmented-db. 

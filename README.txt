###################
Prepare dataset:
1: Download the Pascal VOC 2012 dataset at: 
https://deepai.org/dataset/pascal-voc

2: Sort the Pascal dataset so that it works with our code:
python3 xmlParser.py

#Our algorithms each had their own preprocessing methods:
3: Convert  the images to png and resize them for clustering:
python3 convert.py

4: Convert  the images to png and resize them for transfer learning:
python3 convertTFLearning.py

5: Move the ResizedPNGImages and ResizedPNGImagesSmall and ImageSets directories to this directory

#For transferLearning we need to sort the images into the predefined training and validation splits
6: Run:
python3 directoryTrainValSplits.py
#This will generate aeroplane_train and aeroplane_val directories

#For Cluster-Then-Classification we need our data in google drive:
7: Copy ResizedPNGImagesSmall and the Imagesets Folder in the Pascal VOC to your google drive
###################
Run the code:

Neural Networks:
1. Adjust the file paths in Neural_Network_Image_Classification_small_data_set.ipynb
to match your filepaths.
2. Run all blocks for the Neural Network Model
python3 Neural_Network_Image_Classification_small_data_set.ipynb

Cluster-Then-Classification:
1. Open Cluster_Then_Classification_Using_Pascal_dataset.ipynb on google collab
2. Run all blocks on google collab
Cluster-Then-Classification MNIST Proof of concept:
1. Open Cluster_Then_Classification_MNIST_Dataset.ipynb on google collab
2. Run all blocks on google collab

Transfer Learning:
1. Run MobileNet Based Model:
python3 transferlearningMobileNet.py
#Results will be in a file named [time]-AeroplaneResults.txt
2. Run Inception Based Model:
python3 transferlearningInception.py
#Results will be in a file named [time]-AeroplaneResults.txt

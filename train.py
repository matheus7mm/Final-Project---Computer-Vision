# USAGE
# python train.py --dataset "aug dataset" --model models/model1.model --labelbin models/lb1.pickle
# python train.py --dataset "aug dataset" --model models/model2.model --labelbin models/lb2.pickle

# python train.py --dataset "dataset" --model models/model.model --labelbin models/lb.pickle --plot "graphs/plot.png"
# python train.py --dataset "case 1" --model models/model_case1.model --labelbin models/lb_case1.pickle --plot "graphs/case1.png"

# python train.py --dataset "case 2" --model models/model_case2.model --labelbin models/lb_case2.pickle --plot "graphs/case2.png"
# python train.py --dataset "case 3" --model models/model_case3.model --labelbin models/lb_case3.pickle --plot "graphs/case3.png"
# python train.py --dataset "case 4" --model models/model_case4.model --labelbin models/lb_case4.pickle --plot "graphs/case4.png"
# python train.py --dataset "case 5" --model models/model_case5.model --labelbin models/lb_case5.pickle --plot "graphs/case5.png"
# python train.py --dataset "case 6" --model models/model_case6.model --labelbin models/lb_case6.pickle --plot "graphs/case6.png"
# python train.py --dataset "case 7" --model models/model_case7.model --labelbin models/lb_case7.pickle --plot "graphs/case7.png"
# python train.py --dataset "case 8" --model models/model_case8.model --labelbin models/lb_case8.pickle --plot "graphs/case8.png"
# python train.py --dataset "case 9" --model models/model_case9.model --labelbin models/lb_case9.pickle --plot "graphs/case9.png"
# python train.py --dataset "case 10" --model models/model_case10.model --labelbin models/lb_case10.pickle --plot "graphs/case10.png"

# python train.py --dataset "case 11" --model models/model_case11.model --labelbin models/lb_case11.pickle --plot "graphs/case11.png"

# python train.py --dataset "case 12" --model models/model_case12.model --labelbin models/lb_case12.pickle --plot "graphs/case12.png"
# python train.py --dataset "case 13" --model models/model_case13.model --labelbin models/lb_case13.pickle --plot "graphs/case13.png"
# python train.py --dataset "case 14" --model models/model_case14.model --labelbin models/lb_case14.pickle --plot "graphs/case14.png"
# python train.py --dataset "case 15" --model models/model_case15.model --labelbin models/lb_case15.pickle --plot "graphs/case15.png"
# python train.py --dataset "case 16" --model models/model_case16.model --labelbin models/lb_case16.pickle --plot "graphs/case16.png"

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# CNN class
from cnn.smallervggnet import SmallerVGGNet

# import the necessary packages]
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="graphs/plot1.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 400
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] Loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

countImages = 0

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	countImages += 1

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] Number of images: {}".format(countImages))

print("[INFO] Data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state=42, stratify=labels)

# initialize the model
print("[INFO] Compiling model...")

model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))

model.compile(loss="categorical_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the network
print("[INFO] Training network...")

H = model.fit(
	x=trainX, 
	y=trainY, 
	batch_size=BS,
	epochs=EPOCHS)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

# save the model to disk
print("[INFO] Serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] Serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()
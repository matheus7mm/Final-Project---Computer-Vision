from numpy import expand_dims
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import random
import skimage
import scipy

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def saveImage(imagePath, image, augPath, label, iImg, iAug):
	imagePath = augPath + "/" + str(label[0]) + "_" + str(iImg) + "_" + str(iAug) + ".png" 
	cv2.imwrite(imagePath, image)

# -------- Horizontal and Vertical Shift Augmentation --------
def shift(samples, imagePath, augPath, label, iImg, iAug):

	print("[INFO] Saving horizontal and vertical shifting images...")

	datagen = ImageDataGenerator(width_shift_range=[-200,200])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples
	for i in range(6):

		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		newImage = batch[0].astype('uint8')

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	datagen = ImageDataGenerator(height_shift_range=0.5)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)

	# generate samples
	for i in range(6):

		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		newImage = batch[0].astype('uint8')

		saveImage(imagePath, newImage, augPath, l, iImg, iAug)

		iAug += 1

	return iAug, iImg

# -------- Horizontal and Vertical Flip Augmentation --------
def flip(samples, imagePath, augPath, label, iImg, iAug):
	
	print("[INFO] Saving horizontal and vertical flip images...")

	for sample in samples:

		newImage = np.fliplr(sample)

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	for sample in samples:

		newImage = np.flipud(sample)

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ------------------ Rotation Augmentation ------------------
def rotation(samples, imagePath, augPath, label, iImg, iAug):
	
	print("[INFO] Saving random rotation images...")

	datagen = ImageDataGenerator(rotation_range=90)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples
	for i in range(5):

		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		newImage = batch[0].astype('uint8')

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ------------------- Brightness Augmentation -----------------
print("[INFO] Saving random brightness images...")
def brightness(samples, imagePath, augPath, label, iImg, iAug):

	datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples
	for i in range(5):

		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		newImage = batch[0].astype('uint8')

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Zoom Augmentation ----------------
def zoom(samples, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random zoom images...")

	datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples
	for i in range(5):

		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		newImage = batch[0].astype('uint8')

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Crop Augmentation ----------------
def crop(image, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random crop images...")

	height, width, channels = image.shape

	for i in range(5):

		cropLocation = random.randint(1,4)

		cropSizeHeight = random.randint(1, height // 3)
		cropSizeWidth = random.randint(1, width // 3)

		# crop image in the width - left
		if cropLocation == 1:
			newImage = image[1:height, cropSizeWidth:width]
		# crop image in the width - right
		if cropLocation == 2:
			newImage = image[1:height, 1:(width - cropSizeWidth)]
		# crop image in the height - top
		if cropLocation == 3:
			newImage = image[cropSizeHeight:height, 1:width]
		# crop image in the height - bottom
		if cropLocation == 4:
			newImage = image[1:(height - cropSizeHeight), 1:width]

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Gaussian Blur Augmentation ----------------
def gaussianBlur(image, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random gaussian blur images...")

	for i in range(5):

		sigma = random.randint(1, 10)

		newImage = cv2.GaussianBlur(image, (11,11), sigma)

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Gaussian Noise Augmentation ----------------
def gaussianNoise(image, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random gaussian noise images...")

	image = skimage.util.img_as_float(image)

	for i in range(5):

		newImage = skimage.util.img_as_ubyte(skimage.util.random_noise(image, mode="gaussian"))

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Shear Augmentation ----------------
def shear(image, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random shear images...")

	image = skimage.util.img_as_float(image)

	for i in range(5):

		shear = round(random.uniform(0.1, 0.4), 4)

		afine_tf = skimage.transform.AffineTransform(shear=shear)

		newImage = skimage.util.img_as_ubyte(skimage.transform.warp(image, inverse_map=afine_tf))

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# ---------------- Random Scale Augmentation ----------------
def scale(image, imagePath, augPath, label, iImg, iAug):
	print("[INFO] Saving random scale images...")

	image = skimage.util.img_as_float(image)

	for i in range(5):

		sX = round(random.uniform(0.7, 1.3), 4)
		sY = round(random.uniform(0.7, 1.3), 4)

		afine_tf = skimage.transform.AffineTransform(scale=(sX, sY))

		newImage = skimage.util.img_as_ubyte(skimage.transform.warp(image, inverse_map=afine_tf))

		saveImage(imagePath, newImage, augPath, label, iImg, iAug)

		iAug += 1

	return iAug, iImg

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to original dataset (i.e., directory of images)")
ap.add_argument("-a", "--augmentation", type=str, required=True,
	help="path to augmentation dataset (i.e., directory of images)")
ap.add_argument("-c", "--case", type=int, required=True,
	help="number of the case")
args = vars(ap.parse_args())

IMAGE_DIMS = (96, 96, 3)

print("[INFO] Loading dataset...")
imagePaths = list(paths.list_images(args["dataset"]))

# loop over the input images
iImg = 0
for imagePath in imagePaths:

	print("[INFO] ------- IMAGE {} -------".format(iImg + 1))

	# load the image
	image = cv2.imread(imagePath)
	height, width, channels = image.shape
	#image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

	# convert to numpy array
	data = img_to_array(image)
	# expand dimension to one sample
	samples = expand_dims(data, 0)
	
	# extract set of class labels from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2].split("_")

	# path to augmentation dataset for the class
	augPath = args["augmentation"] + "/" + str(label[0])
	print("[INFO] Dir {} created".format(augPath))

	if not os.path.exists(augPath):
		os.mkdir(augPath)	
	
	print("[INFO] Saving original image...")

	imagePath = augPath + "/" + str(label[0]) + "_" + str(iImg) + ".png"
	cv2.imwrite(imagePath, image)

	# i is a counter for augmentation images
	iAug = 0

	if (args["case"] == 1):
		# python augmentation.py --dataset "dataset" --augmentation "case 1" --case 1
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 2):
		# python augmentation.py --dataset "dataset" --augmentation "case 2" --case 2
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 3):
		# python augmentation.py --dataset "dataset" --augmentation "case 3" --case 3
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 4):
		# python augmentation.py --dataset "dataset" --augmentation "case 4" --case 4
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 5):
		# python augmentation.py --dataset "dataset" --augmentation "case 5" --case 5
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = shear(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 6):
		# python augmentation.py --dataset "dataset" --augmentation "case 6" --case 6
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 7):
		# python augmentation.py --dataset "dataset" --augmentation "case 7" --case 7
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianNoise(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 8):
		# python augmentation.py --dataset "dataset" --augmentation "case 8" --case 8
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 9):
		# python augmentation.py --dataset "dataset" --augmentation "case 9" --case 9
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = shear(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 10):
		# python augmentation.py --dataset "dataset" --augmentation "case 10" --case 10
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 11):
		# python augmentation.py --dataset "dataset" --augmentation "case 11" --case 11
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianNoise(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 12):
		# python augmentation.py --dataset "dataset" --augmentation "case 12" --case 12
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 13):
		# python augmentation.py --dataset "dataset" --augmentation "case 13" --case 13
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = shear(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 14):
		# python augmentation.py --dataset "dataset" --augmentation "case 14" --case 14
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianNoise(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 15):
		# python augmentation.py --dataset "dataset" --augmentation "case 15" --case 15
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = shear(image, imagePath, augPath, label, iImg, iAug)
	if (args["case"] == 16):
		# python augmentation.py --dataset "dataset" --augmentation "case 16" --case 16
		iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = rotation(samples, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = gaussianNoise(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)
		iAug, iImg = shear(image, imagePath, augPath, label, iImg, iAug)

	if (args["case"] == 100):
		iAug, iImg = scale(image, imagePath, augPath, label, iImg, iAug)

	#iAug, iImg = flip(samples, imagePath, augPath, label, iImg, iAug)
	# iAug, iImg = crop(image, imagePath, augPath, label, iImg, iAug)
	
	# iAug, iImg = gaussianBlur(image, imagePath, augPath, label, iImg, iAug)

	iImg += 1

cv2.waitKey(0)
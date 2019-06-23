# python result.py --model models/model.model --labelbin models/lb.pickle --examples test

# python result.py --model models/model_case1.model --labelbin models/lb_case1.pickle --examples test
# python result.py --model models/model_case2.model --labelbin models/lb_case2.pickle --examples test
# python result.py --model models/model_case3.model --labelbin models/lb_case3.pickle --examples test
# python result.py --model models/model_case4.model --labelbin models/lb_case4.pickle --examples test
# python result.py --model models/model_case5.model --labelbin models/lb_case5.pickle --examples test
# python result.py --model models/model_case6.model --labelbin models/lb_case6.pickle --examples test
# python result.py --model models/model_case7.model --labelbin models/lb_case7.pickle --examples test
# python result.py --model models/model_case8.model --labelbin models/lb_case8.pickle --examples test
# python result.py --model models/model_case9.model --labelbin models/lb_case9.pickle --examples test
# python result.py --model models/model_case10.model --labelbin models/lb_case10.pickle --examples test
# python result.py --model models/model_case11.model --labelbin models/lb_case11.pickle --examples test

# python result.py --model models/model_case12.model --labelbin models/lb_case12.pickle --examples test
# python result.py --model models/model_case13.model --labelbin models/lb_case13.pickle --examples test
# python result.py --model models/model_case14.model --labelbin models/lb_case14.pickle --examples test
# python result.py --model models/model_case15.model --labelbin models/lb_case15.pickle --examples test
# python result.py --model models/model_case16.model --labelbin models/lb_case16.pickle --examples test

from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-e", "--examples", required=True,
	help="path to input examples (i.e., directory of images)")
args = vars(ap.parse_args())

IMAGE_DIMS = (96, 96, 3)

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

print("[INFO] loading examples of images...")
imagePaths = list(paths.list_images(args["examples"]))

# loop over the input images
iImg = 0
correctNum = 0;
for imagePath in imagePaths:

	# load the image
	image = cv2.imread(imagePath)
	output = image.copy()

	# pre-process the image for classification
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image
	print("[INFO] classifying image...")
	print("Image {}: {}".format(iImg, imagePath))
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	# we'll mark our prediction as "correct" of the input image filename
	# contains the predicted label text (obviously this makes the
	# assumption that you have named your testing image files this way)
	filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
	correct = "correct" if filename.lower().rfind(label.lower()) != -1 else "incorrect"

	if correct == "correct":
		correctNum += 1

	# show the output image
	iImg += 1
	print("[INFO] {}".format(label))
# show the result
print("[INFO] Number of correct predictions: {}".format(correctNum))

accuracy = correctNum / iImg

print("[INFO] Accuracy: {}".format(accuracy))
cv2.waitKey(0)
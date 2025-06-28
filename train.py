# USAGE
# python3 train.py --dataset datasets-generate --model model/plastic.model --label-bin model/plastic_lb.pickle --plot model/plastic_plot.png

# Set matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")

# Import required packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, RepeatVector, Masking, TimeDistributed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from imutils import paths
from cv2 import cv2

# import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Build argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Initialize data and labels
print("[INFO] loading images...")
data = []
labels = []

# Get image paths and shuffle them randomly
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# Loop over input images
for imagePath in imagePaths:
	# Load the image, resize to 32x32 pixels (ignore aspect ratio),
	# flatten the image to 32x32x3 = 3072 pixels into a list,
	# and store the image in the data list
	print("[INFO] Preproccessing image: ", imagePath)
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)

	# Extract class label from image path and update the labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Scale raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split data into training and testing sets using 75% for training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# Convert labels from integers to vectors (for binary 2-class classification,
# here we use 'to_categorical' because scikit-learn's LabelBinarizer does not return vectors)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Backpropagation
# Define the 3072-1024-512-7 architecture using Keras
# The input layer and first hidden layer are defined at line 91.
# It will have input_shape 3072 because there are 32x32x3 = 3072 pixels in the flattened input image.
# The first hidden layer will have 1024 nodes.
# The second hidden layer will have 512 nodes (line 92).
# Finally, the number of nodes in the output layer (line 93) will be the number of possible class labels.
# In this case, the output layer will have 7 nodes (HDPE, LDPE, Other, PET, PP, PS, PVC).
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# Initialize the initial learning rate and number of epochs/iterations for training
INIT_LR = 0.01
EPOCHS = 75

# cross-entropy loss (menggunakan binary_crossentropy untuk klasifikasi 2 kelas)
# Compile the model using SGD as the optimizer and categorical cross-entropy loss
# (use binary_crossentropy for 2-class classification)
# where SGD is an optimization library that uses backpropagation
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Plot training loss and accuracy
N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["accuracy"], label="train_acc")
# plt.plot(N, H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy (Backpropagation NN)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig(args["plot"])

# Save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
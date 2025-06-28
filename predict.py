# USAGE
# python3 predict.py --image testing/pet3.jpg --model model/plastic.model --label-bin model/plastic_lb.pickle --width 32 --height 32 --flatten 1

# Import required packages
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from cv2 import cv2
import argparse, pickle, os, uuid, json, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_time 		= time.time()
predict_token 	= str(uuid.uuid4())
save_token		= str(uuid.uuid4())
save_place 		= 'detection-results/' + predict_token + '/'
save_name 		= save_place + save_token + '.jpg'
arr_data 		= {}
binarizer_data	= []

# Create directory for storing output images
os.makedirs(save_place)

# Build argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we will classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
	help="target spatial width dimension")
ap.add_argument("-e", "--height", type=int, default=28,
	help="target spatial height dimension")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="option to flatten input image")
args = vars(ap.parse_args())

# Load input image and resize it to target spatial dimensions
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# Scale pixel values to [0, 1]
image = image.astype("float") / 255.0

# Check if we need to flatten the image and add batch dimension
if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

# If not, we need to work with CNN - don't flatten the image, just add batch dimension
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# Load model and label binarizer
# print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# Make prediction on the image
preds = model.predict(image)

# Find the class label index with the largest associated probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# Draw class label + probability on output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite(save_name, output)

# Store binarizer data into array
binarizer_data.append({
	"shape":image.shape,
	"data":image.tolist()})

# Insert required variables into array for display
arr_data['_id']			= predict_token
arr_data['type'] 		= label
arr_data['percentage'] 	= preds[0][i] * 100
# arr_data['binarizer']	= binarizer_data
arr_data['file']		= save_name
arr_data['time_used'] 	= time.time() - start_time

# Display response in JSON format
resJson = json.dumps(arr_data, ensure_ascii=False, sort_keys=False, indent=4, separators=(',', ': '))
print(resJson)
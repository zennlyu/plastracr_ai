import tensorflow as tf
import numpy as np
import cv2 as cv2

model_path = "model/plastic_type.tflite"

# Load tflite model
# interpreter = tf.lite.Interpreter(modelpath="model/plastic_type.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)

# Initialize the model
interpreter.allocate_tensors()

# Get input and output tensors
inputdetails = interpreter.get_input_details()
outputdetails = interpreter.get_output_details()

# Print information of input and output tensors
print("inputdetails:", inputdetails)
print("outputdetails:", outputdetails)

def image_process(image_path):
    image=cv2.imread(image_path)
    image=cv2.resize(image,(224,224))
    image=tf.convert_to_tensor(image)
    image=tf.reshape(image,[1,224,224,3])
    image = tf.cast(image, dtype=np.uint8)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    image = image.eval(session=sess)
    return image


def main():
    # Load the model
    interpreter = tf.lite.Interpreter(model_path="model/plastic_type.tflite")
    interpreter.allocate_tensors()

    # Model input and output details
    input_details = interpreter.get_input_details()
    #print(input_details)
    #[{'name': 'input', 'index': 88, 'shape': array([  1, 224, 224,   3]), 'dtype': <class 'numpy.uint8'>, 'quantization': (0.0078125, 128)}]
    output_details = interpreter.get_output_details()
    #print(output_details)
    #[{'name': 'MobilenetV1/Predictions/Reshape_1', 'index': 87, 'shape': array([   1, 1001]), 'dtype': <class 'numpy.uint8'>, 'quantization': (0.00390625, 0)}]

    image_path='testing/ldpe2.jpg'
    image=image_process(image_path)

    interpreter.set_tensor(input_details[0]['index'], image )#传入的数据必须为ndarray类型
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)

if __name__ == '__main__':
    main()

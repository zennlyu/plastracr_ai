# 导入必要的库
import tensorflow as tf

model_path = "model/plastic_type.tflite"

# 加载 tflite 模型
# interpreter = tf.lite.Interpreter(modelpath="model/plastic_type.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
# 初始化模型
interpreter.allocate_tensors()

# 获取输入和输出张量
inputdetails = interpreter.get_input_details()
outputdetails = interpreter.get_output_details()

# 打印输入和输出张量的信息
print("inputdetails:", inputdetails)
print("outputdetails:", outputdetails)

import tensorflow as tf
import numpy as np
import cv2 as cv2


#图片处理，
def image_process(image_path):
    image=cv2.imread(image_path)
    image=cv2.resize(image,(224,224))
    image=tf.convert_to_tensor(image)
    image=tf.reshape(image,[1,224,224,3])
    image = tf.cast(image, dtype=np.uint8)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    image = image.eval(session=sess)  # 转化为numpy数组
    return image


def main():
    # 加载模型
    interpreter = tf.lite.Interpreter(model_path="model/plastic_type.tflite")
    interpreter.allocate_tensors()
    
    
    # 模型输入和输出细节
    input_details = interpreter.get_input_details()
    #print(input_details)
    #[{'name': 'input', 'index': 88, 'shape': array([  1, 224, 224,   3]), 'dtype': <class 'numpy.uint8'>, 'quantization': (0.0078125, 128)}]
    output_details = interpreter.get_output_details()
    #print(output_details)
    #[{'name': 'MobilenetV1/Predictions/Reshape_1', 'index': 87, 'shape': array([   1, 1001]), 'dtype': <class 'numpy.uint8'>, 'quantization': (0.00390625, 0)}]
    

    #图片传入与处理
    image_path='testing/ldpe2.jpg'
    image=image_process(image_path)

    #模型预测
    interpreter.set_tensor(input_details[0]['index'], image)#传入的数据必须为ndarray类型
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)

if __name__ == '__main__':
    main()

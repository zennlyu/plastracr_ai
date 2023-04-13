# plastracr_ai
back-propagation algorithms for plastic type identification

- 使用命令安装所需库

  ```
  pip3 install -r requirements.txt
  ```

- 通过运行 augmentation.php 程序来解决数据集的限制问题。

      php augmentation.php

- 使用 train.py 训练

      python3 train.py --dataset datasets-generate --model model/plastic.model --label-bin model plasticlb.pickle --plot model/plasticplot.png

- 使用 predict.py 程序进行模型预测

      python3 predict.py --image testing/pet3.jpg --model model/plastic.model --label-bin model/plasticlb.pickle --width 32 --height 32 --flatten 1

- predict.py 程序的预测结果将以 JSON 响应的形式呈现，如下所示：

      {
          "id": "0bfab180-abeb-48f2-821a-85d38826b1c9",
          "type": "PET",
          "percentage": 88.48974108695984,
          "file": "detectionresults/0bfab180-abeb-48f2-821a-85d38826b1c9/62bfdf31-68c6-4a96-8ece-3282f96e66e6.jpg",
          "timeused": 1.0790858268737793
      }

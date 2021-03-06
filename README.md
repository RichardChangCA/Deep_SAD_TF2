# Deep_SAD_TF2

Unofficial implementation of 2020 ICLR paper "DEEP SEMI-SUPERVISED ANOMALY DETECTION" TensorFlow 2.0 version

Paper: https://arxiv.org/abs/1906.02694

Official PyTorch Version: https://github.com/lukasruff/Deep-SAD-PyTorch

```python
from google.colab import drive
drive.mount('/content/cifar10')

!python3 -m pip install tensorflow-addons

base_dir = '/content/cifar10/MyDrive/Deep_SAD'

# os.path.join(base_dir, "current_file.txt")
```

Dataset: cifar10 https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data

| Label | Description |
| :---    |   ---: |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

------

All training samples are labeled: labeled normal or labeled abnormal)

Normal class: 5, Abnormal classes: 6,7,8,9,0

abnormal = 0.01 * normal

final AUC_ROC: 0.6566

abnormal = 0.1 * normal

final AUC_ROC: 0.8430

'nu' can be tunned for decision threshold selection --> trade-off precision and recall

Pipeline: train autoencoder --> get center c and initialized encoder --> train deep-sad

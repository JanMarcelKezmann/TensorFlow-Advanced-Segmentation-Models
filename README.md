# TensorFlow Advanced Segmentation Models
A Python Library for High-Level Semantic Segmentation Models.

<p align="center">
 <img src="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/images/tasm%20logo%20big%20white%20bg.PNG" width="700" height="220">
</p>

## Preface
<p>Since the breakthrough of Deep Learning and Computer Vision was always one of the core problems that researcher all over the world have worked on, to create better models every day. One Computer Vision area that got huge attention in the last couple of years is Semantic Segmentation. The task to segment every pixel on a given image led to the invention of many great models starting with the classical U-Net up to now more and more complex neural network structures. But even though many new algorithms were developed, the distribution of easy to use open source libraries which contain High-Level APIs that make the technology accessible for everyone is still far behind the huge amount of research that is published continuously.</p>
<p>Inspired by <a href="https://github.com/qubvel">qubvel's</a> <a href="https://github.com/qubvel/segmentation_models">segmentation_models</a> this repository builds upon his work and extends it by a variety of recently developed models which achieved great results on the <a href="https://www.cityscapes-dataset.com/">Cityscapes</a>, <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">PASCAL VOC 2012</a>, <a href="https://cs.stanford.edu/~roozbeh/pascal-context/">PASCAL Context</a>, <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a> dataset and many more.</p>
<p>The library contains to date 14 different Semantic Segmentation Model Architecters for multi-class semantic segmentation as well as many on imagenet pretrained backbones. An important new feature is the upgrade to Tensorflow 2.x including the use of the advanced model subclassing feauture to build customized segmentation models. Further are now all system platforms compatible with the library this means that tasm can run on Windows, Linux and MacOS as well.</p>

### Main Library Features
- High Level API
- 14 Segmentation Model Architectures for multi-class semantic segmentation
  - **New:** HRNet + OCR Model
- Many already pretrained backbones for each architecture
- Many useful segmentation losses (Dice, Focal, Tversky, Jaccard and many more combinations of them)
- **New:** Models can be used as Subclassed or Functional Model
- **New:** TASM works now on all platforms, i.e. Windows, Linux, MacOS with Intel or Apple Silicon Chips

## Table of Contents

 - [Installation and Setup](#installation-and-setup)
 - [Training Pipeline](#training-pipeline)
 - [Examples](#examples)
 - [Models and Backbones](#models-and-backbones)
 - [Citing](#citing)
 - [License](#license)
 - [References](#references)

## Installation and Setup

<p>To get the repository running just check the following requirements.</p>

**Requirements**
**Windows or Linus**
1) Python 3.6 or higher
2) tensorflow >= 2.3.0 (>= 2.0.0 is sufficient if no efficientnet backbone is used)
3) numpy
4) matplotlib

**MacOS**
1) Python 3.9 or higher
2) tensorflow-macos >= 2.5.0
3) numpy >= 1.21.0
4) matplotlib

<p>Furthermore just execute the following command to download and install the git repository.</p>

**Clone Repository**

    $ git clone https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models.git

or directly install it:<br>
**Pip Install Repository**

    $ pip install git+https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models.git

## Training Pipeline

Please check that **Tensorflow** is installed on your computer.

To import the library just use the standard python import statement:

   
```python
import tensorflow_advanced_segmentation_models as tasm
```
      
Then pick any model backbone from the list below and define weights, height and width:

```python
BACKBONE_NAME = "efficientnetb3"
WEIGHTS = "imagenet"
HEIGHT = 320
WIDTH = 320
```

Load the data

```python
TrainingGenerator, ValidationGenerator = get_data(...)
```

Create the base model that works as backbone for the segmentation model:

```python
base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH)
```

Define a Model and compile it with an appropriate loss:
 
```python
model = tasm.DANet(n_classes=3, base_model=base_model, output_layers=layers, backbone_trainable=False)
model.compile(tf.keras.optimizers.Adam(0.0001), loss=tasm.losses.CategoricalFocalLoss, tasm.metrics.IOUScore(threshold=0.5))
```

If you want to use the Functional Model class define instead:

```python
model = tasm.DANet(n_classes=3, base_model=base_model, output_layers=layers, backbone_trainable=False).model()
model.compile(tf.keras.optimizers.Adam(0.0001), loss=tasm.losses.CategoricalFocalLoss, tasm.metrics.IOUScore(threshold=0.5))
```

Now finally train the model:

```python
history = model.fit(
    TrainingGenerator
    batch_size=8,
    epochs=50,
    validation_data=ValidationGenerator
)
```
 
You can use the fit_generator method too, e.g. if you want to apply augmentations to the data.
For complete training pipelines, go to the <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples">Examples</a> folder

## Examples

- [Jupyter Notebook] Multi-class (3 classes) segmentation (sky, building, background) on CamVid dataset <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples/TASM_Example_1.ipynb">here</a>
- [Jupyter Notebook] Multi-class (11 classes) segmentation on CamVid dataset <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples/TASM_Example_2.ipynb">here</a>
- [Jupyter Notebook] Multi-class (11 classes) segmentation on CamVid dataset with a custom training loop<a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples/TASM_Example_3.ipynb">here</a>
- [Jupyter Notebook] Two-class (2 classes) segmentation on <a href="https://www.tensorflow.org/datasets/catalog/caltech_birds2010">Caltech-Birds-2010</a> dataset <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples/TASM_Example_4.ipynb">here</a>

## Models and Backbones

**Models**

- **<a href="https://arxiv.org/pdf/1411.4038.pdf">FCN</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/FCN.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/abs/1505.04597">UNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/UNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf">FPN</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/FPN.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/abs/1612.01105">PSPNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/PSPNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1606.00915.pdf">DeepLab</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/DeepLab.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1706.05587.pdf">DeepLabV3</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/DeepLabV3.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1802.02611.pdf">DeepLabV3+</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/DeepLabV3plus.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1809.02983.pdf">DANet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/DANet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1809.00916.pdf">OCNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/OCNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Co-Occurrent_Features_in_Semantic_Segmentation_CVPR_2019_paper.pdf">CFNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/CFNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1909.11065.pdf">SpatialOCRNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/SpatialOCRNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- **<a href="https://arxiv.org/pdf/1909.11065.pdf">ASPOCRNet</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/ASPOCRNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- <a href="https://arxiv.org/pdf/1909.09408.pdf">**ACFNet**</a> &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/ACFNet.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
- <a href="https://arxiv.org/pdf/1904.04514.pdf">**HRNet + OCR**</a> &nbsp; <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/tensorflow_advanced_segmentation_models/models/HRNetOCR.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a> **NEW!**
    
**Backbones**
(For Details see <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/tree/master/tensorflow_advanced_segmentation_models/backbones">here</a>.)

|Type         | Names |
|-------------|-------|
|**VGG**          | ``'vgg16' 'vgg19'``|
|**ResNet**       | ``'resnet50' 'resnet50v2' 'resnet101' 'resnet101v2' 'resnet152' 'resnet152v2'``|
|**Xception**     | ``'xception'``|
|**MobileNet**    | ``'mobilenet' 'mobilenetv2'``|
|**NASNet**       | ``'nasnetlarge' 'nasnetmobile'``|
|**DenseNet**     | ``'densenet121' 'densenet169' 'densenet201'``|
|**EfficientNet** | ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' 'efficientnetb6' efficientnetb7'``|
    

    All backbones have weights trained on 2012 ILSVRC ImageNet dataset.
    
**Further Model Information**

A new feature makes it possible to define the model as a Subclassed Model or as a Functional Model instead. To define the model as a Subclassed Model just write: **tasm.UNet** to define the UNet or replace it with any other model. If you want to define the Functional Model instead just append **.model()**, i.e. **tasm.UNet.model()**. This provides further TensorFlow features like saving the model in the "tf" format.

## Citing

    @misc{Kezmann:2020,
      Author = {Jan-Marcel Kezmann},
      Title = {Tensorflow Advanced Segmentation Models},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models}}
    } 
    
## License

Project is distributed under <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/LICENSE">MIT License</a>.

## References
<p>Thank you for all the papers that made this repository possible and especially thank you Pavel Yakubovskiy's initial segmentation models repository.</p>

 - Pavel Yakubovskiy, Segmentation Models, 2019, GitHub, GitHub Repository, https://github.com/qubvel/segmentation_models
 - Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam, Tensorflow Models DeepLab, 2020, GitHub, GitHub Repository, https://github.com/tensorflow/models/tree/master/research/deeplab
 - Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang, Zhiwei and Lu, Hanqing, DANet, 2020, GitHub, GitHub Repository, https://github.com/junfu1115/DANet
 - Yuhui Yuan and Jingdong Wang, openseg.OCN.pytorch, 2020, GitHub, GitHub Repository, https://github.com/openseg-group/OCNet.pytorch
 - Yuhui Yuan and Xilin Chen and Jingdong Wang, openseg.pytotrch, 2020, GitHub, GitHub Repository, https://github.com/openseg-group/openseg.pytorch
 - Fan Zhang, Yanqin Chen, Zhihang Li, Zhibin Hong, Jingtuo Liu, Feifei Ma, Junyu Han, Errui Ding, 2020, GitHub, GitHub Repository, https://github.com/zrl4836/ACFNet
 - Xie Jingyi, Ke Sun, Jingdong Wang, RainbowSecret, 2021, GitHub, GitHub Repository, https://github.com/HRNet/HRNet-Semantic-Segmentation

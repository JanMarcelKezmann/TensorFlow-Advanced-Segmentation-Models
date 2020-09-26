# TensorFlow Advanced Segmentation Models
A Python Library for High-Level Semantic Segmentation Models.

<p align="center">
 <img src="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/images/tasm%20logo%20big%20white%20bg.PNG" width="700" height="220">
</p>

## Preface
<p>Since the breakthrough of Deep Learning and Computer Vision was always one of the core problems that researcher all over the world have worked on, to create better models every day. One Computer Vision area that got huge attention in the last couple of years is Semantic Segmentation. The task to segment every pixel on a given image led to the invetion of many great models starting with the classical U-Net up to now more and more complex neural network structures. But even though many new algorithms were developed, the distribution of easy to use open source libraries which contain High-Level APIs that make the technology accessible for everyone.</p>
<p>Inspired by <a href="https://github.com/qubvel">qubvel's</a> <a href="https://github.com/qubvel/segmentation_models">segmentation_models</a> this repository builds upon his work and extends it by a variety of recently developed models which achieved great results on the <a href="https://www.cityscapes-dataset.com/">Cityscapes</a>, <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">PASCAL VOC 2012</a>, <a href="https://cs.stanford.edu/~roozbeh/pascal-context/">PASCAL Context</a>, <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a> dataset and many more. An important new feature is the upgrade to Tensorflow 2.x including the use of the advanced model subclassing feauture to build customized segmentation models.</p>

### Main Library Features
- High Level API
- 9 Segmentation Model Architectures for multi-class semantic segmentation
- Many already pretrained backbones for each architecture
- Many useful segmentation losses

## Table of Contents

 - [Installation and Setup](#installation-and-setup)
 - [Training Pipeline](#training-pipeline)
 - [Documentation](#documentation)
 - [Change Log](#change-log)
 - [Citing](#citing)
 - [References](#references)

## Installation and Setup

<p>To get the repository running just check the following requirements.</p>

**Requirements**
1) Python 3.6 or higher
2) tensorflow >= 2.3.0 (>= 2.0.0 is sufficient if no efficientnet backbone is used)
3) numpy

<p>Furthermore just execute the following command to download and install the git repository.</p>

**Source latest version**

    $ git clone https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models

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
HEIGHT = 160
WIDTH = 160
```

Load the data

```python
X_train, y_train, X_val, y_val = get_data(...)
```

Create the base model that works as backbone for the segmentation model:

```python
base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
```

Define a Model and compile it with an appropriate loss:
 
```python
model = tasm.DANet(n_classes=3, base_model=base_model, output_layers=layers, backbone_trainable=False)
model.compile(tf.keras.optimizers.Adam(0.0001), loss=tasm.losses.CategoricalFocalLoss, tasm.metrics.IOUScore(threshold=0.5))
```

Now finally train the model:

```python
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=8,
    epochs=50,
    validation_data(x_val, y_val)
)
```
 
You can use the fit_generator method too, e.g. if you want to apply augmentations to the data.
For complete training pipelines, go to the <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples">Examples</a> folder
## Examples
- [Jupyter Notebook] Multi-class segmentation (sky, building) on CamVid dataset <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/examples/TASM_Example_1.ipynb">here</a>

## Models and Backbones

**Models**

- **<a href="https://arxiv.org/pdf/1411.4038.pdf">FCN</a>**
- **<a href="https://arxiv.org/abs/1505.04597">UNet</a>** (Orig<a href=""> qubvel </a>Code)
- **<a href="http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf">FPN</a>** (Orig<a href=""> qubvel </a>Code)
- **<a href="https://arxiv.org/abs/1612.01105">PSPNet</a>** (Orig<a href=""> qubvel </a>Code)
- **<a href="https://arxiv.org/pdf/1606.00915.pdf">DeepLab</a>**
- **<a href="https://arxiv.org/pdf/1706.05587.pdf">DeepLabV3</a>**
- **<a href="https://arxiv.org/pdf/1802.02611.pdf">DeepLabV3+</a>**
- **DANet**
- **OCNet**

Coming Soon...
- **CFNet**
    
**Backbones**
(For Details see <a href="">here</a>.)

|Type         | Names |
|-------------|-------|
|**VGG**          | ``'vgg16' 'vgg19'``|
|**ResNet**       | ``''resnet50' 'resnet50v2' 'resnet101' 'resnet101v2' 'resnet152' ' resnet152v2``|
|**Xception**     | ``''xception''``|
|**DenseNet**     | ``'densenet121' 'densenet169' 'densenet201'``|
|**EfficientNet** | ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'``|
    

    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 

## Documentation

## Change Log

## Citing

    @misc{Kezmann:2020,
      Author = {Jan-Marcel Kezmann},
      Title = {Segmentation Models},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation_Models}}
    } 
    
## License

Project is distributed under <a href="https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models/blob/master/LICENSE">MIT License</a>.

## References
<p>Thank you for all the papers that made this repository possible and especially thank you Pavel Yakubovskiy's initial segmentation models repository.</p>
- Pavel Yakubovskiy, Segmentation Models, 2019, GitHub, GitHubRepository, https://github.com/qubvel/segmentation_models

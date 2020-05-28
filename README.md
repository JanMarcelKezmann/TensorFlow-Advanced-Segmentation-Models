# Keras Advanced Segmentation Models
A Python Library for High-Level Semantic Segmentation Models.

## Preface
<p>Since the breakthrough of Deep Learning and Computer Vision was always one of the core problems that researcher all over the world have worked on, to create better models every day. One Computer Vision area that got huge attention in the last couple of years is Semantic Segmentation. The task to segment every pixel on a given image led to the invetion of many great models starting with the classical U-Net up to now more and more complex neural network structures. But even though many new algorithms were developed, the distribution of easy to use open source libraries which contain High-Level APIs that make the technology accessible for everyone.</p>
<p>Inspired by <a href="https://github.com/qubvel">qubvel's</a> <a href="https://github.com/qubvel/segmentation_models">segmentation_models</a> this repository builds upon his work and extends it by a variety of recently developed models which achieved great results on the <a href="https://www.cityscapes-dataset.com/">Cityscapes</a>, <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">PASCAL VOC 2012</a>, <a href="https://cs.stanford.edu/~roozbeh/pascal-context/">PASCAL Context</a>, <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a> dataset and many more.</p>

### Main Library Features
- High Level API
- 8 Segmentation Model Architectures for multi-class semantic segmentation
- Many already pretrained backbones for each architecture
- Many useful segmentation losses

## Table of Contents


**Backbones**

    =============  ===== 
    Type           Names
    =============  =====
    VGG            ``'vgg16' 'vgg19'``
    ResNet         ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    SE-ResNet      ``'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'``
    ResNeXt        ``'resnext50' 'resnext101'``
    SE-ResNeXt     ``'seresnext50' 'seresnext101'``
    SENet154       ``'senet154'``
    DenseNet       ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception      ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet      ``'mobilenet' 'mobilenetv2'``
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'``
    =============  =====

    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 

# Installation

**Requirements**
1) Python 3.6
2) keras >= 2.2.0 or tensorflow >= 1.15
3) keras-applications >= 1.0.7, <=1.0.8
4) image-classifiers == 1.0.*
5) efficientnet == 1.0.*

**Source latest version**

    $ pip install git+https://github.com/JanMarcelKezmann/Keras-Advamced-Segmentation-Models

# Documentation

# Change Log

# Citing

    @misc{Kezmann:2020,
      Author = {Jan-Marcel Kezmann},
      Title = {Segmentation Models},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/JanMarcelKezmann/Keras-Advanced-Segmentation_Models}}
    } 
    
# License

Project is distributed under <a href="https://github.com/JanMarcelKezmann/Keras-Advanced-Segmentation-Models/blob/master/LICENSE">MIT License</a>.

# References
<p>Thank you for all the papers that made this repository possible and especially thank you Pavel Yakubovskiy's initial segmentation models repository.</p>
- Pavel Yakubovskiy, Segmentation Models, 2019, GitHub, GitHubRepository, https://github.com/qubvel/segmentation_models



# I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models

## This paper was accepted to European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2023 (ECML-PKDD 2023)

Code accompanying the paper:
[I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models", ECML-PKDD Machine Learning and Cybersecurity Workshop, 2023](https://arxiv.org/abs/2306.07591).

### Abstract:
Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

## Prerequisites
    conda create -n query_attack python=3.8.12
    pip install -r requirements.txt

## Download the trained models weights
2. Download models' weights from this link: [models weights](https://drive.google.com/file/d/1LKLicAXgL-Q9QFtvMWDkHN-8ESPBNjtO/view?usp=sharing)
3. Unzip it and place it in models/state_dicts/*.pt

## Run
    python main.py --model=<model_name> --dataset=<dataset_name> --eps=<epsilon> --pop=<pop_size> --gen=<n_gen> --images=<n_images> --tournament=<n_tournament> --path=<imagenet_path>
- For MNIST dataset, run the above command with --model=custom
- The values used in the paper are:
  - pop_size=70
  - gen=600
  - images=200
  - tournament=25

If you wish to cite this paper:
```
@Article{a15110407,
AUTHOR = {Lapid, Raz and Haramaty, Zvika and Sipper, Moshe},
TITLE = {An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Convolutional Neural Networks},
JOURNAL = {Algorithms},
VOLUME = {15},
YEAR = {2022},
NUMBER = {11},
ARTICLE-NUMBER = {407},
URL = {https://www.mdpi.com/1999-4893/15/11/407},
ISSN = {1999-4893},
ABSTRACT = {Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces Qu ery-Efficient Evolutionary Attack&mdash;QuEry Attack&mdash;an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different, commonly used, pretrained image-classifications models&mdash;Inception-v3, ResNet-50, and VGG-16-BN&mdash;against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack&rsquo;s performance on non-differential transformation defenses and robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.},
DOI = {10.3390/a15110407}
}
```
![alt text](https://github.com/razla/QuEry-Attack/blob/master/figures/examples.png)

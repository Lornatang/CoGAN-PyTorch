# CoGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Coupled Generative Adversarial Networks](http://arxiv.org/pdf/1606.07536).

### Table of contents

1. [About Coupled Generative Adversarial Networks](#about-coupled-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
    * [Torch Hub call](#torch-hub-call)
    * [Base call](#base-call)
5. [Train](#train-eg-mnist)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Coupled Generative Adversarial Networks

If you're new to CoGANs, here's an abstract straight from the paper:

We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In contrast to the existing
approaches, which require tuples of corresponding images in different domains in the training set, CoGAN can learn a joint distribution without any
tuple of corresponding images. It can learn a joint distribution with just samples drawn from the marginal distributions. This is achieved by
enforcing a weight-sharing constraint that limits the network capacity and favors a joint distribution solution over a product of marginal
distributions one. We apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth images, and
learning a joint distribution of face images with different attributes. For each task it successfully learns the joint distribution without any tuple
of corresponding images. We also demonstrate its applications to domain adaptation and image transformation.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates
images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is
a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/CoGAN-PyTorch.git
$ cd CoGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. mnist)

```bash
$ cd weights/
$ python3 download_weights.py
```

### Test

#### Torch hub call

```python
# Using Torch Hub library.
import torch
import torchvision.utils as vutils

# Choose to use the device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model into the specified device.
model = torch.hub.load("Lornatang/CoGAN-PyTorch", "cogan", pretrained=True, progress=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
noise = torch.randn([num_images, 100], device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images1, generated_images2 = model(noise)
generated_images = torch.cat([generated_images1, generated_images2], dim=0)

# Save generate image.
vutils.save_image(generated_images, "mnist.png", normalize=True)
```

#### Base call

```text
usage: test.py [-h] [-a ARCH] [--num-images NUM_IMAGES] [--model-path PATH] [--pretrained] [--seed SEED] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: cogan. (Default: `cogan`)
  --num-images NUM_IMAGES
                        How many samples are generated at one time. (Default: 64)
  --model-path PATH     Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing testing.
  --gpu GPU             GPU id to use.

# Example (e.g. MNIST)
$ python3 test.py -a cogan --pretrained
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--image-size IMAGE_SIZE] [--channels CHANNELS] [--netD PATH] [--netG PATH] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: cogan. (Default: `cogan`)
  -j N, --workers N     Number of data loading workers. (Default: 4)
  --epochs N            Number of total epochs to run. (Default: 128)
  --start-epoch N       Manual epoch number (useful on restarts). (Default: 0)
  -b N, --batch-size N  Mini-batch size (default: 64), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (Default: 0.0002)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 32)
  --channels CHANNELS   The number of channels of the image. (Default: 3)
  --netD PATH           Path to Discriminator checkpoint.
  --netG PATH           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.
 
# Example (e.g. MNIST)
$ python3 train.py -a cogan --image-size 32 --channels 3 --pretrained --gpu 0 data
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a cogan --netD weights/Discriminator_epoch8.pth --netG weights/Generator_epoch8.pth --start-epoch 8 --gpu 0 data
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Coupled Generative Adversarial Networks

*Ming-Yu Liu, Oncel Tuzel*

**Abstract**

We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In contrast to the existing
approaches, which require tuples of corresponding images in different domains in the training set, CoGAN can learn a joint distribution without any
tuple of corresponding images. It can learn a joint distribution with just samples drawn from the marginal distributions. This is achieved by
enforcing a weight-sharing constraint that limits the network capacity and favors a joint distribution solution over a product of marginal
distributions one. We apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth images, and
learning a joint distribution of face images with different attributes. For each task it successfully learns the joint distribution without any tuple
of corresponding images. We also demonstrate its applications to domain adaptation and image transformation.

[[Paper]](http://xxx.itp.ac.cn/pdf/1606.07536)

```
@misc{1606.07536,
Author = {Ming-Yu Liu and Oncel Tuzel},
Title = {Coupled Generative Adversarial Networks},
Year = {2016},
Eprint = {arXiv:1606.07536},
}
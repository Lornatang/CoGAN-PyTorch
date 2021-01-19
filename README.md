# CoGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Coupled Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1606.07536).

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

We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In
contrast to the existing approaches, which require tuples of corresponding images in different domains in the training
set, CoGAN can learn a joint distribution without any tuple of corresponding images. It can learn a joint distribution
with just samples drawn from the marginal distributions. This is achieved by enforcing a weight-sharing constraint that
limits the network capacity and favors a joint distribution solution over a product of marginal distributions one. We
apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth
images, and learning a joint distribution of face images with different attributes. For each task it successfully learns
the joint distribution without any tuple of corresponding images. We also demonstrate its applications to domain
adaptation and image transformation.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

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
model = torch.hub.load("Lornatang/CoGAN-PyTorch", "mnist", pretrained=True, progress=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
noise = torch.randn(num_images, 100, device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images = model(noise)

# Save generate image.
vutils.save_image(generated_images, "mnist.png", normalize=True)
```

#### Base call

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

An implementation of CoGAN algorithm using PyTorch framework.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | mnist (default: mnist)
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. MNIST)
$ python3 test.py -a mnist --device cpu
```

### Train (e.g. MNIST)

```text
usage: train.py [-h] [-a ARCH] [-j N] [--start-iter N] [--iters N] [-b N]
                [--lr LR] [--image-size IMAGE_SIZE] [--channels CHANNELS]
                [--pretrained] [--netD PATH] [--netG PATH]
                [--manualSeed MANUALSEED] [--device DEVICE]
                DIR

An implementation of CoGAN algorithm using PyTorch framework.

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | mnist (default: mnist)
  -j N, --workers N     Number of data loading workers. (default:8)
  --start-iter N        manual iter number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        model. (default: 50000)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.0002)
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 32).
  --channels CHANNELS   The number of channels of the image. (default: 3).
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``0``).
                        
# Example (e.g. MNIST)
$ python3 train.py data -a mnist --image-size 32 --channels 3 --pretrained --device 0
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py data \
                   -a mnist \
                   --image-size 32 \
                   --channels 3 \
                   --start-iter 10000 \
                   --netG weights/mnist_G_iter_10000.pth \
                   --netD weights/mnist_D_iter_10000.pth \
                   --device 0
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Coupled Generative Adversarial Networks

*Ming-Yu Liu, Oncel Tuzel*

**Abstract**

We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In
contrast to the existing approaches, which require tuples of corresponding images in different domains in the training
set, CoGAN can learn a joint distribution without any tuple of corresponding images. It can learn a joint distribution
with just samples drawn from the marginal distributions. This is achieved by enforcing a weight-sharing constraint that
limits the network capacity and favors a joint distribution solution over a product of marginal distributions one. We
apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth
images, and learning a joint distribution of face images with different attributes. For each task it successfully learns
the joint distribution without any tuple of corresponding images. We also demonstrate its applications to domain
adaptation and image transformation.

[[Paper]](http://xxx.itp.ac.cn/pdf/1606.07536)

```
@misc{1606.07536,
Author = {Ming-Yu Liu and Oncel Tuzel},
Title = {Coupled Generative Adversarial Networks},
Year = {2016},
Eprint = {arXiv:1606.07536},
}
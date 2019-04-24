# Neural Cryptography

These are tools experimenting with neural cryptography. More specifically,
tools developed to allow neural networks to communicate in the presence
of adversarial networks. Experiments with currently only symmetric key 
encryption have been done. However, we indent to broaden our model designs
as development continues. 

This was done as the final project to an MIT course, 6.857. You can view the paper 
[here](https://courses.csail.mit.edu/6.857/2018/project/Modesitt-Henry-Coden-Lathe-NeuralCryptography.pdf)

The models are written in Keras and are trained using the 
tensorflow backend. Once completed, this 
ReadMe will include a comprehensive summary of our
findings and results. 

# Cryptographic and Architectural Descriptions

## Symmetric Key Encryption

These are network architecture's for performing symmetric key encryption. In 
this symmetric key encryption schheme, two parties known as 
[Alice and Bob](https://en.wikipedia.org/wiki/Alice_and_Bob) 
would like to communicate securely in the presence of an eavesdropping adversary, Eve.

We have developed several approaches to teaching neural networks to perform
this kind of encryption. These are described as follows.

### Small Convolutional Networks

This architecture is largely based on the [Google Brain Paper](https://arxiv.org/pdf/1610.06918v1.pdf)
here. Alice, Bob, and Eve are neural networks trained adversarially in the form
of alternating training Alice/Bob and Eve. Alice is given a plaintext, P, and a key,
K, and is asked to produce an encrypted ciphertext, C. Bob is asked to decrypt that
C into P given K. Eve is asked to do the same without K. Specific training schedule, loss
functions, can be found in the code.

### Bitwise Function Networks

In an attempt to learn more standardized symmetric key encryption operations,
such as the One-Time Pad, we have created a network architecture that is localized
to be element-wise or "Bitwise" among the inputs.

<img src='/assets/element_wise_nn.png' alt='Bitwise Function Network' style='max-width:50%;'></img>

## Purely Adversarial Neural Cryptography

In purely adversarial neural cryptography, we explore the capacity for Neural Networks
to be capable in detecting broken encryption. We format this goal as one of several
games, in the hope to allign with general cryptographic techniques and approaches.  

# Setup

After cloning this repository, setup a python virtual environment. This can be done with

```
source setup
```

this can also be used after initially setting up the environment to activate the virtual environment.

# Project Structure

The project is structured in the following fashion.

```
.
+-- _setup
+-- _bin
|   +-- this includes binary files like weights,
|      cached data files, or visualizations
+-- _data
|   +-- data.py
|        +-- data generation programs 
+-- _genneral
|        +-- general software like custom keras layers,
|            utility functions, etc.
+-- _models
|   +-- code that builds, trains, and visualizes keras
+-- assets
|   +-- other assets (like pictures)

```

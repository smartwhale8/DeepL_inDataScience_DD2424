# Deep Learning in Data Science
This repository contains project work completed as part of the course *Deep Learning for Data Science* at KTH. Each assignment showcases the practical application of various deep learning techniques. The assignments are designed to build a deeper understanding of neural networks, optimization, and model evaluation through implementation and experimentation.

## Assignments Overview

### Assignment 1: One-Layer Neural Network from Scratch for Image Classification
[Assignment 1 Code](Assignment1)

In this assignment, a one-layer neural network was trained to classify images from the CIFAR-10 dataset using mini-batch gradient descent. The cost function combined cross-entropy loss with L2 regularization on the weight matrix. The goal was to implement a basic neural network for image classification, focusing on understanding the fundamentals of training and optimization.
#### Key Learning Objectives:
1. **Data Handling and Preprocessing**: Learned to apply loading and pre-processing on the popular CIFAR-10 image dataset, including normalizing the image data and applying one-hot encoding technique on the labels.
2. **Neural Network Architecture**: Implemented, from scratch, a one-layer neural network with multiple outputs, understanding the role of weights (W) and biases (b) in a nerual network.
3. **Actvation Functions with Numerical Stability**: Implemented and understood the softmax function for multi-class classification, including techniques for improving numerical stability
4. **Loss Functions**: Implemented a cross-entropy loss for multi-class classification
5. **L2 Regularization**: Applied L2 regularization to reduce overfitting and improve generalization.
6. **Optimization**: Implemented mini-batch gradient descent, computed gradients analytically using backpropagation. Compared anatlytical gradients with numerical gradients for verification.
7. **Performance Metrics**: Calculated and tracked accuracy for model evaluation. Plotted and interpreted learning curves (training and validation losses).
8. **Hyperparamter Tuning**: Experimented with different learning rates (&eta;), Explored the impact of differnet regularization strengths (&lambda;)
---

### Assignment 2: Two-Layer Network for Image Classification
[Assignment 2 Code](Assignment2)

In this assignment, a two-layer neural network was trained to classify images from the CIFAR-10 dataset using mini-batch gradient descent (SGD) with a cyclic learning rate. The cost function combined cross-entropy loss with L2 regularization on the weight matrix. The forward and backward pass functions were adapted to handle more parameters, and hyperparameters like the learning rate and regularization term were fine-tuned to optimize performance.
#### Key Learning Objectives:
1. **Neural Network Design**: Learned to design a two-layer neural network with an input layer and an output layer using ReLU and Softmax activation functions, respectively.
2. **Data Preprocessing**: Implemented data normalization (zero-mean and unit variance) to prepare CIFAR-10 dataset for training, validation, and testing.
3. **Forward and Backward Propagation**: Developed forward and backward propagation functions for the two-layer network, calculating predictions and gradients.
4. **Mini-batch Gradient Descent**: Implemented mini-batch gradient descent and optimized with cyclical learning rates to improve model convergence.
5. **L2 Regularization**: Applied L2 regularization to reduce overfitting and improve generalization.
6. **Numerical Gradient Verification**: Verified the correctness of computed gradients through comparison with numerically computed gradients.
7. **Performance Metrics**: Computed model accuracy and evaluated its performance using cross-entropy loss.
---

### Assignment 3:: k-Layer Neural Network with Batch Normalization
[Assignment 3 Code](Assignment3)

In this assignment, we extended the k-layer neural network to classify images from the CIFAR-10 dataset. The task involved generalizing the network to support multiple layers and integrating batch normalization for both training and testing. The network was trained using mini-batch gradient descent and cyclical learning rates, similar to Assignment 2. The loss function combined cross-entropy and L2 regularization to optimize performance.
#### Key Learning Objectives:
1. **Multi-layer Neural Networks**: Expanded on neural network architecture by adding multiple hidden layers and leveraging batch normalization.
2. **Weight Initialization**: Used He initialization for weights in hidden layers to improve learning in deep neural networks.
3. **Batch Normalization**: Implemented batch normalization to improve the stability and speed of training by normalizing the inputs to each layer.
4. **Cyclical Learning Rates**: Enhanced training efficiency by implementing cyclical learning rates, dynamically adjusting the learning rate throughout training.
5. **Gradient Clipping**: Applied gradient clipping to mitigate the exploding gradient problem in deep networks, especially during backpropagation.
6. **Confusion Matrix**: Used confusion matrix visualizations to better understand model performance and error patterns across different classes.
7. **Dropout Regularization**: Introduced dropout in the model to prevent overfitting by randomly omitting neurons during training.
---

### Assignment 4: Recurrent Neural Network (RNN)
In this assignment, a vanilla Recurrent Neural Network (RNN) was trained to synthesize English text character by character using The Goblet of Fire by J.K. Rowling. Key tasks included preparing the data by mapping characters to indices, implementing backpropagation for gradient computation, and optimizing the RNN using AdaGrad. The final task involved synthesizing text using the trained RNN, given an initial hidden state and input character.
[Assignment 4 Code](Assignment4)

#### Key Learning Objectives:
1. **Recurrent Neural Networks (RNNs)**: Implemented an RNN model to learn from sequential data (text data), capturing temporal dependencies.
2. **Text Generation**: Learned how to generate text sequences using the RNN's learned parameters, demonstrating the ability to model sequential data.
3. **Backpropagation Through Time (BPTT)**: Applied backpropagation through time to compute gradients for the RNN, adjusting weights based on sequential dependencies.
4. **AdaGrad Optimization**: Utilized AdaGrad for efficient training by adapting learning rates based on the frequency of parameter updates.
5. **Numerical Gradient Checking**: Verified gradient correctness through numerical gradient checking, ensuring the implementation of BPTT is correct.
6. **Text Synthesis**: Generated text using the RNN and tracked model performance through loss curves, improving with each iteration.
7. **Gradient Clipping**: Incorporated gradient clipping to stabilize RNN training and prevent the exploding gradient problem.
---


## Usage
The code is shared as Interactive Python Notebook which can be executed in environments such as Google Colab or Jupyter. A pdf report accompnies the code with results to the assignment questions.


---

# Custom CUDA-Accelerated Neural Network Framework

## Overview

This repository contains a high-performance, object-oriented deep learning framework developed entirely from scratch using **C++** and **CUDA**. Inspired by modern architectures such as PyTorch, the project provides a low-level, highly customizable approach to constructing and training Convolutional Neural Networks (CNNs) directly on the GPU.

By bypassing standard high-level libraries (e.g., TensorFlow, PyTorch), this framework serves as a comprehensive implementation of core deep learning mathematics, forward and backward propagation algorithms, and memory-optimized parallel GPU computing utilizing custom CUDA kernels. The system has been validated on the MNIST and Fashion-MNIST datasets, demonstrating robust convergence and computational efficiency.

## Core Features

* **Custom GPU Kernels:** Matrix multiplications, convolutions, spatial pooling, and gradient computations are implemented exclusively via custom CUDA `__global__` functions.


* **Object-Oriented Architecture:** A modular `ILayer` interface enables polymorphic stacking of various neural network layers.


* **Layer Implementations:**
* `Conv2DLayer`: Multi-channel 2D convolution operations with dynamic filter generation.


* `MaxPoolLayer`: 2D spatial down-sampling with precise gradient routing during backpropagation.


* `FlattenLayer`: Dimensionality reduction to bridge multi-dimensional feature maps to 1D linear arrays.


* `LinearLayer`: Fully connected dense layers with GPU-accelerated matrix operations.




* **Mathematical Operations:** Hardware-optimized algorithms for ReLU, Sigmoid, Softmax, and Cross-Entropy Loss.


* **Memory Management:** Efficient VRAM allocation, dynamic memory handling, and device synchronization to ensure thread safety and prevent memory leaks.


* **Model Serialization:** Capability to save and load trained network states (weights and biases) via binary file operations (`.bin`).



---

## System Requirements

To compile and execute this framework, the following dependencies must be met:

* **CUDA Toolkit:** Compatible nvcc compiler for GPU execution.
* **C++ Compiler:** GCC, Clang, or MSVC with C++14 support or higher.
* **CMake:** Build system generator.
* **Dataset:** MNIST or Fashion-MNIST binary files placed within a root `data/` directory (e.g., `train-images.idx3-ubyte`).



---

## Build and Execution

### 1. Compilation

Navigate to the project root directory and utilize CMake to build the executable:

```bash
mkdir build
cd build
cmake ..
cmake --build .

```

### 2. Model Training

To initiate the training sequence, execute the compiled program without arguments. The network performs mini-batch gradient descent and logs the cross-entropy loss per epoch.

```bash
./build/prog

```

*Note: Upon successful completion of the defined epochs, the framework automatically serializes the trained weights and filter configurations to `data/mnist_cnn_oop_model.bin*`.

### 3. Model Evaluation

To evaluate the serialized model against the test dataset, pass the `test` argument. This operation loads the saved `.bin` file into VRAM and outputs the final classification accuracy.

```bash
./build/prog test

```

---

## System Architecture

The architecture is strictly modular, dividing responsibilities among distinct polymorphic components.

### `Neural_Network` (Core Engine)

The primary controller class managing a sequential container of `ILayer` pointers.

* `forward(std::vector<float>& input)`: Executes the forward pass by iterating through all instantiated layers, passing the output of layer $L$ as the input to layer $L+1$.


* `backward(std::vector<float>& target, float learning_rate)`: Computes Softmax and Cross-Entropy gradients on the device memory, subsequently propagating the error backwards through the layer hierarchy.


* `save_model()` / `load_model()`: Handles I/O operations for model persistence.



### `Neural_Matrix`

A foundational matrix mathematics wrapper for CUDA device pointers. It encapsulates VRAM memory allocation (`cudaMalloc`), host-to-device data transfers (`cudaMemcpy`), and executes tensor operations using optimal block/grid thread dimensions.

### Convolutional and Spatial Layers

* **`Conv2DLayer`:** Executes multi-channel 2D convolutions. The `forward` pass applies parameterized filters across input tensors, while the `backward` pass calculates filter gradients and input error gradients simultaneously using highly parallelized CUDA kernels.


* **`MaxPoolLayer`:** Reduces spatial dimensions to lower computational complexity. During backpropagation, it routes the incoming gradients exclusively to the coordinate indices that contained the maximum values during the forward pass.



### Feed-Forward Layers

* **`FlattenLayer`:** A structural layer mapping 3D/2D spatial outputs from the CNN blocks into a 1D vector format required by downstream dense networks.


* **`LinearLayer`:** The fully connected component. It maintains optimized `Neural_Matrix` instances for weights and biases, performing large-scale matrix multiplications utilizing CUDA computational grids.
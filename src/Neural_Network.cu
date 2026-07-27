#include "Neural_Network.h"
#include <algorithm>
#include <cstddef>
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void forward_kernel(float* weights, float* input, float* bias, float* output, int rows, int cols){
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += weights[x * cols + col] * input[col];
        }
        output[x] = sum + bias[x];
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && data[idx] < 0.0f) {
        data[idx] = 0.0f;
    }
}

__global__ void relu_derivative_kernel(float* gradients, const float* pre_activations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && pre_activations[idx] <= 0.0f) {
        gradients[idx] = 0.0f;
    }
}

__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = fmaxf(-40.0f, fminf(40.0f, data[idx]));
        data[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void sigmoid_derivative_kernel(float* gradients, const float* pre_activations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = fmaxf(-40.0f, fminf(40.0f, pre_activations[idx]));
        float sig = 1.0f / (1.0f + expf(-x));
        gradients[idx] *= sig * (1.0f - sig);
    }
}

__global__ void multiply_scalar_kernel(float* data, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] *= scalar;
    }
}

__global__ void subtract_kernel(float* data, const float* other, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] -= other[idx];
    }
}

__global__ void add_kernel(float* data, const float* other, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] += other[idx];
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < rows * cols) {
        int r = index / cols;
        int c = index % cols;
        output[c * rows + r] = input[index];
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for(int i = 0; i < A_cols; i++) {
            sum += A[row * A_cols + i] * B[i * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}



Neural_Matrix::Neural_Matrix(int r, int c){
    rows = r;
    cols = c;
    device_data = nullptr;

    data.resize(rows * cols, 0.0f);
}

Neural_Matrix::~Neural_Matrix() {
    if (device_data != nullptr) {
        cudaFree(device_data);
    }
}


Neural_Matrix::Neural_Matrix(const Neural_Matrix& other){
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    device_data = nullptr;

    if (other.device_data != nullptr) {
        allocate_device_memory();
        cudaMemcpy(device_data, other.device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}


Neural_Matrix& Neural_Matrix::operator=(const Neural_Matrix& other) {
    if (this == &other) return *this;

    rows = other.rows;
    cols = other.cols;
    data = other.data;

    if (device_data != nullptr) cudaFree(device_data);
    device_data = nullptr;

    if (other.device_data != nullptr) {
        allocate_device_memory();
        cudaMemcpy(device_data, other.device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

void Neural_Matrix::allocate_device_memory() {

    if (device_data == nullptr) {
        cudaMalloc(&device_data, rows * cols * sizeof(float));
    }
}

void Neural_Matrix::copy_to_device() {

    if (device_data == nullptr) allocate_device_memory();
    cudaMemcpy(device_data, data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Neural_Matrix::copy_to_host() {

    if (device_data != nullptr) {
        cudaMemcpy(data.data(), device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void Neural_Matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dis(gen);
    }
}

void Neural_Matrix::print() const {

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void Neural_Matrix::apply_relu(){

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::relu_derivative(const Neural_Matrix& matrix){

    int size = rows * cols;
    int threadsPerBlock = 256;

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, matrix.device_data, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::apply_sigmoid(){

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::sigmoid_derivative(const Neural_Matrix& pre_activations) {
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, pre_activations.device_data, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::multiply_scalar(float scalar){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiply_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, scalar, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::subtract(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    subtract_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    cudaDeviceSynchronize();
}

void Neural_Matrix::add(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    cudaDeviceSynchronize();
}

Neural_Matrix Neural_Matrix::transpose() const{
    Neural_Matrix result(this->cols, this->rows);
    result.allocate_device_memory();

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, result.device_data, this->rows, this->cols);
    cudaDeviceSynchronize();
    return result;
}

Neural_Matrix Neural_Matrix::multiply(const Neural_Matrix& other) const{
    if (this->cols != other.rows) { throw std::runtime_error("Matrix dimensions are incompatible."); }

    Neural_Matrix result(this->rows, other.cols);
    result.allocate_device_memory();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((other.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (this->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, other.device_data, result.device_data, this->rows, this->cols, other.cols);
    cudaDeviceSynchronize();
    return result;
}


Neural_Network::Neural_Network(std::vector<int> topology) {
    this->topology = topology;

    for (size_t i = 0; i < topology.size() - 1; ++i) {

        Neural_Matrix weight_matrix(topology[i + 1], topology[i]);
        weight_matrix.randomize();

        weight_matrix.allocate_device_memory();
        weight_matrix.copy_to_device();
        weights.push_back(weight_matrix);

        Neural_Matrix bias_matrix(topology[i + 1], 1);
        bias_matrix.randomize();

        bias_matrix.allocate_device_memory();
        bias_matrix.copy_to_device();
        biases.push_back(bias_matrix);
    }
}

std::vector<float> Neural_Network::forward(const std::vector<float>& input) {
    layer_activations.clear();
    layer_outputs.clear();

    Neural_Matrix current_layer(input.size(), 1);
    current_layer.data = input;
    current_layer.allocate_device_memory();
    current_layer.copy_to_device();

    layer_activations.push_back(current_layer);

    for (size_t i = 0; i < weights.size(); i++) {
        Neural_Matrix next_layer(weights[i].rows, 1);
        next_layer.allocate_device_memory();

        int threadsPerBlock = 256;
        int blocksPerGrid = (weights[i].rows + threadsPerBlock - 1) / threadsPerBlock;

        forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            weights[i].device_data,
            current_layer.device_data,
            biases[i].device_data,
            next_layer.device_data,
            weights[i].rows,
            weights[i].cols
        );
        cudaDeviceSynchronize();

        current_layer = next_layer;
        layer_outputs.push_back(current_layer);

        if (i < weights.size() - 1) {
            current_layer.apply_relu();
        } else {
            current_layer.apply_sigmoid();
        }
        layer_activations.push_back(current_layer);
    }

    current_layer.copy_to_host();
    return current_layer.data;
}

float Neural_Network::calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target){

    float differance = 0.0f;

    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        differance += diff * diff;
    }

    return differance / static_cast<float>(predicted.size());
}

void Neural_Network::backpropagate(const std::vector<float>& target, float learning_rate) {
    Neural_Matrix error(target.size(), 1);

    layer_activations.back().copy_to_host();

    for (size_t i = 0; i < target.size(); i++) {
        float tahmin = layer_activations.back().data[i];
        error.data[i] = 2.0f * (tahmin - target[i]);
    }
    error.allocate_device_memory();
    error.copy_to_device();

    for (int i = weights.size() - 1; i >= 0; i--) {
        Neural_Matrix gradients = error;

        if (i < weights.size() - 1) {
            gradients.relu_derivative(layer_outputs[i]);
        } else {
            gradients.sigmoid_derivative(layer_outputs[i]);
        }

        Neural_Matrix prev_activation_T = layer_activations[i].transpose();
        Neural_Matrix weight_deltas = gradients.multiply(prev_activation_T);

        Neural_Matrix bias_deltas = gradients;
        bias_deltas.multiply_scalar(learning_rate);
        biases[i].subtract(bias_deltas);

        weight_deltas.multiply_scalar(learning_rate);
        Neural_Matrix old_weights = weights[i];

        weights[i].subtract(weight_deltas);

        if (i > 0) {
            Neural_Matrix weights_T = old_weights.transpose();
            error = weights_T.multiply(error);
        }
    }
}

void Neural_Network::save_model(const std::string& filename) const{
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    size_t topology_size = topology.size();
    out.write(reinterpret_cast<const char*>(&topology_size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(topology.data()), topology_size * sizeof(int));

    for (size_t i = 0; i < weights.size(); i++) {
        
        const_cast<Neural_Matrix&>(weights[i]).copy_to_host();
        const_cast<Neural_Matrix&>(biases[i]).copy_to_host();

        size_t weights_size = weights[i].data.size();
        out.write(reinterpret_cast<const char*>(weights[i].data.data()), weights_size * sizeof(float));

        size_t biases_size = biases[i].data.size();
        out.write(reinterpret_cast<const char*>(biases[i].data.data()), biases_size * sizeof(float));
    }
    out.close();
}

void Neural_Network::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);

    size_t topology_size;
    in.read(reinterpret_cast<char*>(&topology_size), sizeof(size_t));
    topology.resize(topology_size);
    in.read(reinterpret_cast<char*>(topology.data()), topology_size * sizeof(int));

    weights.clear();
    biases.clear();

    for (size_t i = 0; i < topology_size - 1; i++) {
        Neural_Matrix weights_matrix(topology[i + 1], topology[i]);
        weights.push_back(weights_matrix);
        Neural_Matrix biases_matrix(topology[i + 1], 1);
        biases.push_back(biases_matrix);
    }

    for (size_t i = 0; i < weights.size(); i++) {
        size_t weights_size = weights[i].data.size();
        in.read(reinterpret_cast<char*>(weights[i].data.data()), weights_size * sizeof(float));
        weights[i].copy_to_device();

        size_t biases_size = biases[i].data.size();
        in.read(reinterpret_cast<char*>(biases[i].data.data()), biases_size * sizeof(float));
        biases[i].copy_to_device();
    }
    in.close();
}

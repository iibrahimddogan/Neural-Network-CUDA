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

__global__ void add_bias_broadcast_kernel(float* output, const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[row * cols + col] += bias[row];
    }
}

__global__ void sum_bias_gradients_kernel(const float* gradients, float* bias_deltas, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += gradients[row * cols + col];
        }
        bias_deltas[row] = sum;
    }
}

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
    
}

void Neural_Matrix::relu_derivative(const Neural_Matrix& matrix){

    int size = rows * cols;
    int threadsPerBlock = 256;

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, matrix.device_data, size);
    
}

void Neural_Matrix::apply_sigmoid(){

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size);
    
}

void Neural_Matrix::sigmoid_derivative(const Neural_Matrix& pre_activations) {
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, pre_activations.device_data, size);
    
}

void Neural_Matrix::multiply_scalar(float scalar){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiply_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, scalar, size);
    
}

void Neural_Matrix::subtract(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    subtract_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    
}

void Neural_Matrix::add(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    
}

void Neural_Matrix::transpose(Neural_Matrix& result) const{

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, result.device_data, this->rows, this->cols);
    
}

void Neural_Matrix::multiply(const Neural_Matrix& other, Neural_Matrix& result) const{
    if (this->cols != other.rows) { throw std::runtime_error("Matrix dimensions are incompatible."); }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((other.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (this->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, other.device_data, result.device_data, this->rows, this->cols, other.cols);
    
}

Neural_Network::Neural_Network(std::vector<int> topology, int batch_size) {
    this->topology = topology;
    this->batch_size = batch_size;
    
    size_t num_layers = topology.size();
    weights.reserve(num_layers - 1);
    biases.reserve(num_layers - 1);

    layer_outputs.reserve(num_layers - 1);
    layer_activations.reserve(num_layers);

    errors.reserve(num_layers - 1);
    weights_T.reserve(num_layers - 1);
    activations_T.reserve(num_layers);
    weight_deltas.reserve(num_layers - 1);

    
    Neural_Matrix input_activation(topology[0], batch_size);
    input_activation.allocate_device_memory();
    layer_activations.push_back(input_activation);

    Neural_Matrix input_activation_T(batch_size, topology[0]);
    input_activation_T.allocate_device_memory();
    activations_T.push_back(input_activation_T);

    for (size_t i = 0; i < num_layers - 1; ++i) {

        
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

        
        Neural_Matrix output(topology[i + 1], batch_size);
        output.allocate_device_memory();
        layer_outputs.push_back(output);

        Neural_Matrix activation(topology[i + 1], batch_size);
        activation.allocate_device_memory();
        layer_activations.push_back(activation);

        
        Neural_Matrix error_mat(topology[i + 1], batch_size);
        error_mat.allocate_device_memory();
        errors.push_back(error_mat);

        Neural_Matrix w_T(topology[i], topology[i + 1]);
        w_T.allocate_device_memory();
        weights_T.push_back(w_T);

        Neural_Matrix a_T(batch_size, topology[i + 1]);
        a_T.allocate_device_memory();
        activations_T.push_back(a_T);

        Neural_Matrix w_delta(topology[i + 1], topology[i]);
        w_delta.allocate_device_memory();
        weight_deltas.push_back(w_delta);
    }
}

std::vector<float> Neural_Network::forward(const std::vector<float>& input) {
    layer_activations[0].data = input;
    layer_activations[0].copy_to_device();

    for (size_t i = 0; i < weights.size(); i++) {

        weights[i].multiply(layer_activations[i], layer_outputs[i]);

        
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((batch_size + 15) / 16, (weights[i].rows + 15) / 16);
        add_bias_broadcast_kernel << <blocksPerGrid, threadsPerBlock >> > (
            layer_outputs[i].device_data, biases[i].device_data, weights[i].rows, batch_size
            );

        
        cudaMemcpy(layer_activations[i + 1].device_data, layer_outputs[i].device_data,
            weights[i].rows * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);

        if (i < weights.size() - 1) {
            layer_activations[i + 1].apply_relu();
        }
        else {
            layer_activations[i + 1].apply_sigmoid();
        }
    }
    layer_activations.back().copy_to_host();
    return layer_activations.back().data;
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
    size_t last_idx = weights.size() - 1;

    
    layer_activations.back().copy_to_host();
    for (size_t i = 0; i < target.size(); i++) {
        float tahmin = layer_activations.back().data[i];
        errors[last_idx].data[i] = 2.0f * (tahmin - target[i]);
    }
    errors[last_idx].copy_to_device();

    
    float batch_learning_rate = learning_rate / (float)batch_size;

    for (int i = last_idx; i >= 0; i--) {
        Neural_Matrix& gradients = errors[i];
        if (i < (int)last_idx) {
            gradients.relu_derivative(layer_outputs[i]);
        }
        else {
            gradients.sigmoid_derivative(layer_outputs[i]);
        }

        layer_activations[i].transpose(activations_T[i]);
        gradients.multiply(activations_T[i], weight_deltas[i]);

        if (i > 0) {
            weights[i].transpose(weights_T[i]);
            weights_T[i].multiply(errors[i], errors[i - 1]);
        }

        
        Neural_Matrix bias_deltas(biases[i].rows, 1);
        bias_deltas.allocate_device_memory();
        int threads = 256;
        int blocks = (biases[i].rows + threads - 1) / threads;
        sum_bias_gradients_kernel << <blocks, threads >> > (
            gradients.device_data, bias_deltas.device_data, biases[i].rows, batch_size
            );

        bias_deltas.multiply_scalar(batch_learning_rate);
        biases[i].subtract(bias_deltas);

        weight_deltas[i].multiply_scalar(batch_learning_rate);
        weights[i].subtract(weight_deltas[i]);
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

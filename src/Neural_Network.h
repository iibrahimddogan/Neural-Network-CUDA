#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if(error != cudaSuccess){ \
        throw std::runtime_error(std::string("CUDA error code: ") + cudaGetErrorString(error) + " -> File: " + std::string(__FILE__) + " line: " + std::to_string(__LINE__)); \
    } \
} while (0)


class Neural_Matrix {
private:
    int rows, cols;
    std::vector<float> data;
    float* device_data;

    friend class LinearLayer;
    friend class Neural_Network;

public:
    Neural_Matrix(int r, int c);
    ~Neural_Matrix();

    Neural_Matrix(const Neural_Matrix& other);
    Neural_Matrix& operator=(const Neural_Matrix& other);

    void allocate_device_memory();
    void copy_to_device();
    void copy_to_host();

    void randomize();
    void print() const;
    void apply_relu();
    void relu_derivative(const Neural_Matrix& matrix);
    void apply_sigmoid();
    void sigmoid_derivative(const Neural_Matrix& pre_activations);
    void multiply_scalar(float scalar);
    void subtract(const Neural_Matrix& other);
    void add(const Neural_Matrix& other);

    void transpose(Neural_Matrix& result) const;
    void multiply(const Neural_Matrix& other, Neural_Matrix& result) const;
};


class ILayer {
public:
    virtual ~ILayer() = default;
    virtual std::vector<float> forward(const std::vector<float>& input, int batch_size) = 0;
    virtual std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) = 0;
};


class Conv2DLayer : public ILayer {
private:
    int input_size;
    int filter_size;
    int output_size;
    int num_of_filters;
    std::vector<float> cnn_filter;
    std::vector<float> last_input;

    float* d_input;
    float* d_filter;
    float* d_conv_output;
    float* d_conv_gradients;
    float* d_filter_gradients;
    int current_batch_size;

public:
    Conv2DLayer(int input_size, int filter_size, int num_of_filters);
    ~Conv2DLayer();

    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;

    const std::vector<float>& get_filter() const { return cnn_filter; }
    void set_filter(const std::vector<float>& f) { cnn_filter = f; }
};


class MaxPoolLayer : public ILayer {
private:
    int input_size;
    int pool_size;
    int output_size;
    int channels;
    std::vector<float> last_input;

    float* d_input;
    float* d_pool_output;
    float* d_pool_gradients;
    float* d_conv_gradients;
    int current_batch_size;

public:
    MaxPoolLayer(int input_size, int pool_size,int channels);
    ~MaxPoolLayer();
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;
};


class LinearLayer : public ILayer {
private:
    Neural_Matrix weights;
    Neural_Matrix biases;
    Neural_Matrix weight_deltas;
    Neural_Matrix weights_T;
    Neural_Matrix output;
    Neural_Matrix activation;
    Neural_Matrix error_mat;
    Neural_Matrix activations_T;
    std::string activation_type; // relu or sigmoid

    std::vector<float> last_input;
public:
    LinearLayer(int in_features, int out_features, const std::string& act_type);
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;

    Neural_Matrix& get_weights() { return weights; }
    Neural_Matrix& get_biases() { return biases; }
};


class Neural_Network {
private:
    int batch_size;
    std::vector<ILayer*> layers;

public:
    Neural_Network(int batch_size);
    ~Neural_Network();

    void add_layer(ILayer* layer);

    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& target, float learning_rate);

    float calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target);

    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};
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


__global__ void max_pooling_backward_kernel(const float* input_images, const float* pool_gradients, float* conv_gradients, int input_size, int pool_size, int output_size, int batch_size) {

    
    int out_col = threadIdx.x + blockDim.x * blockIdx.x;
    int out_row = threadIdx.y + blockDim.y * blockIdx.y;
    int batch_idx = threadIdx.z + blockDim.z * blockIdx.z;

    if (out_col < output_size && out_row < output_size && batch_idx < batch_size) {

        int start_row = out_row * pool_size;
        int start_col = out_col * pool_size;

        int in_batch_offset = batch_idx * (input_size * input_size);
        int out_batch_offset = batch_idx * (output_size * output_size);

        
        float max_val = -999999.9f;
        int max_r = 0;
        int max_c = 0;

        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                float val = input_images[in_batch_offset + ((start_row + i) * input_size) + (start_col + j)];
                if (val > max_val) {
                    max_val = val;
                    max_r = i;
                    max_c = j;
                }
            }
        }

        
        float grad_val = pool_gradients[out_batch_offset + (out_row * output_size + out_col)];

        
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int conv_grad_idx = in_batch_offset + ((start_row + i) * input_size) + (start_col + j);

                if (i == max_r && j == max_c) {
                    conv_gradients[conv_grad_idx] = grad_val; 
                }
                else {
                    conv_gradients[conv_grad_idx] = 0.0f;     
                }
            }
        }
    }
}


__global__ void convolution_filter_gradient_kernel(const float* input_images, const float* conv_gradients, float* filter_gradients, int input_size, int filter_size, int conv_output_size, int batch_size) {

    
    int f_col = threadIdx.x + blockDim.x * blockIdx.x;
    int f_row = threadIdx.y + blockDim.y * blockIdx.y;

    if (f_row < filter_size && f_col < filter_size) {
        float grad_sum = 0.0f;

        
        for (int b = 0; b < batch_size; ++b) {
            int in_batch_offset = b * (input_size * input_size);
            int out_batch_offset = b * (conv_output_size * conv_output_size);

            
            for (int r = 0; r < conv_output_size; ++r) {
                for (int c = 0; c < conv_output_size; ++c) {

                    
                    float input_val = input_images[in_batch_offset + (r + f_row) * input_size + (c + f_col)];
                    
                    float grad_val = conv_gradients[out_batch_offset + (r * conv_output_size + c)];

                    grad_sum += input_val * grad_val;
                }
            }
        }

        
        filter_gradients[f_row * filter_size + f_col] = grad_sum / (float)batch_size;
    }
}

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

__global__ void matmul_kernel(const float* A_matrix, const float* B_matrix, float* outputMatrix, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for(int i = 0; i < A_cols; i++) {
            sum += A_matrix[row * A_cols + i] * B_matrix[i * B_cols + col];
        }
        outputMatrix[row * B_cols + col] = sum;
    }
}

__global__ void convolution_kernel(const float* input_images, const float* filter, float* output_images, int input_size, int filter_size, int output_size, int batch_size) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int batch_idx = threadIdx.z + blockDim.z * blockIdx.z;

    if (row < output_size && col < output_size && batch_idx < batch_size)
    {
        float sum = 0.0f;

        int input_offset = batch_idx * (input_size * input_size);
        int output_offset = batch_idx * (output_size * output_size);

        for (int  i = 0; i < filter_size; i++)
        {
            for (int j = 0; j < filter_size; j++) {
                int image_row = row + i;
                int image_col = col + j;

                float pixel_value = input_images[input_offset + (image_row * input_size + image_col)];
                float filter_value = filter[i * filter_size + j];

                sum += pixel_value * filter_value;
            }
        }

        if (sum < 0.0f)
        {
            sum = 0.0f;
        }

        output_images[output_offset + (row * output_size + col)] = sum;
    }
}

__global__ void max_pooling_kernel(const float* input_images, float* output_images, int input_size, int pool_size, int output_size, int batch_size) {
    int output_col = threadIdx.x + blockDim.x * blockIdx.x;
    int output_row = threadIdx.y + blockDim.y * blockIdx.y;
    int batch_idx = threadIdx.z + blockDim.z * blockIdx.z;

    if (output_col < output_size && output_row < output_size && batch_idx < batch_size)
    {
        int start_row = output_row * pool_size;
        int start_col = output_col * pool_size;

        int input_batch_offset = batch_idx * (input_size * input_size);
        int output_batch_offset = batch_idx * (output_size * output_size);

        float max_val = -999999.9f; //min for now

        for (int i = 0; i < pool_size; i++)
        {
            for (int j = 0; j < pool_size; j++)
            {
                float value = input_images[input_batch_offset + ((start_row + i) * input_size) + (start_col + j)];

                if (value > max_val)
                {
                    max_val = value;
                }
            }
        }
        output_images[output_batch_offset + (output_row * output_size + output_col)] = max_val;
    }
}


Conv2DLayer::Conv2DLayer(int input_size, int filter_size) {
    this->input_size = input_size;
    this->filter_size = filter_size;
    this->output_size = input_size - filter_size + 1;
    this->cnn_filter = {
        0.1f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f,
        0.1f, 0.1f, 0.1f
    };
}

std::vector<float> Conv2DLayer::forward(const std::vector<float>& input, int batch_size) {
    this->last_input = input;
    int conv_output_size = this->output_size;
    float* d_input, * d_filter, * d_conv_output;

    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_size * filter_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_output, batch_size * conv_output_size * conv_output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, cnn_filter.data(), filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (conv_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (conv_output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    convolution_kernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_filter, d_conv_output, input_size, filter_size, conv_output_size, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(batch_size * conv_output_size * conv_output_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_conv_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_conv_output));

    return output;
}

std::vector<float> Conv2DLayer::backward(const std::vector<float>& gradient, float learning_rate, int batch_size) {
    float* d_input, * d_conv_gradients, * d_filter_gradients;

    CUDA_CHECK(cudaMalloc(&d_input, last_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_gradients, gradient.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter_gradients, filter_size * filter_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, last_input.data(), last_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv_gradients, gradient.data(), gradient.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(filter_size, filter_size, 1);
    dim3 blocksPerGrid(1, 1, 1);

    convolution_filter_gradient_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_input, d_conv_gradients, d_filter_gradients,
        input_size, filter_size, output_size, batch_size
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_filter_gradients(filter_size * filter_size);
    CUDA_CHECK(cudaMemcpy(h_filter_gradients.data(), d_filter_gradients, h_filter_gradients.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_conv_gradients));
    CUDA_CHECK(cudaFree(d_filter_gradients));

    for (int i = 0; i < filter_size * filter_size; i++) {
        cnn_filter[i] = cnn_filter[i] - (learning_rate * h_filter_gradients[i]);
    }

    return std::vector<float>();
}


MaxPoolLayer::MaxPoolLayer(int input_size, int pool_size) {
    this->input_size = input_size;
    this->pool_size = pool_size;
    this->output_size = input_size / pool_size;
}

std::vector<float> MaxPoolLayer::forward(const std::vector<float>& input, int batch_size) {
    this->last_input = input;
    float* d_input, * d_pool_output;
    int flattened_size = output_size * output_size;

    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool_output, batch_size * flattened_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    max_pooling_kernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_pool_output, input_size, pool_size, output_size, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> pooled_images(batch_size * flattened_size);
    CUDA_CHECK(cudaMemcpy(pooled_images.data(), d_pool_output, pooled_images.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_pool_output));

    std::vector<float> mlp_inputs(batch_size * flattened_size);
    for (int p = 0; p < flattened_size; ++p) {
        for (int b = 0; b < batch_size; ++b) {
            mlp_inputs[p * batch_size + b] = pooled_images[b * flattened_size + p];
        }
    }
    return mlp_inputs;
}

std::vector<float> MaxPoolLayer::backward(const std::vector<float>& gradient, float learning_rate, int batch_size) {
    int flattened_size = output_size * output_size;
    std::vector<float> pool_gradients(batch_size * flattened_size);
    for (int p = 0; p < flattened_size; ++p) {
        for (int b = 0; b < batch_size; ++b) {
            pool_gradients[b * flattened_size + p] = gradient[p * batch_size + b];
        }
    }

    float* d_input, * d_pool_gradients, * d_conv_gradients;
    CUDA_CHECK(cudaMalloc(&d_input, last_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool_gradients, pool_gradients.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_gradients, batch_size * input_size * input_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, last_input.data(), last_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pool_gradients, pool_gradients.data(), pool_gradients.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    max_pooling_backward_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_input, d_pool_gradients, d_conv_gradients,
        input_size, pool_size, output_size, batch_size
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> conv_gradients(batch_size * input_size * input_size);
    CUDA_CHECK(cudaMemcpy(conv_gradients.data(), d_conv_gradients, conv_gradients.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_pool_gradients));
    CUDA_CHECK(cudaFree(d_conv_gradients));

    return conv_gradients;
}


LinearLayer::LinearLayer(int in_features, int out_features, const std::string& act_type)
    :
    weights(out_features, in_features),
    biases(out_features, 1),

    weight_deltas(out_features, in_features),
    weights_T(in_features, out_features),

    output(out_features, 1),
    activation(out_features,1),
    error_mat(out_features,1),
    activations_T(1, out_features),
    activation_type(act_type)

{
    weights.randomize();
    weights.allocate_device_memory();
    weights.copy_to_device();

    biases.randomize();
    biases.allocate_device_memory();
    biases.copy_to_device();

    weight_deltas.allocate_device_memory();
    weights_T.allocate_device_memory();

    output.allocate_device_memory();
    activation.allocate_device_memory();
    error_mat.allocate_device_memory();
    activations_T.allocate_device_memory();
}

std::vector<float> LinearLayer::forward(const std::vector<float>& input, int batch_size) {
    last_input = input;

    Neural_Matrix input_activation(weights.cols, batch_size);
    input_activation.data = input;
    input_activation.allocate_device_memory();
    input_activation.copy_to_device();

    if (output.cols != batch_size) {
        output = Neural_Matrix(weights.rows, batch_size);
        output.allocate_device_memory();
        activation = Neural_Matrix(weights.rows, batch_size);
        activation.allocate_device_memory();
    }

    weights.multiply(input_activation, output);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((batch_size + 15) / 16, (weights.rows + 15) / 16);
    add_bias_broadcast_kernel<<<blocksPerGrid, threadsPerBlock >>>(output.device_data, biases.device_data, weights.rows, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(activation.device_data, output.device_data, weights.rows * batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

    if (activation_type == "relu") {
        activation.apply_relu();
    }
    else if (activation_type == "sigmoid") {
        activation.apply_sigmoid();
    }

    activation.copy_to_host();
    return activation.data;
}

std::vector<float> LinearLayer::backward(const std::vector<float>& gradient, float learning_rate, int batch_size) {
    if (error_mat.cols != batch_size) {
        error_mat = Neural_Matrix(weights.rows, batch_size);
        error_mat.allocate_device_memory();
        activations_T = Neural_Matrix(batch_size, weights.cols);
        activations_T.allocate_device_memory();
    }

    error_mat.data = gradient;
    error_mat.copy_to_device();

    if (activation_type == "relu") {
        error_mat.relu_derivative(output);
    }
    else if (activation_type == "sigmoid") {
        error_mat.sigmoid_derivative(output);
    }

    Neural_Matrix input_activation(weights.cols, batch_size);
    input_activation.data = last_input;
    input_activation.allocate_device_memory();
    input_activation.copy_to_device();

    input_activation.transpose(activations_T);
    error_mat.multiply(activations_T, weight_deltas);

    Neural_Matrix prev_error(weights.cols, batch_size);
    prev_error.allocate_device_memory();

    weights.transpose(weights_T);
    weights_T.multiply(error_mat, prev_error);

    Neural_Matrix bias_deltas(biases.rows, 1);
    bias_deltas.allocate_device_memory();
    int threads = 256;
    int blocks = (biases.rows + threads - 1) / threads;
    sum_bias_gradients_kernel << <blocks, threads >> > (
        error_mat.device_data, bias_deltas.device_data, biases.rows, batch_size
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float batch_lr = learning_rate / (float)batch_size;

    bias_deltas.multiply_scalar(batch_lr);
    biases.subtract(bias_deltas);

    weight_deltas.multiply_scalar(batch_lr);
    weights.subtract(weight_deltas);

    prev_error.copy_to_host();
    return prev_error.data;
}


static std::vector<float> s_last_prediction;

Neural_Network::Neural_Network(int batch_size) {
    this->batch_size = batch_size;
}

Neural_Network::~Neural_Network() {
    for (auto layer : layers) {
        delete layer;
    }
}

void Neural_Network::add_layer(ILayer* layer) {
    layers.push_back(layer);
}

std::vector<float> Neural_Network::forward(const std::vector<float>& input) {
    std::vector<float> current_data = input;
    for (auto layer : layers) {
        current_data = layer->forward(current_data, batch_size);
    }
    s_last_prediction = current_data;
    return current_data;
}

void Neural_Network::backward(const std::vector<float>& target, float learning_rate) {
    std::vector<float> current_gradient(target.size());
    for (size_t i = 0; i < target.size(); ++i) {
        current_gradient[i] = 2.0f * (s_last_prediction[i] - target[i]);
    }

    for (int i = layers.size() - 1; i >= 0; --i) {
        current_gradient = layers[i]->backward(current_gradient, learning_rate, batch_size);
    }
}

float Neural_Network::calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target) {
    float differance = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        differance += diff * diff;
    }
    return differance / static_cast<float>(predicted.size());
}

void Neural_Network::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    for (auto layer : layers) {
        if (auto* conv = dynamic_cast<Conv2DLayer*>(layer)) {
            auto filter = conv->get_filter();
            out.write(reinterpret_cast<const char*>(filter.data()), filter.size() * sizeof(float));
        }
        else if (auto* linear = dynamic_cast<LinearLayer*>(layer)) {
            linear->get_weights().copy_to_host();
            linear->get_biases().copy_to_host();
            out.write(reinterpret_cast<const char*>(linear->get_weights().data.data()), linear->get_weights().data.size() * sizeof(float));
            out.write(reinterpret_cast<const char*>(linear->get_biases().data.data()), linear->get_biases().data.size() * sizeof(float));
        }
    }
    out.close();
}

void Neural_Network::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    for (auto layer : layers) {
        if (auto* conv = dynamic_cast<Conv2DLayer*>(layer)) {
            std::vector<float> filter(9);
            in.read(reinterpret_cast<char*>(filter.data()), filter.size() * sizeof(float));
            conv->set_filter(filter);
        }
        else if (auto* linear = dynamic_cast<LinearLayer*>(layer)) {
            in.read(reinterpret_cast<char*>(linear->get_weights().data.data()), linear->get_weights().data.size() * sizeof(float));
            linear->get_weights().copy_to_device();
            in.read(reinterpret_cast<char*>(linear->get_biases().data.data()), linear->get_biases().data.size() * sizeof(float));
            linear->get_biases().copy_to_device();
        }
    }
    in.close();
}

/// <summary>
/// 
/// </summary>
/// <param name="d_input_images"></param>
/// <param name="d_pool_gradients"></param>
/// <param name="d_conv_gradients"></param>
/// <param name="input_size"></param>
/// <param name="pool_size"></param>
/// <param name="output_size"></param>
/// <param name="batch_size"></param>
void max_pooling_backward(float* d_input_images, float* d_pool_gradients, float* d_conv_gradients, int input_size, int pool_size, int output_size, int batch_size) {

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    max_pooling_backward_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_input_images, d_pool_gradients, d_conv_gradients,
        input_size, pool_size, output_size, batch_size
        );

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

/// <summary>
/// 
/// </summary>
/// <param name="d_input_images"></param>
/// <param name="d_conv_gradients"></param>
/// <param name="d_filter_gradients"></param>
/// <param name="input_size"></param>
/// <param name="filter_size"></param>
/// <param name="conv_output_size"></param>
/// <param name="batch_size"></param>
void calculate_filter_gradients(float* d_input_images, float* d_conv_gradients, float* d_filter_gradients, int input_size, int filter_size, int conv_output_size, int batch_size) {

    
    dim3 threadsPerBlock(filter_size, filter_size, 1);
    dim3 blocksPerGrid(1, 1, 1);

    convolution_filter_gradient_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_input_images, d_conv_gradients, d_filter_gradients,
        input_size, filter_size, conv_output_size, batch_size
        );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/// <summary>
/// 
/// </summary>
/// <param name="d_input_images"></param>
/// <param name="d_output_images"></param>
/// <param name="input_size"></param>
/// <param name="pool_size"></param>
/// <param name="output_size"></param>
/// <param name="batch_size"></param>
void max_pooling(float* d_input_images, float* d_output_images,
    int input_size, int pool_size, int output_size, int batch_size) {

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    max_pooling_kernel<<<blocksPerGrid, threadsPerBlock >>>(d_input_images, d_output_images, input_size, pool_size, output_size, batch_size);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/// <summary>
/// 
/// </summary>
/// <param name="input_images"></param>
/// <param name="filter"></param>
/// <param name="output_image"></param>
/// <param name="input_size"></param>
/// <param name="filter_size"></param>
/// <param name="output_size"></param>
/// <param name="batch_size"></param>
void convolution(float* input_images, float* filter, float* output_image, int input_size, int filter_size, int output_size, int batch_size) {
    dim3 threadsPerBlock(16, 16, 1);

    dim3 blocksPerGrid(
        (output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (batch_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
    convolution_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_images, filter, output_image, input_size, filter_size, output_size, batch_size);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

/// <summary>
/// 
/// </summary>
/// <param name="r"></param>
/// <param name="c"></param>
Neural_Matrix::Neural_Matrix(int r, int c){
    rows = r;
    cols = c;
    device_data = nullptr;

    data.resize(rows * cols, 0.0f);
}

Neural_Matrix::~Neural_Matrix() {
    if (device_data != nullptr) {
        CUDA_CHECK(cudaFree(device_data));
    }
}

/// <summary>
/// 
/// </summary>
/// <param name="other"></param>
Neural_Matrix::Neural_Matrix(const Neural_Matrix& other){
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    device_data = nullptr;

    if (other.device_data != nullptr) {
        allocate_device_memory();
        CUDA_CHECK(cudaMemcpy(device_data, other.device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}


Neural_Matrix& Neural_Matrix::operator=(const Neural_Matrix& other) {
    if (this == &other) return *this;

    rows = other.rows;
    cols = other.cols;
    data = other.data;

    if (device_data != nullptr) CUDA_CHECK(cudaFree(device_data));
    device_data = nullptr;

    if (other.device_data != nullptr) {
        allocate_device_memory();
        CUDA_CHECK(cudaMemcpy(device_data, other.device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return *this;
}

void Neural_Matrix::allocate_device_memory() {

    if (device_data == nullptr) {
        CUDA_CHECK(cudaMalloc(&device_data, rows * cols * sizeof(float)));
    }
}

void Neural_Matrix::copy_to_device() {

    if (device_data == nullptr) allocate_device_memory();
    CUDA_CHECK(cudaMemcpy(device_data, data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void Neural_Matrix::copy_to_host() {

    if (device_data != nullptr) {
        CUDA_CHECK(cudaMemcpy(data.data(), device_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void Neural_Matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::uniform_real_distribution<float> dis(0.1f, 1.0f);
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f); 

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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="matrix"></param>
void Neural_Matrix::relu_derivative(const Neural_Matrix& matrix){

    int size = rows * cols;
    int threadsPerBlock = 256;

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, matrix.device_data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Neural_Matrix::apply_sigmoid(){

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

void Neural_Matrix::sigmoid_derivative(const Neural_Matrix& pre_activations) {
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, pre_activations.device_data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="scalar"></param>
void Neural_Matrix::multiply_scalar(float scalar){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiply_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, scalar, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="other"></param>
void Neural_Matrix::subtract(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    subtract_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="other"></param>
void Neural_Matrix::add(const Neural_Matrix& other){
    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, other.device_data, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="result"></param>
void Neural_Matrix::transpose(Neural_Matrix& result) const{

    int size = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, result.device_data, this->rows, this->cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}

/// <summary>
/// 
/// </summary>
/// <param name="other"></param>
/// <param name="result"></param>
void Neural_Matrix::multiply(const Neural_Matrix& other, Neural_Matrix& result) const{
    if (this->cols != other.rows) { throw std::runtime_error("Matrix dimensions are incompatible."); }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((other.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (this->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(this->device_data, other.device_data, result.device_data, this->rows, this->cols, other.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}


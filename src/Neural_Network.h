#pragma once

#include <iostream>
#include <vector>
#include <string>

class Neural_Matrix {
    private:
        int rows, cols;
        std::vector<float> data;
        float* device_data; // Ekran kartındaki (VRAM) verinin adresi

        friend class Neural_Network;

    public:
        Neural_Matrix(int r, int c);
        ~Neural_Matrix();

        // CUDA Bellek kopyalamaları için (Çift Serbest Bırakmayı Önler)
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

        Neural_Matrix transpose() const;
        Neural_Matrix multiply(const Neural_Matrix& other) const;
};

class Neural_Network {
private:
    std::vector<int> topology;
    std::vector<Neural_Matrix> weights;
    std::vector<Neural_Matrix> biases;

    std::vector<Neural_Matrix> layer_outputs;
    std::vector<Neural_Matrix> layer_activations;

public:
    Neural_Network(std::vector<int> topology);
    std::vector<float> forward(const std::vector<float>& input);

    float calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target);
    void backpropagate(const std::vector<float>& target, float learning_rate);

    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};

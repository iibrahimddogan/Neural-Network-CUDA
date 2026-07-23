#include "Neural_Network.h"
#include <algorithm>
#include <cstddef>
//#include <iterator>
#include <random>
#include <cmath>
#include <iostream>
//#include <stdatomic.h>
#include <stdexcept>
#include <vector>
#include <fstream>

Neural_Matrix::Neural_Matrix(int r, int c) {
    rows = r;
    cols = c;
    data.resize(rows * cols, 0.0f); // r * c boyutunda, içi 0 dolu dizi
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
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void Neural_Matrix::relu_derivative(const Neural_Matrix& matrix){
    if (this->data.size() != matrix.data.size()) {
        throw std::runtime_error("ReLU derivative: Matrix sizes do not match.");
    }

    for (size_t i = 0; i < data.size(); i++) {
        if (matrix.data[i] <= 0) {
            data[i] = 0;
        }
    }
}

void Neural_Matrix::apply_sigmoid(){
    for (size_t i = 0; i < data.size(); i++) {
        float x = std::max(-40.0f, std::min(40.0f, data[i]));
        data[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

void Neural_Matrix::sigmoid_derivative(const Neural_Matrix& pre_activations) {
    if (this->data.size() != pre_activations.data.size()) {
        throw std::runtime_error("Sigmoid derivative error: Matrix sizes mismatch.");
    }
    for (size_t i = 0; i < data.size(); i++) {
        float x = std::max(-40.0f, std::min(40.0f, pre_activations.data[i]));
        float sig = 1.0f / (1.0f + std::exp(-x));

        //sigmoid(x) * (1 - sigmoid(x))
        this->data[i] *= sig * (1.0f - sig);
    }
}


void Neural_Matrix::multiply_scalar(float scalar){
    for (size_t i = 0; i < data.size(); i++) {
        this->data[i] *= scalar;
    }
}

void Neural_Matrix::subtract(const Neural_Matrix& other){
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Subtract: Matrix dimensions do not match.");
    }
    for (size_t i = 0; i < data.size(); i++) {
        data[i] -= other.data[i];
    }
}

Neural_Matrix Neural_Matrix::transpose() const{
    Neural_Matrix result(this-> cols, this-> rows);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.data[j * result.cols + i] = this->data[i * this->cols + j];
        }
    }
    return result;
}

Neural_Matrix Neural_Matrix::multiply(const Neural_Matrix& other) const{
    if (this-> cols != other.rows) {
    throw std::runtime_error("Matrix dimensions are incompatible.");
    }

    Neural_Matrix result(this->rows, other.cols);

    for (int i = 0 ; i < result.rows ; i++) {
        for (int j = 0; j < result.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < this->cols; k++) {
                sum += this->data[i * this->cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }

    return result;
}

void Neural_Matrix::add(const Neural_Matrix& other){
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::runtime_error("Matrix dimensions must be the same for sum.");
    }

    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i] += other.data[i];
    }
}

Neural_Network::Neural_Network(std::vector<int> topology) {
    this->topology = topology;

    for (size_t i = 0; i < topology.size() - 1; ++i) {
        Neural_Matrix weight_matrix(topology[i + 1], topology[i]);
        weight_matrix.randomize();
        weights.push_back(weight_matrix);

        Neural_Matrix bias_matrix(topology[i + 1], 1);
        bias_matrix.randomize();
        biases.push_back(bias_matrix);
    }
}

std::vector<float> Neural_Network::forward(const std::vector<float>& input) {

    layer_activations.clear();
    layer_outputs.clear();

    Neural_Matrix current_layer(input.size(), 1);
    current_layer.data = input;

    layer_activations.push_back(current_layer);

    for (size_t i = 0; i < weights.size(); i++) {
        current_layer = weights[i].multiply(current_layer); // W.X
        current_layer.add(biases[i]);  //(W.X) + B

        layer_outputs.push_back(current_layer);

        if (i < weights.size() - 1) {
            current_layer.apply_relu();    // relu(W.X + B)
        }
        else {
            current_layer.apply_sigmoid();
        }

        layer_activations.push_back(current_layer);
    }

    return current_layer.data;
}

float Neural_Network::calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target){

    if (predicted.size() != target.size()) {
        throw std::runtime_error("Mse calculation: vector sizes not the same!");
    }

    if (predicted.empty()) {
        throw std::runtime_error("Mse calculation: vector size is zero!");
    }
    float differance = 0.0f;

    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        differance += diff * diff;
    }

    return differance / static_cast<float>(predicted.size());
}

void Neural_Network::backpropagate(const std::vector<float>& target, float learning_rate) {


    Neural_Matrix error(target.size(), 1);
    for (size_t i = 0; i < target.size(); i++) {

        float tahmin = layer_activations.back().data[i];
        error.data[i] = 2.0f * (tahmin - target[i]);
    }


    for (int i = weights.size() - 1; i >= 0; i--) {

        Neural_Matrix gradients = error;


        if (i < weights.size() - 1) {

            gradients.relu_derivative(layer_outputs[i]);
        }
        else {
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

    if (!out.is_open()) {
        throw std::runtime_error("Save model: File could not be oppened.");
    }

    size_t topology_size = topology.size();
    out.write(reinterpret_cast<const char*>(&topology_size), sizeof(size_t));

    out.write(reinterpret_cast<const char*>(topology.data()), topology_size * sizeof(int));

    for (size_t i = 0; i < weights.size(); i++) {
        size_t weights_size = weights[i].data.size();
        out.write(reinterpret_cast<const char*>(weights[i].data.data()), weights_size * sizeof(float));

        size_t biases_size = biases[i].data.size();
        out.write(reinterpret_cast<const char*>(biases[i].data.data()), biases_size * sizeof(float));
    }

    out.close();
}

void Neural_Network::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        throw std::runtime_error("Load model: File could not be oppened.");
    }

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

        size_t biases_size = biases[i].data.size();
        in.read(reinterpret_cast<char*>(biases[i].data.data()), biases_size * sizeof(float));
    }

    in.close();
}

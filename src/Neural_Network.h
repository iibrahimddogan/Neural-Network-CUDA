#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

/// @brief Runs a CUDA runtime call and throws if it fails.
///
/// Wrap every CUDA API call with this macro, e.g. `CUDA_CHECK(cudaMalloc(&ptr, bytes));`.
/// On failure it throws std::runtime_error containing the CUDA error string,
/// the source file and the line number.
///
/// @param call A CUDA runtime API call that returns cudaError_t.
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if(error != cudaSuccess){ \
        throw std::runtime_error(std::string("CUDA error code: ") + cudaGetErrorString(error) + " -> File: " + std::string(__FILE__) + " line: " + std::to_string(__LINE__)); \
    } \
} while (0)


/// @brief Row-major float matrix with a host copy and an optional GPU copy.
///
/// `data` lives on the host (CPU); `device_data` lives on the GPU.
/// The two are NOT kept in sync automatically: call copy_to_device() after
/// changing `data`, and copy_to_host() after a GPU operation if you need the
/// result on the CPU.
///
/// Element (r, c) is stored at index `r * cols + c`.
/// Most math methods (apply_relu, multiply, subtract, ...) run on the GPU and
/// only touch `device_data`.
class Neural_Matrix {
private:
    int rows, cols;
    std::vector<float> data;
    float* device_data;

    friend class LinearLayer;
    friend class Neural_Network;

public:
    /// @brief Creates an r x c matrix filled with zeros on the host.
    ///
    /// No GPU memory is allocated yet; call allocate_device_memory() or copy_to_device().
    /// @param r Number of rows.
    /// @param c Number of columns.
    Neural_Matrix(int r, int c);

    /// @brief Frees the GPU buffer (if one was allocated).
    ~Neural_Matrix();

    /// @brief Deep copy: copies host data and, if present, the GPU buffer.
    /// @param other Matrix to copy from.
    Neural_Matrix(const Neural_Matrix& other);

    /// @brief Deep copy assignment: frees this matrix's GPU buffer and copies `other` (host + GPU).
    /// @param other Matrix to copy from.
    /// @return Reference to this matrix.
    Neural_Matrix& operator=(const Neural_Matrix& other);

    /// @brief Allocates rows*cols floats on the GPU. Does nothing if already allocated.
    void allocate_device_memory();

    /// @brief Copies host `data` to the GPU (allocates first if needed).
    void copy_to_device();

    /// @brief Copies the GPU buffer back to host `data`. Does nothing if there is no GPU buffer.
    void copy_to_host();

    /// @brief Fills host `data` with uniform random values in [-0.1, 0.1].
    ///
    /// Host only: call copy_to_device() afterwards to use the values on the GPU.
    void randomize();

    /// @brief Prints host `data` to stdout, one row per line (does not copy from the GPU first).
    void print() const;

    /// @brief In-place ReLU on the GPU: x = max(0, x).
    void apply_relu();

    /// @brief Multiplies this matrix (the incoming gradient) by the ReLU derivative, on the GPU.
    ///
    /// Sets an element to 0 where the matching pre-activation is <= 0; other elements are unchanged.
    /// @param matrix Pre-activation values z (same size as this matrix, already on the GPU).
    void relu_derivative(const Neural_Matrix& matrix);

    /// @brief In-place sigmoid on the GPU: x = 1 / (1 + e^-x). Input is clamped to [-40, 40].
    void apply_sigmoid();

    /// @brief Multiplies this matrix (the incoming gradient) by the sigmoid derivative, on the GPU.
    ///
    /// gradient *= s * (1 - s), where s = sigmoid(pre_activation).
    /// @param pre_activations Pre-activation values z (same size as this matrix, already on the GPU).
    void sigmoid_derivative(const Neural_Matrix& pre_activations);

    /// @brief In-place multiply every element by a constant, on the GPU.
    /// @param scalar Value to multiply by.
    void multiply_scalar(float scalar);

    /// @brief In-place element-wise subtraction on the GPU: this -= other.
    /// @param other Matrix of the same size (already on the GPU).
    void subtract(const Neural_Matrix& other);

    /// @brief In-place element-wise addition on the GPU: this += other.
    /// @param other Matrix of the same size (already on the GPU).
    void add(const Neural_Matrix& other);

    /// @brief Writes the transpose of this matrix into `result`, on the GPU.
    /// @param result Output matrix; must already have cols x rows size and GPU memory allocated.
    void transpose(Neural_Matrix& result) const;

    /// @brief Matrix product on the GPU: result = this * other.
    ///
    /// Shapes: this is (rows x cols), other is (cols x other.cols),
    /// result must be (rows x other.cols) with GPU memory already allocated.
    /// @param other Right-hand matrix; other.rows must equal this->cols.
    /// @param result Output matrix (pre-allocated on the GPU).
    /// @throws std::runtime_error if this->cols != other.rows.
    void multiply(const Neural_Matrix& other, Neural_Matrix& result) const;
};


/// @brief Common interface for all network layers.
///
/// A layer takes a flat float vector for the whole mini-batch, returns its output
/// in forward(), and in backward() takes the gradient of the loss w.r.t. its output,
/// updates its own weights (if any) and returns the gradient w.r.t. its input.
class ILayer {
public:
    virtual ~ILayer() = default;

    /// @brief Forward pass for one mini-batch.
    /// @param input Layer input for the whole batch (layout depends on the layer).
    /// @param batch_size Number of samples in the batch.
    /// @return Layer output for the whole batch.
    virtual std::vector<float> forward(const std::vector<float>& input, int batch_size) = 0;

    /// @brief Backward pass for one mini-batch; also updates the layer's weights with SGD.
    /// @param gradient Gradient of the loss w.r.t. this layer's output (same layout as forward's output).
    /// @param learning_rate SGD step size.
    /// @param batch_size Number of samples in the batch.
    /// @return Gradient of the loss w.r.t. this layer's input (same layout as forward's input).
    virtual std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) = 0;
};


/// @brief 2D convolution layer with a fused ReLU (valid padding, stride 1, no bias).
///
/// Input layout:  [batch][in_channels][input_size][input_size]
/// Output layout: [batch][num_of_filters][output_size][output_size], output_size = input_size - filter_size + 1
/// Filter layout: [num_of_filters][in_channels][filter_size][filter_size]
/// Filters are initialised uniformly in [-0.1, 0.1].
class Conv2DLayer : public ILayer {
private:
    int in_channels;
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
    /// @brief Creates a convolution layer with random filters.
    /// @param in_channels Number of input channels (1 for grayscale images, or the previous layer's filter count).
    /// @param input_size Width/height of the square input image (e.g. 28 for MNIST).
    /// @param filter_size Width/height of the square filter (e.g. 3 for 3x3).
    /// @param num_of_filters Number of filters = number of output channels.
    Conv2DLayer(int in_channels, int input_size, int filter_size, int num_of_filters);

    /// @brief Frees all GPU buffers owned by the layer.
    ~Conv2DLayer();

    /// @brief Convolution + ReLU on the GPU.
    /// @param input [batch][in_channels][input_size][input_size] values.
    /// @param batch_size Number of images in the batch.
    /// @return [batch][num_of_filters][output_size][output_size] values (after ReLU).
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;

    /// @brief Backward pass: applies the ReLU mask, updates the filters, returns the input gradient.
    ///
    /// Filter gradients are averaged over the batch. Must be called after forward() on the same batch.
    /// @param gradient d(loss)/d(output), same layout as forward's output.
    /// @param learning_rate SGD step size.
    /// @param batch_size Number of images in the batch.
    /// @return d(loss)/d(input), same layout as forward's input.
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;

    /// @brief Returns the filter weights ([num_of_filters][in_channels][filter_size][filter_size]).
    const std::vector<float>& get_filter() const { return cnn_filter; }

    /// @brief Replaces the filter weights (used when loading a model).
    /// @param f New weights; must have the same size and layout as get_filter().
    void set_filter(const std::vector<float>& f) { cnn_filter = f; }
};


/// @brief Max-pooling layer with non-overlapping windows (stride = pool_size, no weights).
///
/// Input layout:  [batch][channels][input_size][input_size]
/// Output layout: [batch][channels][output_size][output_size], output_size = input_size / pool_size (rounded down)
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
    /// @brief Creates a max-pooling layer.
    /// @param input_size Width/height of the square input feature map (e.g. 26 after a 3x3 conv on 28x28).
    /// @param pool_size Width/height of the pooling window, also used as the stride (e.g. 2).
    /// @param channels Number of feature maps per image (= the previous conv layer's filter count).
    MaxPoolLayer(int input_size, int pool_size,int channels);

    /// @brief Frees all GPU buffers owned by the layer.
    ~MaxPoolLayer();

    /// @brief Takes the maximum of each pool_size x pool_size window, on the GPU.
    /// @param input [batch][channels][input_size][input_size] values.
    /// @param batch_size Number of images in the batch.
    /// @return [batch][channels][output_size][output_size] values.
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;

    /// @brief Routes each output gradient back to the position that held the maximum; all other positions get 0.
    /// @param gradient d(loss)/d(output), same layout as forward's output.
    /// @param learning_rate Unused (pooling has no weights).
    /// @param batch_size Number of images in the batch.
    /// @return d(loss)/d(input), same layout as forward's input.
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;
};


/// @brief Converts convolution feature maps into the column-per-sample layout used by LinearLayer.
///
/// Input layout:  [batch][channels][input_size][input_size]
/// Output layout: [feature][batch], i.e. element (feature f, sample b) is at index f * batch_size + b,
/// with flattened_size = channels * input_size * input_size features. Runs on the CPU, no weights.
class FlattenLayer : public ILayer {
private:
    int channels;
    int input_size;
    int flattened_size;

public:
    /// @brief Creates a flatten layer.
    /// @param channels Number of feature maps per image (e.g. 32).
    /// @param input_size Width/height of each square feature map (e.g. 5).
    FlattenLayer(int channels, int input_size);

    /// @brief Reorders [batch][channels][h][w] into [feature][batch].
    /// @param input Feature maps for the whole batch.
    /// @param batch_size Number of images in the batch.
    /// @return Flattened features, index = feature * batch_size + sample.
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;

    /// @brief Reorders the gradient from [feature][batch] back to [batch][channels][h][w].
    /// @param gradient d(loss)/d(output) in [feature][batch] layout.
    /// @param learning_rate Unused (no weights).
    /// @param batch_size Number of images in the batch.
    /// @return Gradient in [batch][channels][input_size][input_size] layout.
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;
};


/// @brief Fully connected layer: y = activation(W * x + b), computed on the GPU.
///
/// Input layout:  [in_features][batch]  (index = feature * batch_size + sample)
/// Output layout: [out_features][batch]
/// W is (out_features x in_features), b is (out_features x 1), both initialised uniformly in [-0.1, 0.1].
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
    /// @brief Creates a fully connected layer with random weights and biases.
    /// @param in_features Number of inputs per sample (e.g. 800 after Flatten(32, 5)).
    /// @param out_features Number of outputs per sample (e.g. 10 for the final class scores).
    /// @param act_type "relu", "sigmoid", or any other string (e.g. "") for no activation (raw logits).
    LinearLayer(int in_features, int out_features, const std::string& act_type);

    /// @brief Computes activation(W * x + b) for the whole batch.
    /// @param input [in_features][batch] values.
    /// @param batch_size Number of samples in the batch.
    /// @return [out_features][batch] values after the activation.
    std::vector<float> forward(const std::vector<float>& input, int batch_size) override;

    /// @brief Backward pass: applies the activation derivative, updates W and b with SGD, returns the input gradient.
    ///
    /// The update uses learning_rate / batch_size, so gradients are averaged over the batch.
    /// Must be called after forward() on the same batch.
    /// @param gradient d(loss)/d(output), [out_features][batch].
    /// @param learning_rate SGD step size.
    /// @param batch_size Number of samples in the batch.
    /// @return d(loss)/d(input), [in_features][batch].
    std::vector<float> backward(const std::vector<float>& gradient, float learning_rate, int batch_size) override;

    /// @brief Returns the weight matrix W (out_features x in_features).
    Neural_Matrix& get_weights() { return weights; }

    /// @brief Returns the bias vector b (out_features x 1).
    Neural_Matrix& get_biases() { return biases; }
};


/// @brief A sequential network: runs layers in the order they were added.
///
/// Training loop: forward(batch) -> backward(targets, lr). The loss is softmax + cross-entropy,
/// computed inside backward() from the logits of the last forward() call.
/// The network owns its layers and deletes them in the destructor.
class Neural_Network {
private:
    int batch_size;
    std::vector<ILayer*> layers;

public:
    /// @brief Creates an empty network.
    /// @param batch_size Number of samples passed to every forward()/backward() call (use 1 for single-image inference).
    Neural_Network(int batch_size);

    /// @brief Deletes all layers added with add_layer().
    ~Neural_Network();

    /// @brief Appends a layer to the end of the network. The network takes ownership (create it with `new`).
    /// @param layer Layer to add, e.g. `new Conv2DLayer(1, 28, 3, 16)`.
    void add_layer(ILayer* layer);

    /// @brief Runs all layers on one batch and returns the final layer's output (logits).
    /// @param input batch_size images stored one after another: [batch][channels][height][width].
    /// @return Logits in [num_classes][batch] layout (index = class * batch_size + sample).
    std::vector<float> forward(const std::vector<float>& input);

    /// @brief Softmax + cross-entropy on the last forward() output, then backpropagates through all layers and updates weights.
    /// @param target One-hot labels in [num_classes][batch] layout (index = class * batch_size + sample).
    /// @param learning_rate SGD step size.
    void backward(const std::vector<float>& target, float learning_rate);

    /// @brief Mean squared error between two vectors of the same size (helper, not used in training).
    /// @param predicted Predicted values.
    /// @param target Expected values.
    /// @return Average of (predicted - target)^2.
    float calculate_mse(const std::vector<float>& predicted, const std::vector<float>& target);

    /// @brief Saves all weights to a binary file.
    ///
    /// Format: int 9999 (magic), int layer count, then for each layer in order:
    /// Conv2D filters, or Linear weights followed by biases. Pool/Flatten layers write nothing.
    /// @param filename Path of the file to write, e.g. "data/mnist_cnn_oop_model.bin".
    void save_model(const std::string& filename) const;

    /// @brief Loads weights saved by save_model(). The network must already have the same layers in the same order.
    /// @param filename Path of the model file.
    /// @throws std::runtime_error if the file is missing, has the wrong magic number, or a different layer count.
    void load_model(const std::string& filename);
};

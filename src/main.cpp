#include "Neural_Network.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <stdexcept>

std::vector<float> load_custom_bmp(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("test.bmp file not found!");

    unsigned char info[54];
    f.read(reinterpret_cast<char*>(info), 54); // skip bmp

    std::vector<float> pixels(784);
    unsigned char rgb[3];

    for (int y = 27; y >= 0; y--) {
        for (int x = 0; x < 28; x++) {
            f.read(reinterpret_cast<char*>(rgb), 3);


            float gray = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
            pixels[y * 28 + x] = 1.0f - (gray / 255.0f); 
        }
    }
    return pixels;
}


int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void read_mnist(const std::string& image_path, const std::string& label_path,
                std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {

    std::ifstream file_images(image_path, std::ios::binary);
    std::ifstream file_labels(label_path, std::ios::binary);

    if (!file_images.is_open() || !file_labels.is_open()) {
        throw std::runtime_error("Files not found! Check the 'data' folder.");
    }

    int magic_number=0, number_of_images=0, n_rows=0, n_cols=0;

    file_images.read((char*)&magic_number, sizeof(magic_number));
    file_images.read((char*)&number_of_images, sizeof(number_of_images));
    file_images.read((char*)&n_rows, sizeof(n_rows));
    file_images.read((char*)&n_cols, sizeof(n_cols));
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    file_labels.read((char*)&magic_number, sizeof(magic_number));
    file_labels.read((char*)&magic_number, sizeof(magic_number));


    int image_count = number_of_images;

    images.reserve(image_count);
    labels.reserve(image_count);

    for (int i = 0; i < image_count; ++i) {
        std::vector<float> image(n_rows * n_cols);
        for (int p = 0; p < n_rows * n_cols; ++p) {
            unsigned char temp = 0;
            file_images.read((char*)&temp, sizeof(temp));
            image[p] = (float)temp / 255.0f;
        }
        images.push_back(std::move(image));

        unsigned char label_byte = 0;
        file_labels.read((char*)&label_byte, sizeof(label_byte));

        std::vector<float> label(10, 0.0f);
        label[(int)label_byte] = 1.0f;
        labels.push_back(std::move(label));
    }
}

// Training main: train the network and save the model
static float compute_batch_cross_entropy_loss(const std::vector<float>& logits, const std::vector<float>& targets, int batch_size) {
    int total = (int)logits.size();
    int num_classes = total / batch_size;
    float loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        // find max for stability
        float m = logits[0 * batch_size + b];
        for (int c = 1; c < num_classes; ++c) {
            float v = logits[c * batch_size + b];
            if (v > m) m = v;
        }
        // compute exp and sum
        float sum = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float v = std::exp(logits[c * batch_size + b] - m);
            sum += v;
        }
        // accumulate -log(prob_true)
        for (int c = 0; c < num_classes; ++c) {
            if (targets[c * batch_size + b] > 0.5f) { // one-hot
                float v = std::exp(logits[c * batch_size + b] - m) / sum;
                loss -= std::log(v + 1e-12f);
                break;
            }
        }
    }
    return loss / (float)batch_size;
}

int main(int argc, char** argv) {
    bool run_test = false;
    if (argc > 1) {
        std::string mode(argv[1]);
        if (mode == "test") run_test = true;
    }
    std::cout << "Reading MNIST dataset..." << std::endl;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    try {
        read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);
        std::cout << inputs.size() << " images loaded successfully\n" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (!run_test) {
        int batch_size = 128;
        Neural_Network nn(batch_size);

        // Model: Conv -> Pool -> Flatten -> MLP

        nn.add_layer(new Conv2DLayer(1, 28, 3, 16));  
        nn.add_layer(new MaxPoolLayer(26, 2, 16));    

        
        nn.add_layer(new Conv2DLayer(16, 13, 3, 32)); 
        nn.add_layer(new MaxPoolLayer(11, 2, 32));    

        
        nn.add_layer(new FlattenLayer(32, 5)); 

        nn.add_layer(new LinearLayer(800, 128, "relu"));
        nn.add_layer(new LinearLayer(128, 64, "relu"));
        nn.add_layer(new LinearLayer(64, 10, ""));

        int epochs = 20;
        float learning_rate = 0.05f;
        const int input_dim = 784;
        const int output_dim = 10;

        std::vector<float> batch_inputs(input_dim * batch_size);
        std::vector<float> batch_targets(output_dim * batch_size);

        std::cout << "\nTraining starting (Mini-Batch: 128)\n" << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int batch_count = 0;

            for (size_t i = 0; i < inputs.size(); i += batch_size) {
                if (i + batch_size > inputs.size()) break;

                // pack inputs and targets in layout [class * batch + sample]
                for (int b = 0; b < batch_size; ++b) {
                    for (int p = 0; p < input_dim; ++p) {
                        batch_inputs[b * input_dim + p] = inputs[i + b][p];
                    }
                }

                for (int t = 0; t < output_dim; ++t) {
                    for (int b = 0; b < batch_size; ++b) {
                        batch_targets[t * batch_size + b] = targets[i + b][t];
                    }
                }

                std::vector<float> logits = nn.forward(batch_inputs);

                float loss = compute_batch_cross_entropy_loss(logits, batch_targets, batch_size);
                total_loss += loss;

                nn.backward(batch_targets, learning_rate);

                batch_count++;
            }
            std::cout << "Epoch: " << epoch + 1 << "/" << epochs
                << " | Average Loss: " << (total_loss / (float)batch_count) << std::endl;
        }

        std::cout << "\nSaving model..." << std::endl;
        nn.save_model("data/mnist_cnn_oop_model.bin");
        std::cout << "Save successful\n" << std::endl;

        return 0;
    }
    else {
        // Test mode: load 10k test images and evaluate saved model
        std::vector<std::vector<float>> test_inputs;
        std::vector<std::vector<float>> test_targets;
        try {
            read_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", test_inputs, test_targets);
            std::cout << test_inputs.size() << " images loaded successfully\n" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Test file not found. Error: " << e.what() << std::endl;
            return 1;
        }

        Neural_Network nn(1);

        // Model: Conv -> Pool -> Flatten -> MLP
        nn.add_layer(new Conv2DLayer(1, 28, 3, 16));  
        nn.add_layer(new MaxPoolLayer(26, 2, 16));    

        
        nn.add_layer(new Conv2DLayer(16, 13, 3, 32)); 
        nn.add_layer(new MaxPoolLayer(11, 2, 32));    

        nn.add_layer(new FlattenLayer(32, 5)); 

        nn.add_layer(new LinearLayer(800, 128, "relu"));
        nn.add_layer(new LinearLayer(128, 64, "relu"));
        nn.add_layer(new LinearLayer(64, 10, "")); // linear logits

        std::cout << "Loading trained model (mnist_cnn_oop_model.bin) into VRAM...\n" << std::endl;
        try {
            nn.load_model("data/mnist_cnn_oop_model.bin");
            std::cout << "Starting test\n" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Model yuklenirken hata olustu: " << e.what() << std::endl;
            return 1;
        }

        int correct_count = 0;
        int total_tests = test_inputs.size();

        for (int i = 0; i < total_tests; ++i) {
            std::vector<float> prediction = nn.forward(test_inputs[i]);
            int ai_prediction = 0;
            float max_score = prediction[0];
            for (int j = 1; j < 10; ++j) {
                if (prediction[j] > max_score) {
                    max_score = prediction[j];
                    ai_prediction = j;
                }
            }

            int true_label = 0;
            for (int j = 0; j < 10; ++j) {
                if (test_targets[i][j] == 1.0f) {
                    true_label = j;
                    break;
                }
            }

            if (ai_prediction == true_label) {
                correct_count++;
            }
        }

        float accuracy_percent = ((float)correct_count / (float)total_tests) * 100.0f;
        std::cout << "                                    " << std::endl;
        std::cout << "            TEST RESULTS            " << std::endl;
        std::cout << "                                    " << std::endl;
        std::cout << "Total Tests    : " << total_tests << std::endl;
        std::cout << "Correct        : " << correct_count << std::endl;
        std::cout << "Incorrect      : " << (total_tests - correct_count) << std::endl;
        std::cout << "Accuracy       : %" << accuracy_percent << std::endl;
        std::cout << "                                    " << std::endl;

        return 0;
    }
}
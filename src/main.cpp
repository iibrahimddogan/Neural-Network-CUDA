#include "Neural_Network.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <stdexcept>

std::vector<float> load_custom_bmp(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("test.bmp dosyasi bulunamadi!");

    unsigned char info[54];
    f.read(reinterpret_cast<char*>(info), 54); // BMP Başlığını atla

    std::vector<float> pixels(784);
    unsigned char rgb[3];

    // BMP resimleri pikselleri aşağıdan yukarı kaydeder, bu yüzden tersten (y=27'den) okuyoruz
    for (int y = 27; y >= 0; y--) {
        for (int x = 0; x < 28; x++) {
            f.read(reinterpret_cast<char*>(rgb), 3);

            // Renkleri tersine çevir (Paint'te beyaz arka plana siyah çizdiğin için)
            float gray = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
            pixels[y * 28 + x] = 1.0f - (gray / 255.0f); // 0.0 siyah, 1.0 beyaz
        }
    }
    return pixels;
}

// İkili dosya okuma çevirmeni
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// MNIST resimlerini okuyan fonksiyon (Sınır kaldırıldı!)
void read_mnist(const std::string& image_path, const std::string& label_path,
                std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {

    std::ifstream file_images(image_path, std::ios::binary);
    std::ifstream file_labels(label_path, std::ios::binary);

    if (!file_images.is_open() || !file_labels.is_open()) {
        throw std::runtime_error("Dosyalar bulunamadi! 'data' klasorunu kontrol et.");
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

    // ÖNEMLİ: Artık 1000 değil, dosyadaki gerçek resim sayısı (60.000) kadar okuyoruz!
    int resim_sayisi = number_of_images;

    images.reserve(resim_sayisi);
    labels.reserve(resim_sayisi);

    for (int i = 0; i < resim_sayisi; ++i) {
        std::vector<float> image(n_rows * n_cols);
        for (int p = 0; p < n_rows * n_cols; ++p) {
            unsigned char temp = 0;
            file_images.read((char*)&temp, sizeof(temp));
            image[p] = (float)temp / 255.0f;
        }
        images.push_back(std::move(image));

        unsigned char etiket = 0;
        file_labels.read((char*)&etiket, sizeof(etiket));

        std::vector<float> label(10, 0.0f);
        label[(int)etiket] = 1.0f;
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
    std::cout << "MNIST Veri Seti Okunuyor (Tam 60.000 Resim)..." << std::endl;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    try {
        read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);
        std::cout << inputs.size() << " Adet Resim Basariyla Yuklendi!\n" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Hata: " << e.what() << std::endl;
        return 1;
    }

    if (!run_test) {
        int batch_size = 128;
        Neural_Network nn(batch_size);

        // Model: Conv -> Pool -> Flatten -> MLP
// 1. BLOK: İlk Evrişim ve Havuzlama (Giriş: 28x28 Siyah-Beyaz Resim)
        nn.add_layer(new Conv2DLayer(1, 28, 3, 16));  // 28x28 girer, 16 Kanal 26x26 çıkar
        nn.add_layer(new MaxPoolLayer(26, 2, 16));    // 26x26 girer, 16 Kanal 13x13 çıkar

        // 2. BLOK: İKİNCİ EVRİŞİM (Gerçek Derinlik Burası!)
        // 16 kanallı 13x13 veriyi alıyoruz. 3x3 boyutunda 32 YENİ FİLTRE uyguluyoruz!
        nn.add_layer(new Conv2DLayer(16, 13, 3, 32)); // 13x13 girer, 32 Kanal 11x11 çıkar
        nn.add_layer(new MaxPoolLayer(11, 2, 32));    // 11x11 girer, 32 Kanal 5x5 çıkar

        // 3. BLOK: Düzleştirme (Flatten)
        // Artık elimizde 32 kanal var ve resimler 5x5 boyutuna kadar küçüldü.
        nn.add_layer(new FlattenLayer(32, 5)); 

        // 4. BLOK: Karar Ağı (MLP)
        // 32 kanal * 5 * 5 = 800 nöronluk giriş (Eskiden 2704'tü, ağ şimdi daha hafif ve zeki!)
        nn.add_layer(new LinearLayer(800, 128, "relu"));
        nn.add_layer(new LinearLayer(128, 64, "relu"));
        nn.add_layer(new LinearLayer(64, 10, "")); // linear logits

        int epochs = 20;
        float learning_rate = 0.05f;
        const int input_dim = 784;
        const int output_dim = 10;

        std::vector<float> batch_inputs(input_dim * batch_size);
        std::vector<float> batch_targets(output_dim * batch_size);

        std::cout << "\nYeni OOP Mimariyle Tam Ogrenen CNN Egitimi Basliyor! (Mini-Batch: 128)\n" << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float toplam_hata = 0.0f;
            int batch_sayisi = 0;

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
                toplam_hata += loss;

                nn.backward(batch_targets, learning_rate);

                batch_sayisi++;
            }

            std::cout << "Tur (Epoch): " << epoch + 1 << "/" << epochs
                << " | Ortalama Loss: " << (toplam_hata / (float)batch_sayisi) << std::endl;
        }

        std::cout << "\nMuazzam egitim bitti! Yeni agirliklar ve ogrenilen filtre kaydediliyor..." << std::endl;
        nn.save_model("data/mnist_cnn_oop_model.bin");
        std::cout << "KAYIT BASARILI!\n" << std::endl;

        return 0;
    }
    else {
        // Test mode: load 10k test images and evaluate saved model
        std::vector<std::vector<float>> test_inputs;
        std::vector<std::vector<float>> test_targets;
        try {
            read_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", test_inputs, test_targets);
            std::cout << test_inputs.size() << " Adet Test Resmi Basariyla Yuklendi!\n" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Test dosyalari bulunamadi! Hata: " << e.what() << std::endl;
            return 1;
        }

        Neural_Network nn(1);

        // Model: Conv -> Pool -> Flatten -> MLP
// 1. BLOK: İlk Evrişim ve Havuzlama (Giriş: 28x28 Siyah-Beyaz Resim)
        nn.add_layer(new Conv2DLayer(1, 28, 3, 16));  // 28x28 girer, 16 Kanal 26x26 çıkar
        nn.add_layer(new MaxPoolLayer(26, 2, 16));    // 26x26 girer, 16 Kanal 13x13 çıkar

        // 2. BLOK: İKİNCİ EVRİŞİM (Gerçek Derinlik Burası!)
        // 16 kanallı 13x13 veriyi alıyoruz. 3x3 boyutunda 32 YENİ FİLTRE uyguluyoruz!
        nn.add_layer(new Conv2DLayer(16, 13, 3, 32)); // 13x13 girer, 32 Kanal 11x11 çıkar
        nn.add_layer(new MaxPoolLayer(11, 2, 32));    // 11x11 girer, 32 Kanal 5x5 çıkar

        // 3. BLOK: Düzleştirme (Flatten)
        // Artık elimizde 32 kanal var ve resimler 5x5 boyutuna kadar küçüldü.
        nn.add_layer(new FlattenLayer(32, 5)); 

        // 4. BLOK: Karar Ağı (MLP)
        // 32 kanal * 5 * 5 = 800 nöronluk giriş (Eskiden 2704'tü, ağ şimdi daha hafif ve zeki!)
        nn.add_layer(new LinearLayer(800, 128, "relu"));
        nn.add_layer(new LinearLayer(128, 64, "relu"));
        nn.add_layer(new LinearLayer(64, 10, "")); // linear logits

        std::cout << "Tam Ogrenen CNN zekasi (mnist_cnn_oop_model.bin) VRAM'e yukleniyor...\n" << std::endl;
        try {
            nn.load_model("data/mnist_cnn_oop_model.bin");
            std::cout << "Zeka basariyla canlandirildi! Buyuk sinav basliyor...\n" << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "Model yuklenirken hata olustu: " << e.what() << std::endl;
            return 1;
        }

        int dogru_bilinen = 0;
        int toplam_soru = test_inputs.size();

        for (int i = 0; i < toplam_soru; ++i) {
            std::vector<float> tahmin = nn.forward(test_inputs[i]);
            int ai_tahmini = 0;
            float en_yuksek_skor = tahmin[0];
            for (int j = 1; j < 10; ++j) {
                if (tahmin[j] > en_yuksek_skor) {
                    en_yuksek_skor = tahmin[j];
                    ai_tahmini = j;
                }
            }

            int gercek_cevap = 0;
            for (int j = 0; j < 10; ++j) {
                if (test_targets[i][j] == 1.0f) {
                    gercek_cevap = j;
                    break;
                }
            }

            if (ai_tahmini == gercek_cevap) {
                dogru_bilinen++;
            }
        }

        float basari_yuzdesi = ((float)dogru_bilinen / (float)toplam_soru) * 100.0f;
        std::cout << "=====================================" << std::endl;
        std::cout << "        10.000 SORULUK SINAV SONUCU  " << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Toplam Soru    : " << toplam_soru << std::endl;
        std::cout << "Dogru Cevap    : " << dogru_bilinen << std::endl;
        std::cout << "Yanlis Cevap   : " << (toplam_soru - dogru_bilinen) << std::endl;
        std::cout << "Genel Basari   : %" << basari_yuzdesi << std::endl;
        std::cout << "=====================================" << std::endl;

        return 0;
    }
}
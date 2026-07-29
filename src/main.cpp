#include "Neural_Network.h"
#include <iostream>
#include <vector>
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

    for (int i = 0; i < resim_sayisi; ++i) {
        std::vector<float> image(n_rows * n_cols);
        for (int p = 0; p < n_rows * n_cols; ++p) {
            unsigned char temp = 0;
            file_images.read((char*)&temp, sizeof(temp));
            image[p] = (float)temp / 255.0f;
        }
        images.push_back(image);

        unsigned char etiket = 0;
        file_labels.read((char*)&etiket, sizeof(etiket));

        std::vector<float> label(10, 0.0f);
        label[(int)etiket] = 1.0f;
        labels.push_back(label);
    }
}


// ... (Yukarıdaki okuma fonksiyonları read_mnist vb. tamamen aynı kalacak)

int main() {
    std::cout << "MNIST Veri Seti Okunuyor (Tam 60.000 Resim)..." << std::endl;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);
    std::cout << inputs.size() << " Adet Resim Basariyla Yuklendi!\n" << std::endl;

    // --- YENİ BATCH MANTIKLI AĞ KURULUMU ---
    int batch_size = 128;
    std::vector<int> topology = { 784, 128, 10 };

    // Ağımızı artık batch_size parametresiyle başlatıyoruz
    Neural_Network nn(topology, batch_size);

    int epochs = 10;
    // Batch (Toplu) kullandığımız için 128 resmin ortalamasını alıyoruz. 
    // Bu yüzden öğrenme oranını (learning rate) biraz artırabiliriz.
    float learning_rate = 0.05f;

    std::cout << "60.000 Resimlik Egitim Basliyor! (Mini-Batch: 128)\n" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float toplam_hata = 0.0f;
        int batch_sayisi = 0;

        // İleriye doğru 1'er 1'er değil, 128'er 128'er (batch_size) atlayarak gidiyoruz
        for (size_t i = 0; i < inputs.size(); i += batch_size) {

            // 60.000 resim 128'e tam bölünmez (Sonda 96 resim artar). 
            // Boyut uyuşmazlığı ve program çökmesi olmasın diye o son eksik paketi atlıyoruz (PyTorch'ta da kural böyledir)
            if (i + batch_size > inputs.size()) break;

            // GPU'daki matris çarpım kuralına (Row-Major) uymak için pikselleri gruplayarak paketliyoruz
            std::vector<float> batch_inputs(784 * batch_size);
            for (int p = 0; p < 784; ++p) {
                for (int b = 0; b < batch_size; ++b) {
                    batch_inputs[p * batch_size + b] = inputs[i + b][p];
                }
            }

            // Aynı kuralı hedefler (targets) için de yapıyoruz
            std::vector<float> batch_targets(10 * batch_size);
            for (int t = 0; t < 10; ++t) {
                for (int b = 0; b < batch_size; ++b) {
                    batch_targets[t * batch_size + b] = targets[i + b][t];
                }
            }

            // 128 RESMİ TEK HAMLEDE AĞA GÖNDER!
            std::vector<float> tahmin = nn.forward(batch_inputs);

            // Sadece bilgi amaçlı ekrana yazdırmak için ortalama hatayı hesaplıyoruz
            for (int t = 0; t < 10; ++t) {
                for (int b = 0; b < batch_size; ++b) {
                    float fark = batch_targets[t * batch_size + b] - tahmin[t * batch_size + b];
                    toplam_hata += fark * fark;
                }
            }

            // 128 RESMİN HATASINI TEK HAMLEDE GERİYE YAY!
            nn.backpropagate(batch_targets, learning_rate);
            batch_sayisi++;
        }

        // Hata oranını hesaplarken toplam resim sayısına (işlenen) bölüyoruz
        int islenen_resim = batch_sayisi * batch_size;
        std::cout << "Tur (Epoch): " << epoch + 1 << "/" << epochs
            << " | Ortalama Hata (Loss): " << (toplam_hata / (float)islenen_resim) << std::endl;
    }

    std::cout << "\nMuazzam egitim bitti! Yeni agirliklar kaydediliyor..." << std::endl;
    nn.save_model("mnist_model_batch128.bin");
    std::cout << "KAYIT BASARILI! Gercek GPU gucuyle egitilen model hazir.\n" << std::endl;

    return 0;
}
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



//int main() {
//    std::cout << "10.000 Resimlik Orijinal MNIST Test Veri Seti Okunuyor..." << std::endl;
//    std::vector<std::vector<float>> test_inputs;
//    std::vector<std::vector<float>> test_targets;
//
//    try {
//        read_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", test_inputs, test_targets);
//        std::cout << test_inputs.size() << " Adet Test Resmi Basariyla Yuklendi!\n" << std::endl;
//    }
//    catch (const std::exception& e) {
//        std::cout << "Test dosyalari bulunamadi! Hata: " << e.what() << std::endl;
//        return 1;
//    }
//
//    // 1. Yeni OOP Mimarisiyle Ağın İskeletini Kur (Test için batch_size = 1)
//    Neural_Network nn(1);
//    nn.add_layer(new Conv2DLayer(28, 3));
//    nn.add_layer(new MaxPoolLayer(26, 2));
//    nn.add_layer(new LinearLayer(169, 128, "relu"));
//    nn.add_layer(new LinearLayer(128, 64, "relu"));
//    nn.add_layer(new LinearLayer(64, 10, "sigmoid"));
//
//    // 2. Eğitilmiş Modeli Yükle
//    std::cout << "Tam Ogrenen CNN zekasi (mnist_cnn_oop_model.bin) VRAM'e yukleniyor...\n" << std::endl;
//    try {
//        nn.load_model("data/mnist_cnn_oop_model.bin");
//        std::cout << "Zeka basariyla canlandirildi! Buyuk sinav basliyor...\n" << std::endl;
//    }
//    catch (const std::exception& e) {
//        std::cout << "Model yuklenirken hata olustu: " << e.what() << std::endl;
//        return 1;
//    }
//
//    int dogru_bilinen = 0;
//    int toplam_soru = test_inputs.size();
//
//    // 10.000 soruluk sınav döngüsü
//    for (int i = 0; i < toplam_soru; ++i) {
//
//        // Zeka sadece resmi görüyor (İleri Besleme)
//        std::vector<float> tahmin = nn.forward(test_inputs[i]);
//
//        // Yapay zekanın en emin olduğu rakamı bul
//        int ai_tahmini = 0;
//        float en_yuksek_skor = tahmin[0];
//        for (int j = 1; j < 10; ++j) {
//            if (tahmin[j] > en_yuksek_skor) {
//                en_yuksek_skor = tahmin[j];
//                ai_tahmini = j;
//            }
//        }
//
//        // Gerçek cevabı bul
//        int gercek_cevap = 0;
//        for (int j = 0; j < 10; ++j) {
//            if (test_targets[i][j] == 1.0f) {
//                gercek_cevap = j;
//                break;
//            }
//        }
//
//        // Eğer zekanın tahmini gerçek cevapla eşleşiyorsa hanesine 1 puan yaz
//        if (ai_tahmini == gercek_cevap) {
//            dogru_bilinen++;
//        }
//    }
//
//    // --- KARNE ---
//    float basari_yuzdesi = ((float)dogru_bilinen / (float)toplam_soru) * 100.0f;
//    std::cout << "=====================================" << std::endl;
//    std::cout << "        10.000 SORULUK SINAV SONUCU  " << std::endl;
//    std::cout << "=====================================" << std::endl;
//    std::cout << "Toplam Soru    : " << toplam_soru << std::endl;
//    std::cout << "Dogru Cevap    : " << dogru_bilinen << std::endl;
//    std::cout << "Yanlis Cevap   : " << (toplam_soru - dogru_bilinen) << std::endl;
//    std::cout << "Genel Basari   : %" << basari_yuzdesi << std::endl;
//    std::cout << "=====================================" << std::endl;
//
//    return 0;
//}

//int main() {
//    // 1. Yeni OOP Mimarisiyle Ağın İskeletini Kur (Test için batch_size = 1)
//    Neural_Network nn(1);
//    nn.add_layer(new Conv2DLayer(28, 3));
//    nn.add_layer(new MaxPoolLayer(26, 2));
//    nn.add_layer(new LinearLayer(169, 128, "relu"));
//    nn.add_layer(new LinearLayer(128, 64, "relu"));
//    nn.add_layer(new LinearLayer(64, 10, "sigmoid"));
//
//    // 2. Eğittiğimiz beyni yükle
//    std::cout << "Tam Ogrenen CNN zekasi (mnist_cnn_oop_model.bin) VRAM'e yukleniyor...\n" << std::endl;
//    try {
//        nn.load_model("data/mnist_cnn_oop_model.bin");
//    }
//    catch (const std::exception& e) {
//        std::cout << "Model yuklenirken hata olustu: " << e.what() << std::endl;
//        return 1;
//    }
//
//    // 3. Klasördeki tüm test dosyalarının listesi
//    std::vector<std::string> dosyalar = {
//        "1test.bmp", "2test.bmp", "3test.bmp", "4test.bmp",
//        "5test.bmp", "6test.bmp", "7test.bmp", "8test.bmp", "9test.bmp"
//    };
//
//    std::cout << "\n==========================================" << std::endl;
//    std::cout << "   YENI CNN ILE PAINT CIZIMLERI TESTI     " << std::endl;
//    std::cout << "==========================================\n" << std::endl;
//
//    for (const auto& dosya_adi : dosyalar) {
//        std::string tam_yol = "data/" + dosya_adi;
//
//        try {
//            std::vector<float> benim_resmim = load_custom_bmp(tam_yol);
//
//            // DİKKAT: Test için sadece ileri besleme (forward) yapıyoruz
//            std::vector<float> tahmin = nn.forward(benim_resmim);
//
//            int ai_karari = 0;
//            float en_yuksek_skor = tahmin[0];
//            for (int i = 1; i < 10; i++) {
//                if (tahmin[i] > en_yuksek_skor) {
//                    en_yuksek_skor = tahmin[i];
//                    ai_karari = i;
//                }
//            }
//
//            std::cout << "Dosya: " << dosya_adi
//                << "  -->  Tahmin: [ " << ai_karari << " ]"
//                << "  (Eminlik: %" << (en_yuksek_skor * 100.0f) << ")" << std::endl;
//
//        }
//        catch (const std::exception& e) {
//            std::cout << "Dosya okunamadi: " << dosya_adi << " (" << e.what() << ")" << std::endl;
//        }
//    }
//
//    std::cout << "\n==========================================" << std::endl;
//
//    return 0;
//}


// int main() {
//     std::cout << "MNIST Veri Seti Okunuyor (Tam 60.000 Resim)..." << std::endl;
//     std::vector<std::vector<float>> inputs;
//     std::vector<std::vector<float>> targets;

//     try {
//         read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);
//         std::cout << inputs.size() << " Adet Resim Basariyla Yuklendi!\n" << std::endl;
//     }
//     catch (const std::exception& e) {
//         std::cout << "Hata: " << e.what() << std::endl;
//         return 1;
//     }

//     int batch_size = 128;

//     // --- İŞTE YENİ PYTORCH TARZI LEGO MİMARİMİZ! ---
// // --- İŞTE YENİ 16 FİLTRELİ DEV MİMARİMİZ! ---
//     Neural_Network nn(batch_size);

//     // 1. Evrişim (16 Filtre) ve Havuzlama
//     nn.add_layer(new Conv2DLayer(28, 3, 16));  // YENİ: 28x28 resme 3x3 boyutunda 16 FARKLI filtre uygula
//     nn.add_layer(new MaxPoolLayer(26, 2, 16)); // YENİ: Çıkan 16 farklı 26x26 haritayı bağımsız küçült

//     // 2. Klasik Karar Ağı (MLP)
//     // YENİ: 16 kanal * 13 * 13 = 2704 nöronluk devasa giriş!
//     nn.add_layer(new LinearLayer(2704, 128, "relu")); 
//     nn.add_layer(new LinearLayer(128, 64, "relu"));
//     nn.add_layer(new LinearLayer(64, 10, "sigmoid"));
//     // ------------------------------------------------
//     // ------------------------------------------------

//     int epochs = 10;
//     float learning_rate = 0.05f;
//     const int input_dim = 784;
//     const int output_dim = 10;

//     std::vector<float> batch_inputs(input_dim * batch_size);
//     std::vector<float> batch_targets(output_dim * batch_size);

//     std::cout << "\nYeni OOP Mimariyle Tam Ogrenen CNN Egitimi Basliyor! (Mini-Batch: 128)\n" << std::endl;

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float toplam_hata = 0.0f;
//         int batch_sayisi = 0;

//         for (size_t i = 0; i < inputs.size(); i += batch_size) {
//             if (i + batch_size > inputs.size()) break;

//             // Verileri paketle (Transpoze işlemleri artık içeride hallediliyor)
//             for (int b = 0; b < batch_size; ++b) {
//                 for (int p = 0; p < input_dim; ++p) {
//                     batch_inputs[b * input_dim + p] = inputs[i + b][p];
//                 }
//             }

//             for (int t = 0; t < output_dim; ++t) {
//                 for (int b = 0; b < batch_size; ++b) {
//                     batch_targets[t * batch_size + b] = targets[i + b][t];
//                 }
//             }

//             // Mükemmel derecede sadeleşen kullanım:
//             std::vector<float> tahmin = nn.forward(batch_inputs);

//             toplam_hata += nn.calculate_mse(tahmin, batch_targets);

//             nn.backward(batch_targets, learning_rate);

//             batch_sayisi++;
//         }

//         std::cout << "Tur (Epoch): " << epoch + 1 << "/" << epochs
//             << " | Ortalama Hata (Loss): " << (toplam_hata / (float)batch_sayisi) << std::endl;
//     }

//     std::cout << "\nMuazzam egitim bitti! Yeni agirliklar ve ogrenilen filtre kaydediliyor..." << std::endl;
//     nn.save_model("data/mnist_cnn_oop_model.bin");
//     std::cout << "KAYIT BASARILI!\n" << std::endl;

//     return 0;
// }



int main() {
    std::cout << "10.000 Resimlik Orijinal MNIST Test Veri Seti Okunuyor..." << std::endl;
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

    // 1. Yeni OOP Mimarisiyle Ağın İskeletini Kur (Test için batch_size = 1)
// 1. Yeni 16 Filtreli Mimarimiz (Eğitilen modelle birebir aynı olmalı!)
    Neural_Network nn(1); // Test için batch_size = 1
    nn.add_layer(new Conv2DLayer(28, 3, 16));  // 16 Filtre
    nn.add_layer(new MaxPoolLayer(26, 2, 16)); 
    nn.add_layer(new LinearLayer(2704, 128, "relu")); // 2704 Nöron giriş
    nn.add_layer(new LinearLayer(128, 64, "relu"));
    nn.add_layer(new LinearLayer(64, 10, "sigmoid"));

    // 2. Eğitilmiş Modeli Yükle
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

    // 10.000 soruluk sınav döngüsü
    for (int i = 0; i < toplam_soru; ++i) {

        // Zeka sadece resmi görüyor (İleri Besleme)
        std::vector<float> tahmin = nn.forward(test_inputs[i]);

        // Yapay zekanın en emin olduğu rakamı bul
        int ai_tahmini = 0;
        float en_yuksek_skor = tahmin[0];
        for (int j = 1; j < 10; ++j) {
            if (tahmin[j] > en_yuksek_skor) {
                en_yuksek_skor = tahmin[j];
                ai_tahmini = j;
            }
        }

        // Gerçek cevabı bul
        int gercek_cevap = 0;
        for (int j = 0; j < 10; ++j) {
            if (test_targets[i][j] == 1.0f) {
                gercek_cevap = j;
                break;
            }
        }

        // Eğer zekanın tahmini gerçek cevapla eşleşiyorsa hanesine 1 puan yaz
        if (ai_tahmini == gercek_cevap) {
            dogru_bilinen++;
        }
    }

    // --- KARNE ---
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
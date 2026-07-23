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


int main() {
    std::cout << "MNIST Veri Seti Okunuyor (Tam 60.000 Resim)..." << std::endl;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);
    std::cout << inputs.size() << " Adet Resim Basariyla Yuklendi!\n" << std::endl;

    std::vector<int> topology = {784, 512, 128, 10};
    Neural_Network nn(topology);


    int epochs = 10;
    float learning_rate = 0.001f;

    std::cout << "60.000 Resimlik Egitim Basliyor! (Release modunda hizli surecektir)\n" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float toplam_hata = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {

            std::vector<float> tahmin = nn.forward(inputs[i]);

            for (size_t j = 0; j < 10; ++j) {
                float fark = targets[i][j] - tahmin[j];
                toplam_hata += fark * fark;
            }

            nn.backpropagate(targets[i], learning_rate);
        }

        std::cout << "Tur (Epoch): " << epoch + 1 << "/" << epochs
                  << " | Ortalama Hata (Loss): " << toplam_hata / inputs.size() << std::endl;
    }

    // --- EĞİTİLMİŞ BEYNİ KAYDET ---
    std::cout << "\nMuazzam egitim bitti! Yeni agirliklar kaydediliyor..." << std::endl;
    nn.save_model("mnist_model(784_512_128_10).bin");
    std::cout << "KAYIT BASARILI! Eski beynin yerini 60.000 resimlik yeni zeka aldi.\n" << std::endl;

    return 0;
}

/* genel test
int main() {
    std::cout << "MNIST Veri Seti Okunuyor..." << std::endl;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    read_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", inputs, targets);

    std::vector<int> topology = {784, 128, 10};
    Neural_Network nn(topology);

    // --- EĞİTİLMİŞ BEYNİ YÜKLE ---
    std::cout << "60.000 resimle egitilmis beyin (mnist_model.bin) yukleniyor..." << std::endl;
    try {
        nn.load_model("mnist_model.bin");
        std::cout << "Beyin basariyla canlandirildi!\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Hata: " << e.what() << std::endl;
        return 1;
    }

    // --- BÜYÜK SINAV (DOĞRULUK TESTİ) ---
    int test_edilecek_resim_sayisi = 10000; // İlk 10.000 resmi soralım
    int dogru_bilinen = 0;

    std::cout << test_edilecek_resim_sayisi << " adet resim uzerinde sinav basliyor...\n" << std::endl;

    for (int i = 0; i < test_edilecek_resim_sayisi; ++i) {
        std::vector<float> tahmin = nn.forward(inputs[i]);

        // Yapay zekanın en yüksek puan verdiği rakamı bul (Örn: 0.95 ile 7 dedi)
        int ai_tahmini = 0;
        float en_yuksek_skor = tahmin[0];
        for (int j = 1; j < 10; ++j) {
            if (tahmin[j] > en_yuksek_skor) {
                en_yuksek_skor = tahmin[j];
                ai_tahmini = j;
            }
        }

        // Gerçek cevabı bul (Hangi index 1.0 ise cevap odur)
        int gercek_cevap = 0;
        for (int j = 0; j < 10; ++j) {
            if (targets[i][j] == 1.0f) {
                gercek_cevap = j;
                break;
            }
        }

        // Tahmin doğruysa skoru artır
        if (ai_tahmini == gercek_cevap) {
            dogru_bilinen++;
        }
    }

    // Sonuçları Hesapla ve Yazdır
    float basari_yuzdesi = ((float)dogru_bilinen / test_edilecek_resim_sayisi) * 100.0f;
    std::cout << "--- SINAV SONUCU ---" << std::endl;
    std::cout << "Toplam Soru: " << test_edilecek_resim_sayisi << std::endl;
    std::cout << "Dogru Cevap: " << dogru_bilinen << std::endl;
    std::cout << "Yanlis Cevap: " << (test_edilecek_resim_sayisi - dogru_bilinen) << std::endl;
    std::cout << "Genel Basari Orani (Accuracy): %" << basari_yuzdesi << std::endl;

    return 0;
}
*/
// int main() {
//     // 1. Ağın iskeletini kur
//     std::vector<int> topology = {784, 128, 10};
//     Neural_Network nn(topology);

//     // 2. Eğittiğimiz beyni yükle
//     std::cout << "Zeki beyin (mnist_model.bin) uyanmis durumda..." << std::endl;
//     nn.load_model("data/mnist_model.bin");

//     // 3. Kendi çizdiğin resmi yükle
//     std::vector<float> benim_resmim = load_custom_bmp("data/5test.bmp");

//     // 4. Tahmin ettir
//     std::vector<float> tahmin = nn.forward(benim_resmim);

//     // 5. En yüksek oranlı sonucu bul
//     int ai_karari = 0;
//     float en_yuksek_skor = tahmin[0];
//     for(int i = 1; i < 10; i++) {
//         if(tahmin[i] > en_yuksek_skor) {
//             en_yuksek_skor = tahmin[i];
//             ai_karari = i;
//         }
//     }

//     std::cout << "-----------------------------------" << std::endl;
//     std::cout << "Yapay Zeka: GORDUGUM BU RAKAM KESINLIKLE -> " << ai_karari << std::endl;
//     std::cout << "Eminlik Orani: %" << (en_yuksek_skor * 100.0f) << std::endl;
//     std::cout << "-----------------------------------" << std::endl;

//     return 0;
// }

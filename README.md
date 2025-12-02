# Fashion MNIST Stacked Autoencoder & PSNR Analysis

Bu proje, **TensorFlow** ve **Keras** kullanılarak **Fashion MNIST** veri seti üzerinde görüntü sıkıştırma ve yeniden oluşturma (reconstruction) işlemi gerçekleştiren bir **Stacked Autoencoder (Yığıtlı Otokodlayıcı)** modelidir.

Proje temel bir eğitim materyali üzerine inşa edilmiş olup; model mimarisi, hiperparametreler ve eğitim stratejileri optimize edilerek **hata oranı (loss) %97 oranında düşürülmüş** ve detaylı performans metrikleri eklenmiştir.

## 🚀 Proje Özellikleri ve İyileştirmeler

Bu çalışmada, standart bir Autoencoder yapısı üzerine önemli iyileştirmeler yapılmıştır:

* **Derin Mimarisi (Deep Architecture):** Model, veriyi daha iyi temsil edebilmek için çok katmanlı (256 -> 128 -> 64) bir Encoder ve simetrik bir Decoder yapısına sahiptir.
* **Early Stopping:** Aşırı öğrenmeyi (overfitting) önlemek ve en iyi ağırlıkları (best weights) korumak için `EarlyStopping` mekanizması entegre edilmiştir.
* **PSNR Analizi:** Modelin başarısını ölçmek için sadece Loss değeri değil, görüntü kalitesini ölçen **PSNR (Peak Signal-to-Noise Ratio)** metriği de hesaplanmıştır.

### 📊 Performans Karşılaştırması

Yapılan optimizasyonlar sonucunda modelin eğitim ve doğrulama kaybında ciddi bir düşüş sağlanmıştır:

| Metrik | Önceki Durum | **Şimdiki Durum (Optimize Edilmiş)** |
| :--- | :--- | :--- |
| **Training Loss** | 0.2620 | **0.0066** |
| **Validation Loss** | 0.2648 | **0.0069** |

> **Not:** Loss değerindeki bu düşüş, modelin görüntüleri bulanık birer leke yerine, ayırt edilebilir kıyafetler olarak yeniden oluşturmasını sağlamıştır.

## 🧠 Model Mimarisi

Model, `784` (28x28) boyutundaki giriş vektörünü `64` boyutlu bir gizli uzaya (latent space) sıkıştırır ve tekrar genişletir.

**Encoder (Kodlayıcı):**
* Input (784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> **Latent Space (64, ReLU)**

**Decoder (Kod Çözücü):**
* Latent Space (64) -> Dense(128, ReLU) -> Dense(256, ReLU) -> Output (784, Sigmoid)

## 🛠️ Kurulum ve Kullanım

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### Gereksinimler
Projenin çalışması için aşağıdaki kütüphanelerin yüklü olması gerekir:
* Python 3.x
* TensorFlow
* NumPy
* Matplotlib

### Kurulum
```bash
git clone [https://github.com/HimmetDemir45/auto-encoders.git](https://github.com/HimmetDemir45/auto-encoders.git)
cd auto-encoders
pip install tensorflow numpy matplotlib

📈 Sonuçlar ve Görselleştirme
Kod çalıştırıldığında:

Veri setinden örnekler görselleştirilir.

Model eğitimi başlar (Early Stopping ile izlenir).

Test verisi üzerinde tahminler yapılır.

Orijinal görüntüler ile Yapay Zeka tarafından yeniden oluşturulan görüntüler yan yana karşılaştırılır.

PSNR (Görüntü Kalitesi) istatistikleri konsola yazdırılır.

Örnek Çıktı (PSNR Değerleri):

Plaintext

Average PSNR: 22.45 dB
Std PSNR: 2.10 dB
Min PSNR: 15.30 dB
Max PSNR: 28.90 dB

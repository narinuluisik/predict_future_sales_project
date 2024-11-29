# predict_future_sales_project

Bu proje, geçmiş satış verileri kullanarak gelecekteki satışları tahmin etmek için XGBoost algoritmasını kullanmaktadır. Proje, bir perakende satış verisi kümesinin analiz edilmesi ve satış adetlerinin tahmin edilmesi üzerine odaklanmaktadır.

## Proje İçeriği

Bu projede aşağıdaki adımlar gerçekleştirilmiştir:

1. **Veri Setinin Yüklenmesi:** Satış verisi, ürün bilgileri, mağaza bilgileri ve ürün kategorileri gibi farklı veri setleri birleştirilmiştir.
2. **Öznitelik Mühendisliği:** Verinin daha anlamlı hale gelmesi için yeni özellikler türetilmiştir. Tarih bilgisinden yıl, ay, gün ve hafta günü gibi bilgileri çıkarılmıştır.
3. **Model Kurulumu:** XGBoost algoritması kullanılarak bir regresyon modeli oluşturulmuştur. Model, geçmiş satış verilerine dayanarak gelecekteki satışları tahmin etmek için eğitilmiştir.
4. **Model Değerlendirme:** Modelin başarısı, **Root Mean Squared Error (RMSE)** metriği kullanılarak ölçülmüştür.
5. **Görselleştirme:** Modelin tahmin sonuçları ve gerçek satış verileri karşılaştırılmış ve görselleştirilmiştir.

## Kullanılan Teknolojiler

- **Python 3.x**  
- **Pandas** - Veri işleme
- **NumPy** - Matematiksel hesaplamalar
- **Scikit-learn** - Model eğitimi ve değerlendirmesi
- **XGBoost** - Makine öğrenmesi algoritması
- **Matplotlib** - Verilerin görselleştirilmesi

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız olacak:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

Gerekli kütüphaneleri yüklemek için şu komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# In[3]:


# Veri setlerini yükleyelim
sales_train = pd.read_csv('sales_train.csv')
items = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')

# Verinin ilk birkaç satırını inceleyelim
print(sales_train.head())
print(items.head())
print(item_categories.head())
print(shops.head())
print(test.head())


# In[4]:


# Verileri birleştirelim
sales_data = sales_train.merge(items, on='item_id', how='left')
sales_data = sales_data.merge(item_categories, on='item_category_id', how='left')
sales_data = sales_data.merge(shops, on='shop_id', how='left')

# 'date' sütununu datetime formatına çeviriyoruz
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')

# Yıl, ay, gün, haftanın günü gibi bilgileri türetiyoruz
sales_data['year'] = sales_data['date'].dt.year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day
sales_data['weekday'] = sales_data['date'].dt.weekday  # 0: Pazartesi, 6: Pazar

# Verinin ilk 5 satırını kontrol edelim
print(sales_data.head())


# In[5]:


# Fiyat bilgisi: 'item_price' sütununu kullanıyoruz
sales_data['price'] = sales_data['item_price']

# Fiyat bilgisinin eklendiğini kontrol edelim
print(sales_data[['item_price', 'price']].head())


# In[6]:


# Özellikler (Features) ve hedef değişkeni (Target) belirleyelim
X = sales_data[['year', 'month', 'day', 'weekday', 'price', 'item_category_id', 'shop_id']]
y = sales_data['item_cnt_day']  # Hedef değişken: Satış adedi

# Özellikler ve hedef değişkenin ilk 5 satırını kontrol edelim
print(X.head())
print(y.head())


# In[7]:


from sklearn.model_selection import train_test_split

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test setlerinin boyutlarını kontrol edelim
print(X_train.shape, X_test.shape)


# In[8]:


from xgboost import XGBRegressor

# XGBoost modelini oluşturuyoruz
model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Modeli eğitelim
model.fit(X_train, y_train)

# Eğitim sonrası modelin tahmin yapabilmesi için hazır hale gelmesini sağlıyoruz
y_pred = model.predict(X_test)

# Tahmin sonuçlarını inceleyelim
print(y_pred[:5])


# In[9]:


from sklearn.metrics import mean_squared_error
import math

# Hata hesaplaması
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

# RMSE sonucunu yazdıralım
print(f'Root Mean Squared Error (RMSE): {rmse}')


# In[10]:


import matplotlib.pyplot as plt

# Gerçek ve tahmin edilen değerleri görselleştirelim
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Gerçek', color='blue')
plt.plot(y_pred, label='Tahmin', color='red', linestyle='dashed')
plt.legend()
plt.title('Gerçek ve Tahmin Edilen Satış Adetleri')
plt.show()


# In[ ]:





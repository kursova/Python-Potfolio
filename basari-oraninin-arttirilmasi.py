# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#gerekli kütüphaneleri yükleme
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#veri setini yükleme
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#normalleştirme
train_images = train_images / 255.0

test_images = test_images / 255.0

#modeli kurma
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(8, activation='elu'), #nöron sayısını değiştirdim.
    tf.keras.layers.Dropout(0.1), #düzenlileştirme
    tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dense(32, activation='relu'), #katman ekledim


    tf.keras.layers.Dense(10,activation="softmax")
])




model.summary()

# modeli derle(compile)

model.compile(optimizer='adam', #optimizerı adagrad olarak değiştirdim.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#uygulama

hist=model.fit(train_images, train_labels,validation_data=(test_images,test_labels), epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

test_loss

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('val accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

#nöron sayısını 16 - 8 yaptığımda accuracy 0,87 test accuracy 0.86 oldu. loss: 0,34 val_loss:0,39

# 8 nöronluk yeni bir katman eklediğimde accuracy 0.8650 test accuracy 0.8534 loss:0,3763 val_loss:0.4281

#optimizerı adagrad yaptığımızda accuracy: 0.8818 - loss: 0.3372 - val_accuracy: 0.8574 - val_loss: 0.4114
#0.88'e 0.85 olması overfitting anlamına gelir mi? yoksa daha büyük bir fark mı olmalıydı?

#aktivasyon kodunu elu yaptığımızda değerlerimiz kötüleşti.accuracy: 0.7509 - loss: 0.7308 - val_accuracy: 0.7423 - val_loss: 0.7426
# leaky relu kullanınca : accuracy: 0.8072 - loss: 0.5833 - val_accuracy: 0.7955 - val_loss: 0.6080
#relu olarak değiştirdiğimizde ve dropout kullanarak düzenlileştirme yaptığımızda accuracy: 0.8443  Test accuracy: 0.844299 loss: 0.4480

#son durumda;

#nöronlardan biri 16 olarak değiştirildi. yeni katman eklendi. dropout eklendi. epoch 15 olarak değiştirildi.
#accuracy: 0.8809 - loss: 0.3292 - val_accuracy: 0.8574 - val_loss: 0.4006
#drop out 
#drop out 0.2:  accuracy: 0.8242 - loss: 0.4896 - val_accuracy: 0.8376 - val_loss: 0.457
#rop out 0.3: accuracy: 0.7916 - loss: 0.5778 - val_accuracy: 0.7974 - val_loss: 0.5574
#drop out 0.4: accuracy: 0.7584 - loss: 0.6606 - val_accuracy: 0.8134 - val_loss: 0.5313               
#drop out 0.1: accuracy: 0.8468 - loss: 0.4271 - val_accuracy: 0.8441 - val_loss: 0.4336
#drop out fonksiyonunda en iyi sonucu 0.1 değeri verdi. bunu kullanabiliriz. 
import joblib
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import joblib
from os.path import join as join
import numpy as np
from preprocess import split, preprocessing

#Modele initial
model = joblib.load(join('models', 'model_2025.pkl'))
df = pd.read_csv(join('dataset','data-all-complete_clean.csv'))

# Charger le préprocesseur
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# preprocesser les data
X, y, _ = preprocessing(df)
# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42)

# =On créé un nouveau modele pour integrer les nouvelles colonnes

model_2 = Sequential([
    Input(shape=(X_train.shape[1],), name='input'),
    Dense(64, activation='relu', name='dense_1'),
    Dense(32, activation='relu', name='dense_2'),
    Dense(1, name='output')
])

# Charger le modèle 1 et transférer les poids compatibles
model_loaded = tf.keras.models.load_model(
    join("models", "model_2025.h5"),
    custom_objects={'mse': tf.keras.losses.MeanSquaredError}
)

for layer in model_2.layers:
    print(layer.name)
    try:

        old_layer = model_loaded.get_layer(layer.name)
        layer.set_weights(old_layer.get_weights())
        print(f"✅ Poids transférés pour : {layer.name}")
    except ValueError:
        print(f"⛔ Incompatible ou nouvelle couche : {layer.name}")

model_2.compile(optimizer='adam', loss='mse')

# Entraînement du modèle avec les 6 features
history2 = model_2.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=0)

# mse = mean_squared_error(y_test, y_train)
# mae = mean_absolute_error(y_test, y_train)
# r2 = r2_score(y_test, y_train)
# print( {'MSE': mse, 'MAE': mae, 'R²': r2})

# Courbes d'apprentissage modèle 2
plt.figure(figsize=(10, 5))
plt.plot(history2.history['loss'], label='train_loss')
plt.plot(history2.history['val_loss'], label='val_loss')
plt.title("Modèle 2 - 17 features (avec transfert)")
plt.xlabel("Époques")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_model2.jpg")
plt.show()
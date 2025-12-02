import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# %% DATASET LOADING
(x_train, _), (x_test, _) = fashion_mnist.load_data()


# %% NORMALIZATION
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# %% DATA VISUALIZATION
plt.figure()
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.show()


# %% FLATTEN DATA
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# %% MODEL PARAMETERS
input_dim = x_train.shape[1]
encoding_dim = 64


# %% ENCODER ARCHITECTURE
input_image = Input(shape=(input_dim,))
encoded = Dense(256, activation="relu")(input_image)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dense(encoding_dim, activation="relu")(encoded)


# %% DECODER ARCHITECTURE
decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(256, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)


# %% COMBINE AUTOENCODER MODEL
autoencoder = Model(input_image, decoded)


# %% COMPILE MODEL
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)
# %% EARLY STOPPING
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
# %% TRAIN MODEL
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1,
)


# %% SEPARATE ENCODER AND DECODER MODELS
encoder = Model(input_image, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_output = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_output)


# %% PREDICT ON TEST DATA
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)


# %% COMPARE ORIGINAL AND RECONSTRUCTED IMAGES
n = 10

plt.figure(figsize=(15, 5))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# %% CALCULATE PSNR (PEAK SIGNAL-TO-NOISE RATIO)
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# %% CALCULATE PSNR FOR TEST DATA
psnr_scores = []
for i in range(len(x_test)):
    score = calculate_psnr(x_test[i], decoded_images[i])
    psnr_scores.append(score)

print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"Std PSNR: {np.std(psnr_scores):.2f} dB")
print(f"Min PSNR: {np.min(psnr_scores):.2f} dB")
print(f"Max PSNR: {np.max(psnr_scores):.2f} dB")


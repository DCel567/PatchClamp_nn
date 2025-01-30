import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Conv1D, AveragePooling1D, Flatten, Activation, Conv2D, BatchNormalization, Input, Concatenate, Dense, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#file_path = 'data_extracted/my_shifted_train_data.csv'
file_path = 'data_extracted/1.csv'
data = pd.read_csv(file_path, header=None)

print(data[:2])

X = data[1].values.reshape(-1, 1)
y = data[2].astype(int)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Normalize the signal values
scaler = MinMaxScaler(feature_range = (0, 1))
#scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_scaled = X

print(X_scaled.shape)

# Reshape data for Conv1D
# Assuming you want to use a window size of 3 for neighbors
window_size = 512
X_scaled_reshaped = as_strided(X_scaled, shape=(X_scaled.shape[0] - window_size + 1, window_size, 1), strides=(X_scaled.strides[0], X_scaled.strides[0], X_scaled.strides[1]))
y_updated = y[:-window_size+1]
y_encoded_updated = encoder.transform(y_updated.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled_reshaped, y_encoded_updated, test_size=0.2, shuffle=False)
print(X_test.shape, y_test.shape)
###
model = Sequential()
#model.add(BatchNormalization(input_shape=(window_size, 1)))
# model.add(Conv1D(64, kernel_size=64, activation='relu', input_shape=(window_size, 1)))  # Convolutional layer
model.add(Conv1D(64, kernel_size=128, activation='relu'))  # Convolutional layer
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=32))  # Pooling layer
model.add(Conv1D(32, kernel_size=4, activation='relu'))  # Second convolutional layer
model.add(MaxPooling1D(pool_size=4))  # Second pooling layer
model.add(Flatten())  # Flatten the output
model.add(Dense(64, activation='relu'))  # Dense layer
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','recall'])
###

# input_shape = (22, 500, 1)
# input_shape = (window_size, 1)
# inputs = Input(shape=input_shape)
# x = BatchNormalization()(inputs)
# a = Conv1D(50, kernel_size=128, strides=6, padding="same")(inputs)
# b = Conv1D(50, kernel_size=64, strides=6, padding="same")(a)
# x = Conv1D(50, kernel_size=32, strides=6, padding="same")(b)
# #x = Concatenate()([a, b, c])
# x = BatchNormalization()(x)
# x = Conv1D(20, kernel_size=22, strides=1, padding="same")(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = AveragePooling1D(pool_size=5, strides=4)(x)
# x = Flatten()(x)
# x = Dense(256, activation="relu")(x)
# x = Dense(3, activation='softmax')(x)
# outputs = x

# model = Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


class_weights = {0: 1, 1: 1, 2: 2}
model.fit(X_train, y_train, epochs=1, batch_size=100, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print(f"predicted: {predicted_classes[:2]}")


plt.figure(figsize=(15, 10))

# Original Signal
plt.subplot(3, 1, 1)
#plt.plot(X_scaled[:, 0, :], label='Original Signal', color='blue')
plt.plot(X_test, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.xlabel('Index')
plt.ylabel('Signal Value')
plt.legend()

# Original Arbitrary Number
plt.subplot(3, 1, 2) #data.index
plt.plot(y_test, label='Original Arbitrary Number', color='orange')
plt.title('Original Arbitrary Number')
plt.xlabel('Index')
plt.ylabel('Arbitrary Number')
plt.legend()

# Predicted Arbitrary Number
plt.subplot(3, 1, 3) # data.index[-len(y_test):]
plt.plot(predicted_classes, label='Predicted Arbitrary Number', color='green')
plt.title('Predicted Arbitrary Number')
plt.xlabel('Index')
plt.ylabel('Predicted Number')
plt.legend()

plt.tight_layout()
plt.show()

# Optionally, save the model
#model.save('my_model.h5')
X_test
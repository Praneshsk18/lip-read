import os
from keras.models import Sequential
from keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, TimeDistributed, Flatten
)

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # ✅ Correcting the model weights path
    model_weights_path = os.path.join(os.getcwd(), 'saved_models', 'checkpoint')
    
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    return model

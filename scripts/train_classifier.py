"""Train the starboard/port classification model.

This script mirrors the main step from Training Classification.ipynb.
It expects data/input_data.npy and data/output_data.npy by default.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def create_classifier(input_dim=6):
    x_in = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(x_in)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    y = Dense(1, activation="sigmoid")(x)
    model = Model(x_in, y)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(data_dir="data", model_dir="models", epochs=30, batch_size=256):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    x_path = data_dir / "input_data.npy"
    y_path = data_dir / "output_data.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Place input_data.npy and output_data.npy under data/ first.")

    x = np.load(x_path).reshape(-1, 6)
    y = np.load(y_path).reshape(-1, 2)[:, 0]
    y = (y > 0).astype("float32")

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = create_classifier()
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    print(model.evaluate(x_test, y_test, batch_size=batch_size, return_dict=True))

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "Classification.h5")


if __name__ == "__main__":
    main()

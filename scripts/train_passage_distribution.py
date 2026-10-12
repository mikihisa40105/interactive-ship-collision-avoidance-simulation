"""Train log-normal passage-position models for starboard and port sides."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from ship_collision_model.losses import LogNormalNLLLoss


def create_lognormal_model(input_dim=6):
    x_in = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(x_in)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    y = Dense(2)(x)
    model = Model(x_in, y)
    model.compile(optimizer="adam", loss=LogNormalNLLLoss())
    return model


def train_one(x, y, output_path, epochs=50, batch_size=256):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = create_lognormal_model()
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    print(output_path.name, model.evaluate(x_test, y_test, batch_size=batch_size, return_dict=True))
    model.save(output_path)


def main(data_dir="data", model_dir="models", epochs=50, batch_size=256):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    x_path = data_dir / "input_data.npy"
    y_path = data_dir / "output_data.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Place input_data.npy and output_data.npy under data/ first.")

    x = np.load(x_path).reshape(-1, 6)
    y = np.load(y_path).reshape(-1, 2)[:, 0]

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    stb = y > 0
    prt = y < 0
    model_dir.mkdir(parents=True, exist_ok=True)

    train_one(x[stb], y[stb].reshape(-1, 1), model_dir / "LogNormStb.h5", epochs, batch_size)
    train_one(x[prt], np.abs(y[prt]).reshape(-1, 1), model_dir / "LogNormPrt.h5", epochs, batch_size)


if __name__ == "__main__":
    main()

import numpy as np
from numpy.random import uniform
from numpy.random import normal
from tensorflow import keras
from tensorflow.keras.optimizers import AdamW

PARAM_LO = 0
PARAM_HI = 100
DATA_LO = 0
DATA_HI = 100
DATA_MEAN = 0
DATA_STD = 10
DATA_N = 1000
COL_X = 0
COL_Y = 1
COL_Z = 2
N_EPOCH = 500


def main() -> None:
    model = SimpleModel()
    model.go()


class SimpleModel:


    def __init__(self) -> None:
        self.slope_x = 15 # uniform(PARAM_LO, PARAM_HI)
        self.slope_y = 2 # uniform(PARAM_LO, PARAM_HI)
        self.offset_x = 4 # uniform(PARAM_LO, PARAM_HI)
        self.offset_y = 0 # uniform(PARAM_LO, PARAM_HI)
        print(f"SlopeX = {self.slope_x:.1f}")
        print(f"SlopeY = {self.slope_y:.1f}")
        print(f"OffsetX = {self.offset_x:.1f}")
        print(f"OffsetY = {self.offset_y:.1f}")
        self.data = self.create_data()
        self.model = self.create_model()


    def create_model(self) -> keras.models.Sequential:
        layers = keras.layers
        model = keras.models.Sequential()
        # model.add(layers.Dense(4, activation="relu", input_dim=2))
        # model.add(layers.Dense(1, activation=None))
        model.add(layers.Dense(1, activation=None, input_dim=2))
        model.compile(
            loss="MSE",
            optimizer=AdamW(learning_rate=0.001),
            metrics=["accuracy"],
        )
        return model


    def create_data(self) -> np.ndarray:
        xs = uniform(DATA_LO, DATA_HI, DATA_N)
        ys = uniform(DATA_LO, DATA_HI, DATA_N)
        zs = self.slope_x * xs + \
            self.slope_y * ys + \
            self.offset_x + \
            self.offset_y
        zs = zs + normal(DATA_MEAN, DATA_STD, DATA_N)
        return np.transpose(np.stack([xs, ys, zs]))


    def sigmoid(self, arr) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-arr))


    def go(self) -> None:
        xdata = self.data[:, [COL_X, COL_Y]]
        ydata = self.data[:, [COL_Z]]
        print(f"Training ...")
        self.model.fit(
            xdata,
            ydata,
            batch_size = 16,
            epochs=N_EPOCH,
            verbose=2,
            
        )
        print(self.model)
        print(self.model.summary())
        for layer in self.model.layers:
            print(layer.get_weights())

        
if __name__ == "__main__":
    main()

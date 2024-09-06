# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sign(value):
    return 1 if value > 0 else -1


def read_csv_convert_to_numpy(fileName="carSUV_normalized.csv"):
    df = pd.read_csv(fileName)

    numpy_x = df[["ZeroToSixty", "PowerHP"]].to_numpy()
    numpy_y = df["IsCar"].replace(0, -1).to_numpy().reshape(-1, 1)

    return numpy_x, numpy_y


numpy_x, numpy_y = read_csv_convert_to_numpy()


def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    error_count = 0
    for index in range(len(numpy_x)):
        predict = sign(w[0] * numpy_x[index][0] + w[1] * numpy_x[index][1])
        if predict != numpy_y[index]:
            error_count += 1

    return error_count / len(numpy_y)


np.random.seed(3)  # to fix randomness
random_w = np.random.randn(2, 1)
print("Random weights array shape", random_w.shape)
print("Random weights values\n", random_w)
print("error:", calc_error_rate_for_single_vector_w(random_w, numpy_x, numpy_y))


def train_and_evaluate(numpy_x, numpy_y, n_epochs=20, c=0.01):
    w = np.random.randn(2, 1)
    for _ in range(n_epochs):
        for index in range(len(numpy_x)):
            predict = sign(w[0] * numpy_x[index][0] + w[1] * numpy_x[index][1])
            if predict == numpy_y[index]:
                continue
            else:
                for w_index in range(len(w)):
                    w[w_index][0] = (
                        w[w_index][0]
                        + c * (numpy_y[index][0] - predict) * numpy_x[index][w_index]
                    )

        calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)
    return w


np.random.seed(8)
trained_w = train_and_evaluate(numpy_x, numpy_y)
print(trained_w)
print("error:", calc_error_rate_for_single_vector_w(trained_w, numpy_x, numpy_y))


def plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w):
    x_vals = np.arange(-2, 2, .01)
    y_vals = -(trained_w[0][0] * x_vals) / trained_w[1][0]
    plt.scatter(numpy_x[:, 0], numpy_x[:, 1], c=numpy_y)
    plt.plot(x_vals, y_vals)


plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w)


# %%

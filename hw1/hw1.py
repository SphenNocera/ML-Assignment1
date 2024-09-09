import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_convert_to_numpy(fileName="carSUV_normalized.csv"):
    df = pd.read_csv(fileName)

    numpy_x = df[["ZeroToSixty", "PowerHP"]].to_numpy()
    numpy_y = df["IsCar"].replace(0, -1).to_numpy().reshape(-1, 1)

    return numpy_x, numpy_y


def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    predict = np.sign(np.matmul(numpy_x, w))
    error_count = np.sum(predict != numpy_y)

    return error_count / len(numpy_y)


def train_and_evaluate(numpy_x, numpy_y, n_epochs=20, c=0.01):
    w = np.random.randn(2, 1)
    for _ in range(n_epochs):
        predict = np.sign(np.matmul(numpy_x, w))
        incorrect = predict != numpy_y
        for i in np.where(incorrect)[0]:
            w += c * (numpy_y[i][0] - predict[i][0]) * numpy_x[i].reshape(2, 1)

        calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)
    return w


def plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w):
    x_vals = np.arange(-2, 2, .01)
    y_vals = -(trained_w[0][0] * x_vals) / trained_w[1][0]
    plt.scatter(numpy_x[:, 0], numpy_x[:, 1], c=numpy_y)
    plt.plot(x_vals, y_vals)
    plt.show()

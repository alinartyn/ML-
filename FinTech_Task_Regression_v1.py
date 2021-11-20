#!/usr/bin/env python

'''
    Task: Show simple regression model using functions

    FINS3648 market prices using Yahoo
        1. Setup Functions
        2. Prepare data
        3. Run as Script
    License: ""
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coefficients(x, y):
    # basic calculations
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)

    # show deviations about x
    SS_xy = np.sum(y * x - n * m_y * m_x)
    SS_xx = np.sum(x * x - n * m_x * m_x)

    # calculating reg coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    return (b_0, b_1)

def plotter(x, y, b):
    # use scatter for closer representations
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # use our model to predict projected values
    y_pred = b[0] + b[1] * x

    # plotting model line as derived by this simple model
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    # this function is used to encapsulate script method/run
    # define x and y variables

    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = 2 * x - 5 + rng.randn(50)

    # read preloaded time series in csv
    #stock1 = pd.read_csv("c:/tmp/V.csv")
    # define y and x time series where x is defined as shifted y(t-1) e.g. yesterday close price
    #y = stock1.Close
    #x = y.shift(-1)
    #print(x)

    # estimating coefficients
    b = coefficients(x, y)
    print("Estimates :\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plotter(x, y, b)


if __name__ == "__main__":
    main()
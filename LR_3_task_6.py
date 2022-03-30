import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

lin_reg = linear_model.LinearRegression()
plot_learning_curves(lin_reg, X, y)

poly = PolynomialFeatures(degree=10, include_bias=False)

polynomial_regression = Pipeline([
    ("poly_features", poly),
    ("lin_reg", lin_reg),
])
plot_learning_curves(polynomial_regression, X, y)

poly = PolynomialFeatures(degree=2, include_bias=False)

polynomial_regression = Pipeline([
    ("poly_features", poly),
    ("lin_reg", lin_reg),
])
plot_learning_curves(polynomial_regression, X, y)

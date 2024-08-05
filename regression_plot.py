from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lr

x, y = make_regression(n_samples=200, n_features=1, noise=30)

model = lr()

model.fit(x,y)

plt.scatter(x,y)

plt.plot(x, model.predict(x), color='red', linewidth=3)

plt.show()

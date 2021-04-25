# %% init
import numpy as np
import pandas as pd

from regression import RegressionTree

np.random.seed(123)

# %% test 0
print('== test 0 ==')
model = RegressionTree(depth=1)
X = [
    [1, 0],
    [3, 0],
    [3, 0],
]
y = [-2, -1, 5]
print('Training the regression decision tree...')
model.fit(X, y)
print('Finished training.')
print('Evaluation loss (MSE):', model.evaluate(X, y))  # 6.0
print('Model:', model.root)  # feat=0, t=3, l=(v=-2.0), r=(v=2.0)

# %% test 1
print('== test 1 ==')
model = RegressionTree(depth=8)
X = np.random.random_sample((1000, 10))
y = np.random.random_sample(1000) * 5
print('Training the regression decision tree...')
model.fit(X, y)
print('Finished training.')
print('Evaluation loss (MSE):', model.evaluate(X, y))  # 1.190281864435004

# %% test 2
print('== test 2 ==')
model = RegressionTree(depth=10)
X = np.random.random_sample((10_000, 10))
y = np.random.random_sample(10_000) * 5
print('Training the regression decision tree...')
model.fit(X, y)
print('Finished training.')
print('Evaluation loss (MSE):', model.evaluate(X, y))  # 1.9560924160407038

# %% test 3
print('== test 3 ==')
model = RegressionTree(depth=9)
X = np.random.random_sample((100_000, 10))
y = np.random.random_sample(100_000) * 5
print('Training the regression decision tree...')
model.fit(X, y)
print('Finished training.')
print('Evaluation loss (MSE):', model.evaluate(X, y))  # 2.073520664825586

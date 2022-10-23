from collections.abc import Callable

import numpy as np

# Sample data for the tests
# The example data is taken from https://realpython.com/linear-regression-in-python
# /#simple-linear-regression-with-scikit-learn
x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])


def _example_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    This gradient uses a modification of ssr to work with only one point
    >>> _example_gradient(np.array([5]), np.array([5]), np.array([0, 0]))
    array([ -5., -25.])
    >>> _example_gradient(np.array([5]), np.array([5]), np.array([5, 5]))
    array([ 25., 125.])
    >>> _example_gradient(np.array([15]), np.array([20]), np.array([5, 5]))
    array([ 60., 900.])
    >>> _example_gradient(np.array([25]),np.array([14]), np.array([5,5]))
    array([ 116., 2900.])
    >>> _example_gradient(np.array([35]), np.array([32]), np.array([5, 5]))
    array([ 148., 5180.])
    """
    intercept = w[0] + w[1] * x - y
    slope = intercept * x
    return np.array([intercept.mean(), slope.mean()])


def stochastic_gradient_descent(
    gradient: Callable,
    x: np.ndarray,
    y: np.array,
    start: np.ndarray = np.array([0, 0], dtype=np.float64),
    max_iterations: int = 50,
    learning_rate: float = 0.1,
    tolerance: float = 1e-6,
    seed: int = None,
):
    """
    >>> stochastic_gradient_descent(_example_gradient, x, y,max_iterations=100, learning_rate=0.1, seed=123)
    array([5.633333333333329, 0.54], dtype=np.float64)
    """
    rng = np.random.default_rng(seed=seed)

    weights = start.copy()

    data = np.c_[x.reshape(len(x), -1), y.reshape(len(y), 1)]

    for _ in range(max_iterations):
        rng.shuffle(data)

        for start in range(len(data)):
            stop = start + 1
            observation, expected_value = data[start:stop, :-1], data[start:stop, -1:]

            gradient_result = gradient(observation, expected_value, weights)
            adjustment = -learning_rate * gradient_result

            if np.all(np.abs(adjustment) <= tolerance):
                break

            weights += adjustment

    return weights


def sgd(
    gradient,
    x,
    y,
    start,
    learn_rate=0.1,
    batch_size=1,
    n_iter=50,
    tolerance=1e-06,
    dtype="float64",
    random_state=None,
):
    # Checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    # Setting up the data type for NumPy arrays
    dtype_ = np.dtype(dtype)

    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # Initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the size of minibatches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError(
            "'batch_size' must be greater than zero and less than "
            "or equal to the number of observations"
        )

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Shuffle x and y
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # Recalculating the difference
            grad = gradient(x_batch, y_batch, vector)
            diff = -learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff

    return vector if vector.shape else vector.item()


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    #
    # result = stochastic_gradient_descent(gradient_mean_squares, x_observations, y_true_labels, max_iterations=500,learning_rate=0.1, seed=123)
    # print(result)
    #

    result = sgd(_example_gradient, x, y, [0, 0], n_iter=100, random_state=123)
    print(result)

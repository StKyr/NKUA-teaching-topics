import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(gradient_func, x0, step_size,
                     tol=10e-6, num_iters=1000):
    converged = False
    x_history = [x0]
    x         = x0
    t         = 0
    while not converged and t < num_iters:
        x_next    = x - step_size * gradient_func(x)
        converged = np.linalg.norm(x-x_next) <= tol
        x         = x_next

        x_history.append(x)
        t += 1
    return x, x_history

# ------------------------------------------------

func = lambda x: 5 * x ** 2 + 1 # minimum: f(0) = 1
grad = lambda x: 2 * 5 * x

x_best, x_history = gradient_descent(grad, 10, 0.02)

print("Gradient descent converged to x = {} with f(x) = {}."
    .format(x_best, func(x_best)))

plot_range = np.linspace(-10, 10)
func_values = [func(x) for x in x_history]
plt.plot(plot_range, [func(x) for x in plot_range])
plt.scatter(x_history, func_values, marker='+', c='r')
plt.show()
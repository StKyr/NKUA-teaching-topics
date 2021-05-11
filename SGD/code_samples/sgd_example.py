import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def gradient_descent(gradient_func, x0, lr, tol=10e-6, num_iters=1000):

    converged = False
    x_history = [x0]
    x         = x0
    t         = 0
    while not converged and t < num_iters:

        x_next    = x - lr * gradient_func(x)
        converged = np.linalg.norm(x-x_next) <= tol
        x         = x_next

        x_history.append(x)
        t += 1
    return x, np.array(x_history)







def run_gd_1D():
    func = lambda x: 5 * x ** 2 + 1
    grad = lambda x: 5 * 2 * x

    x_best, x_history = gradient_descent(grad, 10, 0.02)

    print("Gradient descent converged to x = {} with f(x) = {}.".format(x_best, func(x_best)))

    plot_range = np.linspace(-10, 10)
    func_values = func(x_history)
    plt.plot(plot_range, func(plot_range))
    plt.scatter(x_history, func_values, marker='+', c='r')
    plt.show()




def run_gd_2D():
    func = lambda X, Y: (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2

    def grad(vector):
        x = vector[0]
        y = vector[1]

        grad_x = 4 * x ** 3 + (4 * y + 2) * x + 2 * y ** 2 - 58
        grad_y = 4 * y ** 3 + (4 * x + 2) * y + 2 * x ** 2 - 58

        return np.array([grad_x, grad_y])

    x0 = np.array([5, -5])
    x_best, x_history = gradient_descent(grad, x0, lr=10e-3, num_iters=10000)
    func_values = [func(x_history[i][0], x_history[i][1]) for i in range(x_history.shape[0])]


    print("Gradient descent converged to x = {} with f(x) = {}.".format(x_best, func(x_best[0], x_best[1])))

    # print(x_history)
    # print(func_values)



    z = np.linspace(-10, 10)
    xx, yy = np.meshgrid(z, z)
    zz = func(xx,yy)

    import matplotlib as mpl
    mpl.use('Qt5Agg')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.5)


    ax.scatter(x_history[:,0], x_history[:,1], np.array([func_values])+0.05, marker='+', c='r', s=50)



    plt.show()


if __name__ == '__main__':


    #run_gd_1D()

    run_gd_2D()


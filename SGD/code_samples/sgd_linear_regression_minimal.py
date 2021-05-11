import numpy as np

x             = np.linspace(0,1,100)
y             = 5 * x + 2 + 0.25*np.random.randn(100)
learning_rate = 0.004
w             = np.random.uniform(0, np.pi)
b             = np.random.uniform(y.min(), y.max())

for i in range(50):
    predictions = w*x + b
    loss        = sum((y-predictions)**2)
    grad_w      = sum(-2*x*(y-predictions))
    grad_b      = sum(-2*(y-predictions))
    w           = w - learning_rate * grad_w
    b           = b - learning_rate * grad_b
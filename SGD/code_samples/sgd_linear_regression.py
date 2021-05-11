'''
Example of using Stochastic Gradient Descent to fit a linear model
on data generated from a known line plus Gaussian noise.

The code requires matplotlib and numpy libraries to be installed.
Press any key on each figure window while focused to resume execution.


---------------------------------------------------------------
Used for teaching in Prof. George Alexandropoulos' course in 
Department of Informatics and Telecommunications, NKUA.

Contact: Kyriakos Stylianopoulos - kstylianop@di.uoa.gr
'''

import numpy as np, matplotlib.pyplot as plt


# Generate the dataset: Use the line y=5x+2 plus some noise.
x      = np.linspace(0,1,100)
y_real = 5 * x + 2
y      = y_real + 0.25*np.random.randn(100)

# The loss function is the Sume of Squared Errors (SSE)
print('Theoretically optimum loss: {:.2f}'.format(sum( (y-y_real)**2)))


# Show the true model and the residual errors
plt.plot(x, y_real, c='green', linewidth=1)
plt.scatter(x, y, s=5)
for i,xi in enumerate(x):
    plt.vlines(xi, min(y[i], y_real[i]), max(y[i], y_real[i]), color='red', linewidth=1)
plt.title("Data gerenrated from linear model with i.i.d. noise")
plt.draw()  
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close()    




# Configure hyperparameters (tuned for this problem)
learning_rate = 0.005
iterations    = 50

# Model parameters (trainable): w: slope, b: intercept
# Initialize them in random values (w in [0,90] degrees and b inside the range of the y values) 
w  = np.random.uniform(1)*np.pi
b  = np.random.uniform(1) + (y.max() - y.min())/2



# Keep track of loss function per iterations for plotting
losses = []

print("Iter |   Loss  |   w   |  b  ")
print("-----|---------|-------|-----")


for i in range(iterations):
    predictions = w*x + b                     # Apply the model to get current predicted y values
    loss        = sum((y-predictions)**2)     # Compute the value of the SSE loss function
    grad_w      = sum(-2*x*(y-predictions))   # Compute the stochastic gradient: θL/θw
    grad_b      = sum(-2*(y-predictions))     # Compute the stochastic gradient: θL/θb
    w           = w - learning_rate * grad_w  # Apply the gradient update step for w 
    b           = b - learning_rate * grad_b  # Apply the gradient update step for b

    losses.append(loss)
    print("{:4d} | {:>7.2f} | {:.3f} | {:.3f}".format(i,loss,w,b))

    # plot current line (press any key to proceed to the next iteration)
    plt.scatter(x,y, s=5)
    plt.plot(x, predictions, c='green')
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close()    


# Plot all losses together to see how the learning process evolved
plt.figure()
plt.plot(range(1,len(losses)+1), losses)
plt.hlines(sum((y-y_real)**2), 0, len(losses), label='Minimum (theoretical)', linestyles=':', color='k')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Sum squared error')
plt.title('Loss function minimization through SGD')
plt.show()
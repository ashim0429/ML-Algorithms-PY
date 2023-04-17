import numpy as np


def gradientDescent(x, y):
    # initial values of m and b
    m_curr = b_curr = 0
    # number of iterations
    iterations = 1000
    # number of elements in x
    n = len(x)
    # learning rate
    learning_rate = 0.01
    for i in range(iterations):
        # y = mx + b (slope formula)
        y_predicted = m_curr * x + b_curr
        # cost function = 1/n * sum((y-y_predicted)**2)
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        # partial derivative of m
        md = -(2/n) * sum(x*(y-y_predicted))
        # partial derivative of b
        bd = -(2/n) * sum(y-y_predicted)
        # updated m and b with respect to learning rate and partial derivatives
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradientDescent(x, y)

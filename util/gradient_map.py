import matplotlib.pyplot as plt
import numpy as np
import math

def func_to_vectorize(x, y, dx, dy, scaling=0.01):
    plt.arrow(x, y, dx*scaling, dy*scaling, fc="k", ec="k", head_width=0.3, head_length=0.2)

def plot_gradient_map(matrix,gradient_x,gradient_y):
    ###y axis direction in this function is opposite to that of the matrix
    matrix = matrix
    gradient_x = np.flip(gradient_x, 0)
    gradient_y = -np.flip(gradient_y, 0)
    horizontal_min, horizontal_max, horizontal_stepsize = 0, matrix.shape[1], 1
    vertical_min, vertical_max, vertical_stepsize = 0, matrix.shape[0], 1

    horizontal_dist = horizontal_max-horizontal_min
    vertical_dist = vertical_max-vertical_min

    horizontal_stepsize = horizontal_dist / float(math.ceil(horizontal_dist/float(horizontal_stepsize)))
    vertical_stepsize = vertical_dist / float(math.ceil(vertical_dist/float(vertical_stepsize)))

    xv, yv = np.meshgrid(np.arange(horizontal_min, horizontal_max, horizontal_stepsize),
                         np.arange(vertical_min, vertical_max, vertical_stepsize))
    xv+=horizontal_stepsize/2.0
    yv+=vertical_stepsize/2.0

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)
    scale_x = 1/np.max(abs(gradient_x))
    scale_y = 1/np.max(abs(gradient_y))
    scale = max([scale_x,scale_y])
    plt.imshow(matrix, extent=[horizontal_min, horizontal_max, vertical_min, vertical_max])
    vectorized_arrow_drawing(xv, yv, gradient_x, gradient_y, scale)
    plt.colorbar()
    plt.show()

def plot_gradient_map_np(matrix):
    horizontal_min, horizontal_max, horizontal_stepsize = 0, matrix[0], 1
    vertical_min, vertical_max, vertical_stepsize = 0, matrix[1], 1

    horizontal_dist = horizontal_max-horizontal_min
    vertical_dist = vertical_max-vertical_min

    horizontal_stepsize = horizontal_dist / float(math.ceil(horizontal_dist/float(horizontal_stepsize)))
    vertical_stepsize = vertical_dist / float(math.ceil(vertical_dist/float(vertical_stepsize)))

    xv, yv = np.meshgrid(np.arange(horizontal_min, horizontal_max, horizontal_stepsize),
                         np.arange(vertical_min, vertical_max, vertical_stepsize))
    xv+=horizontal_stepsize/2.0
    yv+=vertical_stepsize/2.0

    yd, xd = np.gradient(matrix)
    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

    plt.imshow(np.flip(matrix,0), extent=[horizontal_min, horizontal_max, vertical_min, vertical_max])
    vectorized_arrow_drawing(xv, yv, xd, yd, 0.1)
    plt.colorbar()
    plt.show()
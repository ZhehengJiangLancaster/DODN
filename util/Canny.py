import cv2
import os
import numpy as np

input_path = (r'__your_input_path')
output_path = (r'__your_output_path')


def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (5,5), 0)
    return image

def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
           for i in range(shape[1])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k

def SobelFilter(image):
    image = Grayscale(GaussianBlur(image))
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    # kernel_x = np.array(([-3, 0, 3], [-10, 0, 10], [-3, 0, 3]))
    # kernel_y = np.array(([-3, -10, -3], [0, 0, 0], [3, 10, 3]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_y))

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles

def our_SobelFilter(image):
    image = Grayscale(image)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    G_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    G_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    # kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    # for i in range(1, size[0] - 1):
    #     for j in range(1, size[1] - 1):
    #         G_x[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_x))
    #         G_y[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_y))

    # ###plot gradient map(x flip image )
    # x_range = np.array([290,310])
    # y_range = np.array([240,260])
    # plot_gradient_map(image[x_range[0]:x_range[1],y_range[0]:y_range[1]],
    #                   G_x[x_range[0]:x_range[1],y_range[0]:y_range[1]],
    #                   G_y[x_range[0]:x_range[1],y_range[0]:y_range[1]])

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())
    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles

def our_SobelFilter_train(image):
    image = Grayscale(image)
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_y))

    # ###plot gradient map(x flip image )
    # x_range = np.array([290,310])
    # y_range = np.array([240,260])
    # plot_gradient_map(image[x_range[0]:x_range[1],y_range[0]:y_range[1]],
    #                   G_x[x_range[0]:x_range[1],y_range[0]:y_range[1]],
    #                   G_y[x_range[0]:x_range[1],y_range[0]:y_range[1]])

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())
    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles


def custom_SobelFilter(image,kernel_size=(3,3)):
    image = Grayscale(GaussianBlur(image))
    # image = Grayscale(image)
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = custom_sobel(kernel_size, 0)
    kernel_y =  custom_sobel(kernel_size, 1)
    shift = int(kernel_size[0]/2)
    # kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    # kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(shift, size[0] - shift):
        for j in range(shift, size[1] - shift):
            G_x[i, j] = np.sum(np.multiply(image[i - shift: i + shift+1, j - shift: j + shift+1], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - shift: i + shift+1, j - shift: j + shift+1], kernel_y))

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles

def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if ((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y] == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def Canny(image, low, high):
    i, angles = our_SobelFilter(image)
    image = Grayscale(image)
    gray_image = image
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    # image = image + gray_image
    # image = double_threshold_hysteresis(image, low, high)
    return image, gradient


if __name__ == "__main__":
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            image = cv2.imread(os.path.join(root, filename))
            image, gradient = Canny(image, 0, 50)
            cv2.imwrite(os.path.join(output_path, filename), image)
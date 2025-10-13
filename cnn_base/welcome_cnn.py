import numpy as np

image = np.array([
    [1, 2, 0, 1, 3],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 0],
    [2, 1, 0, 1, 2],
    [3, 2, 1, 0, 1]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

output = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        region = image[i:i+3, j:j+3]
        output[i, j] = np.sum(region * kernel)

print("Output after applying the convolutional kernel:")
print(output)

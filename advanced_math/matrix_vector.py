import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

dot_np = np.dot(v1, v2)
print("dot_np: ", dot_np)

def dot_manual(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

print("dot_manual: ", dot_manual(v1, v2))

matmul_np = np.dot(A, B)
print("matmul_np: ", matmul_np)


def matmul_manual(X, Y):
    m, n = X.shape
    n2, p = Y.shape
    assert n == n2, "Number of columns in X must match number of rows in Y"
    result = [[sum(X[i, k] * Y[k, j] for k in range(n)) for j in range(p)] for i in range(m)]
    return np.array(result)

print("matmul_manual: ", matmul_manual(A, B))








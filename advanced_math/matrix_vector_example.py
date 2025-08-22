import numpy as np

'''
example 1: 
'''

v1 = np.random.randint(0, 10, size=(4, 1)).flatten()
v2 = np.random.randint(0, 10, size=(4, 1)).flatten()
print("v1: ", v1)
print("v2: ", v2)

def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

print("dot_product: ", dot_product(v1, v2))

dot_np = np.dot(v1, v2)
print("dot_np: ", dot_np)

'''
example 2: 

'''

A = np.random.randint(0, 10, size=(3, 3))
B = np.random.randint(0, 10, size=(3,1))

def matmul_manual(X, Y):
    m, n = X.shape
    n2, p = Y.shape
    assert n == n2, "Number of columns in X must match number of rows in Y"
    result = [[sum(X[i, k] * Y[k, j] for k in range(n)) for j in range(p)] for i in range(m)]
    return np.array(result)

matrix_manual = matmul_manual(A, B)
print("matrix_manual: ", matrix_manual)


'''
example 3: 
'''

ar1 = np.random.randint(0, 10, size=(3, 1)).flatten()
ar2 = np.random.randint(0, 10, size=(3, 1)).flatten()

def outter_product(v1, v2):
    result = [[a * b for b in v2] for a in v1]
    return np.array(result)

outer_manual = outter_product(ar1, ar2)
print("outer_manual: ", outer_manual)

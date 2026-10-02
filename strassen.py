import numpy as np
import time
import sys

def read_input_file(filename, d):
    with open(filename, 'r') as f:
        values = list(map(int, f.read().split()))
    A = np.array(values[:d*d]).reshape(d, d)
    B = np.array(values[d*d:]).reshape(d, d)
    return A, B


def standard_matmul(X, Y):

    n = X.shape[0]
    Z = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            for j in range(n):
                Z[i, j] += X[i, k] * Y[k, j]
    return Z

def strassen_matmul(X, Y, n0 = None):
    n = X.shape[0]
    
    if n0 == None:
        n0 = 10 if n % 2 == 0 else 37
    
    if n <= n0:
        return standard_matmul(X, Y)

    if n % 2 != 0:
        X = np.pad(X, ((0, 1), (0, 1)), mode='constant')
        Y = np.pad(Y, ((0, 1), (0, 1)), mode='constant')
        n += 1
    
    mid = n//2 
    A, B, C, D = X[:mid, :mid], X[:mid, mid:], X[mid:, :mid], X[mid:, mid:]
    E, F, G, H = Y[:mid, :mid], Y[:mid, mid:], Y[mid:, :mid], Y[mid:, mid:]

    Z11 = np.zeros((mid, mid))
    Z12 = np.zeros((mid, mid))
    Z21 = np.zeros((mid, mid))
    Z22 = np.zeros((mid, mid))

    P = strassen_matmul(A, F - H)
    P = P[:mid, :mid]
    Z12 += P
    Z22 += P

    P = strassen_matmul(A + B, H)
    P = P[:mid, :mid]
    Z12 += P
    Z11 -= P

    P = strassen_matmul(C + D, E)
    P = P[:mid, :mid]
    Z21 = P
    Z22 -= P

    P = strassen_matmul(D, G - E)
    P = P[:mid, :mid]
    Z11 += P
    Z21 += P

    P = strassen_matmul(A + D, E + H)
    P = P[:mid, :mid]
    Z11 += P
    Z22 += P

    P = strassen_matmul(B - D, G + H)
    P = P[:mid, :mid]
    Z11 += P

    P = strassen_matmul(C - A, E + F)
    P = P[:mid, :mid]
    Z22 += P

    
    Z = np.vstack((np.hstack((Z11, Z12)), np.hstack((Z21, Z22))))
    return Z

def main():
    flag = int(sys.argv[1])
    d = int(sys.argv[2])
    inputfile = sys.argv[3]

    X, Y = read_input_file(inputfile, d)

    Z = strassen_matmul(X, Y)

    for i in range(d):
        print(int(Z[i, i]))


if __name__ == "__main__":
    main()

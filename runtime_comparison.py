import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import comb

#Try different orders of loops
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


sizes = [2**i for i in range(1, 9)]
standard_times = []
strassen_times = []
theoretical_n3 = []
theoretical_strassen = []

for n in sizes:
    X = np.random.rand(n, n)
    Y = np.random.rand(n, n)

    # Time standard matmul
    start = time.time()
    standard_matmul(X, Y)
    end = time.time()
    standard_times.append(end - start)

    # Time strassen matmul
    start = time.time()
    strassen_matmul(X, Y)  # You can tune n0
    end = time.time()
    strassen_times.append(end - start)

    # Theoretical growth
    theoretical_n3.append(n**3)
    theoretical_strassen.append(n**2.807)

# Normalize theoretical curves for better visual comparison
n3_norm = [t / theoretical_n3[-1] * standard_times[-1] for t in theoretical_n3]
strassen_norm = [t / theoretical_strassen[-1] * strassen_times[-1] for t in theoretical_strassen]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, standard_times, 'o-', label='Standard (Measured)', color='blue')
plt.plot(sizes, strassen_times, 'o-', label='Strassen (Measured)', color='red')
plt.plot(sizes, n3_norm, '--', label='n³ (Theoretical)', color='green')
plt.plot(sizes, strassen_norm, '--', label='n^2.807 (Theoretical)', color='purple')

plt.xlabel("Matrix size (n x n)")
plt.xticks([2**i for i in range(1, 9)], ['$2^{}$'.format(i) for i in range(1, 9)])
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison: Standard vs Strassen Matrix Multiplication")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


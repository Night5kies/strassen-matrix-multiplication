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

def strassen_matmul_test(X, Y):
    n = X.shape[0]

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

    P = standard_matmul(A, F - H)
    P = P[:mid, :mid]
    Z12 += P
    Z22 += P

    P = standard_matmul(A + B, H)
    P = P[:mid, :mid]
    Z12 += P
    Z11 -= P

    P = standard_matmul(C + D, E)
    P = P[:mid, :mid]
    Z21 = P
    Z22 -= P

    P = standard_matmul(D, G - E)
    P = P[:mid, :mid]
    Z11 += P
    Z21 += P

    P = standard_matmul(A + D, E + H)
    P = P[:mid, :mid]
    Z11 += P
    Z22 += P

    P = standard_matmul(B - D, G + H)
    P = P[:mid, :mid]
    Z11 += P

    P = standard_matmul(C - A, E + F)
    P = P[:mid, :mid]
    Z22 += P

    
    Z = np.vstack((np.hstack((Z11, Z12)), np.hstack((Z21, Z22))))
    return Z

def find_crossover_point(n_values, trials):
    results = []  
    std_time = 0
    strassen_time = 0
    for n in n_values:
        print("n: ", n)
        std_faster_count = 0  
        for _ in range(trials):
            X = np.random.randint(0, 1, (n, n))
            Y = np.random.randint(0, 1, (n, n))
            
            start = time.perf_counter()
            standard_matmul(X, Y)
            std_time = time.perf_counter() - start

            start = time.perf_counter()
            strassen_matmul_test(X, Y)
            strassen_time = time.perf_counter() - start

            if std_time <= strassen_time:
                std_faster_count += 1  
            else:
                std_faster_count -= 1  

        if std_faster_count >= 0:
            results.append(n)
        

    return results


#odd: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29]
#even: [2, 4, 6, 8, 10]


n_values = [x for x in range(1, 101)]
results = find_crossover_point(n_values, 50)
odd = [n for n in results if n % 2 == 1]
even = [n for n in results if n % 2 == 0]
print(f"n where standard is faster - odd: {odd}, even: {even}")


def count_triangles(matrix):
    A3 = strassen_matmul(matrix, matrix)
    A3 = strassen_matmul(A3, matrix)
    triangles = np.trace(A3) // 6
    return triangles


n = 1024
p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
observed_triangles = []
expected_triangles = []

for p in p_values:
    A = (np.random.rand(n, n) < p).astype(int)
    np.fill_diagonal(A, 0)  
    A = np.triu(A)  
    A += A.T  
    
    num_triangles = count_triangles(A)
    observed_triangles.append(num_triangles)
    
    expected_triangles.append(comb(n, 3) * p**3)

print("Observed Triangles:", observed_triangles)
print("Expected Triangles:", expected_triangles)

plt.figure(figsize=(8, 6))
plt.plot(p_values, observed_triangles, label="Observed Triangles", marker='o')
plt.plot(p_values, expected_triangles, label="Expected Triangles", marker='s')
plt.xlabel("Edge Probability p")
plt.ylabel("Number of Triangles")
plt.title("Observed vs. Expected Number of Triangles in Random Graphs")
plt.legend()
plt.grid()
plt.show()
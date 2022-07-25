from naive_algorithm import multiply
from divide_and_conquer import multiply_matrix, strassen_multiplication
import random
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    n = 2**7
    A = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    B = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    start = time.time()
    C = multiply(A, B)
    end = time.time()
    print("NAIVE_TIME: ", end - start)

    start = time.time()
    c_div = multiply_matrix(A, B, 1)
    end = time.time()
    print("DIVIDE_AND_CONQUER_TIME: ", end - start)
    print(c_div == C)

    start = time.time()
    c_strass = strassen_multiplication(A, B, 1)
    end = time.time()
    print("STRASSEN_TIME: ", end - start)
    print(c_strass == C)

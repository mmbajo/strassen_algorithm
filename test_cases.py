from naive_algorithm import multiply
from divide_and_conquer import multiply_matrix, strassen_multiplication
import random
import time
import matplotlib.pyplot as plt
import pandas as pd


def collate_time(n, n_switch):
    A = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    B = [[random.randint(1, 10) for _ in range(n)] for _ in range(n)]
    start = time.time()
    C = multiply(A, B)
    end = time.time()
    n_time = end - start
    print("NAIVE_TIME: ", n_time)

    start = time.time()
    c_div = multiply_matrix(A, B, n_switch)
    end = time.time()
    dnc_time = end - start
    print("DIVIDE_AND_CONQUER_TIME: ", dnc_time)
    print(c_div == C)

    start = time.time()
    c_strass = strassen_multiplication(A, B, n_switch)
    end = time.time()
    s_time = end - start
    print("STRASSEN_TIME: ", s_time)
    print(c_strass == C)
    return (n_time, dnc_time, s_time)

if __name__ == "__main__":
    square_length = []
    n_times = []
    dnc_times = []
    s_times = []
    for i in range(1, 12):
        try:
            n, dnc, s = collate_time(2**i, 1)
            square_length.append(2**i)
            n_times.append(n)
            dnc_times.append(dnc)
            s_times.append(s)
        except:
            print("Not supported length case")

    df = pd.DataFrame()
    df["matrix_length"] = square_length
    df["naive_time"] = n_times
    df["dnc_time"] = dnc_times
    df["strassen_time"] = s_times
    df.to_csv("assets/results.csv")


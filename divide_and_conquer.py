from typing import List, Tuple
from naive_algorithm import multiply


def split_matrix(m: List[List[float]]) -> Tuple[List[List[float]]]:
    n = len(m)
    row, col = n // 2, n // 2

    a = [x[:col] for x in m[:row]]
    b = [x[col:] for x in m[:row]]
    c = [x[:col] for x in m[row:]]
    d = [x[col:] for x in m[row:]]
    return a, b, c, d


def combine_results(a, b, c, d):
    n = len(a)
    m = len(c)
    a_and_b = [a[i] + b[i] for i in range(n)]
    c_and_d = [c[i] + d[i] for i in range(m)]
    comb = a_and_b + c_and_d
    return comb


def add_matrix(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A[0])
    m = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def subtract_matrix(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A[0])
    m = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def multiply_matrix(
    A: List[List[float]], B: List[List[float]], n_switch=128
) -> List[List[float]]:
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    # Base case
    if n <= n_switch:
        C = multiply(A, B)
    else:
        # Split Matrix
        a, b, c, d = split_matrix(A)
        e, f, g, h = split_matrix(B)

        # Colate results for submatrices
        # Recurse until 1 x 1
        C_00 = add_matrix(multiply_matrix(a, e), multiply_matrix(b, g))  # subproblem

        C_01 = add_matrix(multiply_matrix(a, f), multiply_matrix(b, h))

        C_10 = add_matrix(multiply_matrix(c, e), multiply_matrix(d, g))

        C_11 = add_matrix(multiply_matrix(c, f), multiply_matrix(d, h))

        # Combine results
        C = combine_results(C_00, C_01, C_10, C_11)
    return C


def strassen_multiplication(
    A: List[List[float]], B: List[List[float]], n_switch=128
) -> List[List[float]]:
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    # Base case
    if n <= n_switch:
        C = multiply(A, B)
    else:
        # Split Matrix
        a, b, c, d = split_matrix(A)
        e, f, g, h = split_matrix(B)

        # Colate results for submatrices
        p1 = strassen_multiplication(a, subtract_matrix(f, h))
        p2 = strassen_multiplication(add_matrix(a, b), h)
        p3 = strassen_multiplication(add_matrix(c, d), e)
        p4 = strassen_multiplication(d, subtract_matrix(g, e))
        p5 = strassen_multiplication(add_matrix(a, d), add_matrix(e, h))
        p6 = strassen_multiplication(subtract_matrix(b, d), add_matrix(g, h))
        p7 = strassen_multiplication(subtract_matrix(a, c), add_matrix(e, f))

        # Calculate C submatrices
        C_00 = subtract_matrix(add_matrix(p4, p5), subtract_matrix(p2, p6))
        C_01 = add_matrix(p1, p2)
        C_10 = add_matrix(p3, p4)
        C_11 = subtract_matrix(add_matrix(p1, p5), add_matrix(p3, p7))

        # Combine results
        C = combine_results(C_00, C_01, C_10, C_11)
    return C

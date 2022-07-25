from typing import List


def multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

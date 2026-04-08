def matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Multiply two matrices and return the result."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError("Incompatible matrix dimensions")

    # BUG 1: result matrix dimensions wrong (should be rows_a x cols_b)
    result = [[0] * cols_a for _ in range(rows_b)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                # BUG 2: accumulation uses wrong index
                result[i][j] += a[i][k] * b[j][k]  # should be b[k][j]

    return result


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    """Transpose a matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    # BUG 3: indices swapped in comprehension
    return [[matrix[j][i] for j in range(rows)] for i in range(rows)]  # should be range(cols)

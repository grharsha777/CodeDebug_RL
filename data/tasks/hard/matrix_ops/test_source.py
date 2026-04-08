from solution import matrix_multiply, transpose


def test_multiply_identity():
    a = [[1, 0], [0, 1]]
    b = [[5, 6], [7, 8]]
    assert matrix_multiply(a, b) == [[5, 6], [7, 8]]


def test_multiply_basic():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert matrix_multiply(a, b) == [[19, 22], [43, 50]]


def test_multiply_non_square():
    a = [[1, 2, 3]]
    b = [[4], [5], [6]]
    assert matrix_multiply(a, b) == [[32]]


def test_multiply_incompatible_raises():
    a = [[1, 2]]
    b = [[3, 4]]
    try:
        matrix_multiply(a, b)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_transpose_square():
    m = [[1, 2], [3, 4]]
    assert transpose(m) == [[1, 3], [2, 4]]


def test_transpose_rectangular():
    m = [[1, 2, 3], [4, 5, 6]]
    assert transpose(m) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_single_row():
    m = [[1, 2, 3]]
    assert transpose(m) == [[1], [2], [3]]


def test_transpose_empty():
    assert transpose([]) == []


def test_transpose_involutory():
    m = [[1, 2], [3, 4]]
    assert transpose(transpose(m)) == m

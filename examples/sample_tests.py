from sample_buggy_program import merge_sorted


def test_basic_merge():
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]


def test_one_empty():
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted([1, 2, 3], []) == [1, 2, 3]


def test_both_empty():
    assert merge_sorted([], []) == []


def test_duplicates():
    assert merge_sorted([1, 2, 2], [2, 3]) == [1, 2, 2, 2, 3]


def test_single_elements():
    assert merge_sorted([1], [2]) == [1, 2]


def test_uneven_lengths():
    assert merge_sorted([1, 5], [2, 3, 4, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]

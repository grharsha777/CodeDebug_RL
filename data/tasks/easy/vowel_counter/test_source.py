from solution import count_vowels, most_common_vowel


def test_lowercase():
    assert count_vowels("hello") == 2


def test_uppercase():
    assert count_vowels("HELLO") == 2


def test_mixed_case():
    assert count_vowels("Hello World") == 3


def test_no_vowels():
    assert count_vowels("bcdfg") == 0


def test_empty():
    assert count_vowels("") == 0


def test_all_vowels():
    assert count_vowels("aeiou") == 5


def test_most_common():
    assert most_common_vowel("aabbaa") == "a"


def test_most_common_empty():
    assert most_common_vowel("xyz") == ""

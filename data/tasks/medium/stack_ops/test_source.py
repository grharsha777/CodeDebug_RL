from solution import Stack, is_balanced


def test_push_pop_order():
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.pop() == 3  # LIFO: last in, first out


def test_peek():
    s = Stack()
    s.push("a")
    s.push("b")
    assert s.peek() == "b"


def test_size():
    s = Stack()
    assert s.size() == 0
    s.push(1)
    assert s.size() == 1
    s.pop()
    assert s.size() == 0


def test_empty_pop_raises():
    s = Stack()
    try:
        s.pop()
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_balanced_simple():
    assert is_balanced("()") is True
    assert is_balanced("[]") is True
    assert is_balanced("{}") is True


def test_balanced_nested():
    assert is_balanced("({[]})") is True


def test_unbalanced():
    assert is_balanced("(]") is False
    assert is_balanced("({)}") is False


def test_balanced_empty():
    assert is_balanced("") is True


def test_balanced_complex():
    assert is_balanced("[({})](){}") is True

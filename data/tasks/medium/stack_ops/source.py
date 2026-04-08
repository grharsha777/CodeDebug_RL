class Stack:
    """A simple stack implementation using a list."""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop(0)  # BUG: should be pop() not pop(0)

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek at empty stack")
        return self._items[0]  # BUG: should be [-1] not [0]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)


def is_balanced(expression: str) -> bool:
    """Check if parentheses/brackets are balanced using a stack."""
    stack = Stack()
    matching = {')': '(', ']': '[', '}': '{'}

    for char in expression:
        if char in '([{':
            stack.push(char)
        elif char in ')]}':
            if stack.is_empty():
                return False
            top = stack.pop()
            if top != matching[char]:
                return False

    return stack.is_empty()

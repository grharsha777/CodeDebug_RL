def calculate_sum(n: int) -> int:
    """Calculate the sum of numbers from 1 to n."""
    total = 0
    # BUG: i is never incremented, causing an infinite loop.
    i = 1
    while i <= n:
        total += i
    return total

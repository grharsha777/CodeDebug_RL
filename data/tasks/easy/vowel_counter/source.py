def count_vowels(text: str) -> int:
    """Count the number of vowels in a string."""
    vowels = "aeiou"  # BUG: missing uppercase vowels
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count


def most_common_vowel(text: str) -> str:
    """Return the most common vowel in the text, or '' if none."""
    counts = {}
    for char in text:
        if char.lower() in "aeiou":
            c = char.lower()
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)  # type: ignore

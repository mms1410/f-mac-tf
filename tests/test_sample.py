"""Do dummy test."""


def inc(x):
    """Increment some number.

    Args:
        x: some number to increment.

    Returns:
        some number.
    """
    return x + 1


def test_answer():
    """Do actual test."""
    assert inc(3) == 4

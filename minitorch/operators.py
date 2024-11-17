"""Collection of the core mathematical operators used throughout the code base."""

import math
import operator as op
from typing import Any, Callable, Iterable


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return op.add(x, y)


def neg(x: float) -> float:
    """Additive inverse of a number"""
    return op.neg(x)


def sub(x: float, y: float) -> float:
    """Subtracts two numbers"""
    return add(x, neg(y))


def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return op.mul(x, y)


def inv(x: float) -> float:
    """Multiplicative inverse of a number"""
    return op.truediv(1.0, x)


def div(x: float, y: float) -> float:
    """Divides two numbers"""
    return mul(x, inv(y))


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def lt(x: float, y: float) -> bool:
    """Checks if one number (x) is less than another (y)"""
    return op.lt(x, y)


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return op.eq(x, y)


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return y if lt(x, y) else x


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return lt(abs(x - y), 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if lt(0, x) or eq(0, x):
        # compute e^(-x) first to avoid overflow when x is large positive
        z = exp(neg(x))
        # when x is large positive, z ≈ 0, so result ≈ 1.0
        return div(1.0, add(1.0, z))
    else:
        # compute e^x first to avoid overflow when x is large negative
        z = exp(x)
        # when x is large negative, z ≈ 0, so result ≈ 0.0
        return div(z, add(1.0, z))


def sigmoid_symmetric(x: float) -> float:
    """Calculates the sigmoid of a number using the symmetry property:
    sigmoid(-x) = 1 - sigmoid(x)
    """
    if lt(0, x) or eq(0, x):
        # compute e^(-x) first to avoid overflow
        z = exp(neg(x))
        return div(1.0, add(1.0, z))
    else:
        # leverage symmetry property
        return sub(1.0, sigmoid_symmetric(neg(x)))


def sigmoid_back(x: float, y: float) -> float:
    """Computes the derivative of sigmoid times a second arg"""
    # sigmoid(x) = 1 / (1 + e^(-x))
    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    sigmoid_derivative = mul(sigmoid(x), sub(1.0, sigmoid(x)))
    return mul(y, sigmoid_derivative)


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return 0 if lt(x, 0) else x


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return mul(y, inv(x))


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return mul(y, neg(inv(x**2)))


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return 0 if lt(x, 0) else y


def map(f: Callable, iterable: Iterable) -> Iterable:
    """Higher-order function that applies a given function to each element of an iterable"""
    return (f(element) for element in iterable)


def zipWith(f: Callable, *iterables: Iterable) -> Iterable:
    """Higher-order function that applies a given function to each element of multiple iterables"""
    return map(lambda args: f(*args), zip(*iterables))


def reduce(f: Callable, iterable: Iterable, initial: Any = None) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    reduced = initial
    iterator = iter(iterable)
    for element in iterator:
        reduced = f(reduced, element)
    return reduced


def negList(iterable: Iterable) -> Iterable:
    """Negate all elements in a list using map"""
    return map(neg, iterable)


def addLists(iterable1: Iterable, iterable2: Iterable) -> Iterable:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, iterable1, iterable2)


def sum(iterable: Iterable) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, iterable, 0)


def prod(iterable: Iterable) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, iterable, 1)

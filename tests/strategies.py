from hypothesis import settings
from hypothesis.strategies import floats, integers

import minitorch


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=200)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=2000)
med_floats = floats(min_value=-1000, max_value=1000, allow_nan=False)
large_ints = integers(min_value=1, max_value=20000)
large_floats = floats(min_value=-10000, max_value=10000, allow_nan=False)


def assert_close(a: float, b: float) -> None:
    assert minitorch.operators.is_close(a, b), f"Failure x={a} y={b}"

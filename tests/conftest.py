"""Settings shared by the test suite."""

from hypothesis import settings

# Property examples build arrays whose size, and so whose time, varies with the drawn values, so Hypothesis's 200 ms
# deadline per example would fail them on a slow machine. Each module's settings(...) inherit the profile.
settings.register_profile("irtlib", deadline=None)
settings.load_profile("irtlib")

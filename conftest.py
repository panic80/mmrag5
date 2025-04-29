import pytest

def pytest_runtest_setup(item):
    # Skip integration tests by default when marked
    if 'integration' in item.keywords:
        pytest.skip("skipping integration test")
"""
Pytest configuration - suppress warnings from generated and dependency code
"""

import warnings


def pytest_configure(config):
    """Configure pytest to suppress deprecation warnings from generated files and dependencies"""

    # Suppress protobuf deprecation warnings from generated _pb2.py files
    # These are from auto-generated protobuf files that use deprecated API
    # Can't be easily fixed without regenerating with newer protobuf compiler
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*Call to deprecated create function.*",
    )

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=".*communicator_objects.*pb2",
    )

    # Suppress pytest-asyncio configuration warning
    warnings.filterwarnings(
        "ignore",
        message=".*asyncio_default_fixture_loop_scope.*",
    )

"""Pytest configuration for ml-agents-envs tests"""

import pytest


@pytest.fixture
def base_port():
    """Fixture providing base port number for Unity environment tests"""
    return 5004

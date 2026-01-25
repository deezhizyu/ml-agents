"""Pytest configuration for ml-agents-envs tests"""

import pytest


@pytest.fixture
def n_ports():
    """Fixture providing port numbers for Unity environment tests"""
    return [5004, 5005, 5006, 5007]

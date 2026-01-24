"""
Tests for CLI doctor diagnostic tool - Phase 3
"""

import pytest
from mlagents.trainers.cli_doctor import MLAgentsDoctor, DiagnosticCheck


class TestDiagnosticCheck:
    """Test DiagnosticCheck class"""

    def test_diagnostic_check_creation(self):
        """Test creating a diagnostic check"""
        check = DiagnosticCheck(
            "Test Check", lambda: (True, "Success"), "Fix suggestion"
        )

        assert check.name == "Test Check"
        assert check.fix_suggestion == "Fix suggestion"
        assert check.result is None

    def test_diagnostic_check_run_success(self):
        """Test running a successful check"""
        check = DiagnosticCheck("Success Check", lambda: (True, "All good"))

        result = check.run()
        assert result
        assert check.result == (True, "All good")

    def test_diagnostic_check_run_failure(self):
        """Test running a failing check"""
        check = DiagnosticCheck("Fail Check", lambda: (False, "Failed"))

        result = check.run()
        assert not result
        assert check.result == (False, "Failed")

    def test_diagnostic_check_exception(self):
        """Test check that raises exception"""

        def failing_check():
            raise ValueError("Test error")

        check = DiagnosticCheck("Error Check", failing_check)
        result = check.run()

        assert result == False
        assert "Test error" in check.result[1]


class TestMLAgentsDoctor:
    """Test MLAgentsDoctor class"""

    def test_doctor_creation(self):
        """Test creating ML-Agents doctor"""
        doctor = MLAgentsDoctor()

        assert doctor is not None
        assert len(doctor.checks) > 0

    def test_doctor_has_required_checks(self):
        """Test that doctor has all required checks"""
        doctor = MLAgentsDoctor()

        check_names = [check.name for check in doctor.checks]

        required_checks = [
            "Python Version",
            "PyTorch Installation",
            "NumPy Installation",
        ]

        for required in required_checks:
            assert required in check_names

    def test_python_version_check(self):
        """Test Python version check"""
        doctor = MLAgentsDoctor()

        # Find Python version check
        python_check = next(
            (c for c in doctor.checks if c.name == "Python Version"), None
        )

        assert python_check is not None
        result = python_check.run()

        # Should pass since we're running Python
        assert result == True

    def test_pytorch_check(self):
        """Test PyTorch installation check"""
        doctor = MLAgentsDoctor()

        pytorch_check = next(
            (c for c in doctor.checks if c.name == "PyTorch Installation"), None
        )

        assert pytorch_check is not None
        result = pytorch_check.run()

        # Result depends on environment
        assert isinstance(result, bool)

    def test_quick_check(self):
        """Test quick diagnostic check"""
        doctor = MLAgentsDoctor()
        result = doctor.quick_check()

        # Should return boolean
        assert isinstance(result, bool)


class TestDoctorIntegration:
    """Integration tests for doctor tool"""

    def test_run_full_diagnostics(self):
        """Test running full diagnostic suite"""
        doctor = MLAgentsDoctor()
        result = doctor.run_diagnostics()

        # Should complete without crashing
        assert isinstance(result, bool)

    def test_all_checks_execute(self):
        """Test that all checks can execute"""
        doctor = MLAgentsDoctor()

        for check in doctor.checks:
            # Should not raise exception
            check.run()
            assert check.result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

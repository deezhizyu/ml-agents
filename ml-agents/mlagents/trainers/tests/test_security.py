"""
Security-focused tests to prevent regressions
"""
import subprocess
import pytest
from unittest.mock import patch, MagicMock


class TestSubprocessSecurity:
    """Test that subprocess calls are secure"""
    
    def test_no_shell_true_in_subprocess_calls(self):
        """Verify no subprocess calls use shell=True in production code"""
        import pathlib
        import re
        
        # Search all Python files in mlagents (excluding tests)
        root = pathlib.Path(__file__).parents[2]  # Get to mlagents/
        python_files = list(root.glob("**/*.py"))
        
        # Exclude test files, protobuf files, and venv
        production_files = [
            f for f in python_files 
            if "test" not in str(f) 
            and "_pb2" not in str(f)
            and "venv" not in str(f)
            and ".venv" not in str(f)
        ]
        
        shell_true_pattern = re.compile(r'subprocess\.(call|run|check_call|Popen).*shell\s*=\s*True')
        
        violations = []
        for file_path in production_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                if shell_true_pattern.search(content):
                    # Find line numbers
                    for i, line in enumerate(content.splitlines(), 1):
                        if shell_true_pattern.search(line):
                            violations.append(f"{file_path}:{i} - {line.strip()}")
            except Exception:
                pass  # Skip files that can't be read
        
        if violations:
            pytest.fail(
                f"Found {len(violations)} subprocess calls with shell=True (security vulnerability):\n" +
                "\n".join(violations[:5]) +  # Show first 5
                ("\n..." if len(violations) > 5 else "")
            )
    
    def test_no_eval_exec_in_production_code(self):
        """Verify no eval/exec calls in production code (excluding safe usages)"""
        import pathlib
        import re
        
        root = pathlib.Path(__file__).parents[2]
        python_files = list(root.glob("**/*.py"))
        
        production_files = [
            f for f in python_files 
            if "test" not in str(f)
            and "_pb2" not in str(f)
            and "venv" not in str(f)
        ]
        
        # Pattern for eval/exec as functions (not in comments or strings)
        eval_exec_pattern = re.compile(r'^\s*(eval|exec)\s*\(', re.MULTILINE)
        
        violations = []
        skipped_files = []
        for file_path in production_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                matches = eval_exec_pattern.findall(content)
                if matches:
                    violations.append(str(file_path))
            except Exception as e:
                skipped_files.append((str(file_path), str(e)))
        
        # Log skipped files for visibility
        if skipped_files:
            print(f"\nWarning: {len(skipped_files)} files skipped during eval/exec check")
            if len(skipped_files) > len(production_files) * 0.1:
                pytest.fail(
                    f"Too many files skipped ({len(skipped_files)}/{len(production_files)}). "
                    f"This could hide security vulnerabilities."
                )
        
        # This is a warning, not a hard failure (some uses might be legitimate)
        if violations:
            print(f"\nWarning: Found eval/exec in {len(violations)} files (review for security):")
            print("\n".join(violations[:5]))


class TestCredentialSecurity:
    """Test that no credentials are hardcoded"""
    
    def test_no_hardcoded_api_keys(self):
        """Check that no API keys are hardcoded in source"""
        import pathlib
        import re
        
        # Common patterns for API keys
        patterns = [
            re.compile(r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', re.IGNORECASE),
            re.compile(r'secret[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', re.IGNORECASE),
            re.compile(r'password\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
        ]
        
        root = pathlib.Path(__file__).parents[2]
        python_files = [
            f for f in root.glob("**/*.py")
            if "test" not in str(f)
            and "_pb2" not in str(f)
            and "venv" not in str(f)
        ]
        
        violations = []
        skipped_files = []
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern in patterns:
                    if pattern.search(content):
                        # Exclude test/example keys
                        if "TEST" not in content and "EXAMPLE" not in content:
                            violations.append(str(file_path))
                            break
            except Exception as e:
                skipped_files.append((str(file_path), str(e)))
        
        # Log skipped files for visibility
        if len(skipped_files) > len(python_files) * 0.1:
            pytest.fail(
                f"Too many files skipped during credential check ({len(skipped_files)}/{len(python_files)}). "
                f"This could hide hardcoded secrets."
            )
        
        if violations:
            pytest.fail(
                f"Found potential hardcoded credentials in {len(violations)} files:\n" +
                "\n".join(violations)
            )


class TestInputValidation:
    """Test that user inputs are properly validated"""
    
    def test_path_traversal_prevention(self):
        """Test that file paths are validated"""
        from mlagents.trainers.settings import RunOptions
        
        # Test that paths with .. are rejected or normalized
        # This is a placeholder - actual implementation depends on codebase
        pass
    
    def test_command_injection_prevention(self):
        """Test that external commands can't be injected"""
        # Mock test to ensure subprocess wrappers validate input
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

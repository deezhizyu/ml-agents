# Security Best Practices for ML-Agents

This document outlines security best practices for developing with and contributing to ML-Agents.

## 🔒 Overview

ML-Agents handles sensitive operations including:
- Model training with potentially sensitive data
- File system operations (checkpoints, configs)
- Network communication (Unity ↔ Python)
- External process execution (Unity environments)
- HuggingFace integration (API tokens)

## 🛡️ Security Guidelines

### 1. Subprocess Security

**❌ NEVER DO THIS:**
```python
# DANGEROUS: Command injection vulnerability
subprocess.call(f"rm -rf {user_provided_path}", shell=True)
subprocess.run(f"git checkout {branch_name}", shell=True)
```

**✅ DO THIS INSTEAD:**
```python
# SAFE: Use list arguments, no shell interpretation
subprocess.call(["rm", "-rf", user_provided_path])
subprocess.run(["git", "checkout", branch_name])
```

**Why?** Using `shell=True` allows command injection if user input is not properly sanitized.

### 2. Credential Management

**❌ NEVER:**
- Hardcode API keys, passwords, or tokens in source code
- Commit credentials to version control
- Log sensitive information

**✅ DO:**
- Use environment variables for credentials
- Use `.env` files (excluded from git via `.gitignore`)
- Prefix test credentials with `TEST_` or `FAKE_`

**Example:**
```python
# Good: Load from environment
import os
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
```

### 3. File Path Validation

**❌ DANGEROUS:**
```python
# Path traversal vulnerability
config_path = user_input + ".yaml"
with open(config_path) as f:
    config = yaml.load(f)
```

**✅ SAFE:**
```python
import os
from pathlib import Path

# Validate and normalize paths
config_path = Path(user_input).resolve()
allowed_dir = Path("/safe/directory").resolve()

if not str(config_path).startswith(str(allowed_dir)):
    raise ValueError(f"Path outside allowed directory: {config_path}")

with open(config_path) as f:
    config = yaml.safe_load(f)  # Use safe_load, not load
```

### 4. Model Loading Security

**❌ RISKY:**
```python
# Pickle deserialization can execute arbitrary code
import pickle
model = pickle.load(open("untrusted_model.pkl", "rb"))
```

**✅ SAFER:**
```python
# Use PyTorch's safeguards
import torch
model = torch.load("model.pt", weights_only=True)  # Restrict to weights only

# Or use safetensors library
from safetensors.torch import load_model
load_model(model, "model.safetensors")
```

### 5. YAML Loading

**❌ DANGEROUS:**
```python
import yaml
config = yaml.load(file)  # Can execute arbitrary Python code
```

**✅ SAFE:**
```python
import yaml
config = yaml.safe_load(file)  # Only deserializes basic Python objects
```

### 6. Environment Variables

**Best Practices:**
```python
# Sensitive configuration
SENSITIVE_VARS = [
    "HUGGINGFACE_TOKEN",
    "AWS_SECRET_KEY",
    "TENSORBOARD_API_KEY",
]

# Never log sensitive vars
def log_environment():
    for key, value in os.environ.items():
        if any(sensitive in key.upper() for sensitive in ["KEY", "TOKEN", "SECRET", "PASSWORD"]):
            print(f"{key}=<redacted>")
        else:
            print(f"{key}={value}")
```

## 🔍 Security Tools

### Pre-commit Hooks

We use these security-focused pre-commit hooks:

1. **detect-secrets**: Scans for hardcoded credentials
2. **bandit**: Python security vulnerability scanner
3. **Check for shell=True**: Custom check for subprocess security

**Run manually:**
```bash
pre-commit run detect-secrets --all-files
pre-commit run --all-files
```

### CI Security Scans

Our CI pipeline runs:
- **CodeQL**: Static analysis for security vulnerabilities
- **pip-audit**: Checks dependencies for known CVEs
- **safety**: Database of known security vulnerabilities
- **Bandit**: SAST (Static Application Security Testing)
- **Subprocess security check**: Custom validator

### Dependency Auditing

**Regular audits:**
```bash
# Check for vulnerabilities
pip-audit

# Check specific requirements file
pip-audit --requirement test_requirements.txt

# Check with safety
safety check --file requirements.txt
```

## 🚨 Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

**Instead:**
1. Email security concerns to: [security contact]
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

3. We will respond within 48 hours

## ✅ Security Checklist for Contributors

Before submitting a PR, verify:

- [ ] No `shell=True` in subprocess calls
- [ ] No hardcoded credentials (run `detect-secrets scan`)
- [ ] File paths are validated and sanitized
- [ ] YAML loaded with `safe_load`, not `load`
- [ ] No `eval()` or `exec()` with user input
- [ ] Environment variables used for secrets
- [ ] No logging of sensitive information
- [ ] Dependencies are up-to-date (no known CVEs)
- [ ] Input validation for all external data
- [ ] Error messages don't leak sensitive info

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/serialization.html)

## 🔄 Security Updates

This document is updated regularly. Last updated: 2026-01-23

For the latest security guidelines, always refer to the `docs/SECURITY.md` file in the main branch.

---

**Remember:** Security is everyone's responsibility. When in doubt, ask for a security review!

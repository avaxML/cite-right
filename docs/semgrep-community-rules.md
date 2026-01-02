# Semgrep Community Rules for Cite-Right

This document explores the available Semgrep community rules that are relevant for the cite-right project, a Python/Rust hybrid library for text alignment in AI-generated citations.

## Current Security Configuration

Cite-right already has security scanning configured in the CI pipeline (`.github/workflows/ci.yml`):
- **Bandit**: Python security linter
- **pip-audit**: Dependency vulnerability scanning
- **Semgrep**: Currently runs with custom `.semgrep.yml` + `--config=auto`

The current `.semgrep.yml` contains 3 custom rules:
- `python-security`: Detects `eval()` usage
- `hardcoded-password`: Warns about hardcoded passwords
- `sql-injection-risk`: Detects SQL injection patterns

## Recommended Community Rulesets

### 1. Core Security Rulesets

| Ruleset | Command | Description |
|---------|---------|-------------|
| **p/python** | `--config p/python` | Python-specific security rules covering common vulnerabilities |
| **p/owasp-top-ten** | `--config p/owasp-top-ten` | OWASP Top 10 web application security risks |
| **p/cwe-top-25** | `--config p/cwe-top-25` | CWE Top 25 application security vulnerabilities |
| **p/security-audit** | `--config p/security-audit` | Lower-confidence audit rules for code review |
| **p/bandit** | `--config p/bandit` | Bandit-equivalent rules in Semgrep format |

### 2. Trail of Bits ML/AI Security Rules (Highly Recommended)

The [Trail of Bits Semgrep rules](https://github.com/trailofbits/semgrep-rules) are particularly relevant for cite-right given its use of ML libraries.

| Rule | Applies To | What It Detects |
|------|-----------|-----------------|
| `pickles-in-pytorch` | `sentence-transformers`, PyTorch | Arbitrary code execution from PyTorch pickling |
| `pickles-in-tensorflow` | TensorFlow | TensorFlow load function vulnerabilities |
| `pickles-in-pandas` | Data processing | Pandas functions relying on unsafe pickling |
| `numpy-load-library` | NumPy | Potential code execution from NumPy library loading |
| `pytorch-package` | PyTorch | `torch.package` arbitrary execution risks |
| `pandas-eval` | Data processing | User-expression evaluation vulnerabilities |
| `onnx-session-options` | ONNX models | ONNX library loading risks |
| `numpy-in-pytorch-datasets` | PyTorch/NumPy | NumPy RNG usage in Torch datasets |
| `lxml-in-pandas` | XML processing | Potential XXE attacks from lxml in pandas |

**Usage:**
```bash
semgrep --config p/trailofbits
```

Or directly from the repository:
```bash
semgrep --config https://semgrep.dev/p/trailofbits
```

### 3. Deserialization Security Rules

Given cite-right's use of ML model loading (sentence-transformers, HuggingFace), these rules are critical:

| Rule ID | Description |
|---------|-------------|
| `python.lang.security.deserialization.pickle.avoid-pickle` | Flags `pickle.load`, `pickle.loads` usage |
| `python.lang.security.deserialization.avoid-pyyaml-load` | Unsafe PyYAML loading |
| `python.lang.security.deserialization.avoid-dill` | Dill deserialization risks |
| `python.lang.security.deserialization.avoid-jsonpickle` | jsonpickle vulnerabilities |
| `python.lang.security.deserialization.avoid-shelve` | Shelve module risks |

**Relevance to cite-right:**
- `sentence-transformers` loads models that may use pickle internally
- HuggingFace `transformers` has model loading that can trigger arbitrary code
- The project already exempts `B615` (huggingface_unsafe_download) in Bandit

### 4. Python Framework-Native Analysis

Semgrep's framework-native analysis provides coverage for:
- **Pydantic**: Data validation patterns (cite-right uses Pydantic v2)
- **FastAPI/Flask/Django**: If APIs are built on top of cite-right
- **NumPy**: Array operations and loading patterns

The `--config=auto` flag already enables many of these, but explicit configuration gives more control.

## Cite-Right Specific Recommendations

Based on the project's technology stack:

### High Priority (Direct Relevance)

1. **ML Model Loading Security**
   ```yaml
   # Add to .semgrep.yml or use p/trailofbits
   - id: unsafe-numpy-load
     patterns:
       - pattern: numpy.load(..., allow_pickle=True)
     message: Unsafe numpy.load with pickle enabled
     severity: ERROR
     languages: [python]

   - id: torch-unsafe-load
     patterns:
       - pattern: torch.load($PATH)
       - pattern-not: torch.load($PATH, weights_only=True)
     message: Use weights_only=True with torch.load to prevent arbitrary code execution
     severity: WARNING
     languages: [python]
   ```

2. **HuggingFace Model Loading**
   ```yaml
   - id: hf-model-trust-remote-code
     patterns:
       - pattern: |
           $MODEL.from_pretrained(..., trust_remote_code=True, ...)
     message: trust_remote_code=True allows arbitrary code execution from model repos
     severity: WARNING
     languages: [python]
   ```

### Medium Priority (General Python Security)

3. **Input Validation** - Already covered by Pydantic, but additional checks:
   - Path traversal in file operations
   - Command injection patterns
   - Regex denial of service (ReDoS)

4. **Rust FFI Safety** (for `rust_core`):
   - Currently no Semgrep rules for PyO3/Rust FFI patterns
   - Use `cargo clippy` (already configured) for Rust-side safety

## Recommended CI Configuration Update

Update the security job in `.github/workflows/ci.yml`:

```yaml
- name: Run Semgrep security scan
  run: |
    uv pip install semgrep
    uv run semgrep scan \
      --config=.semgrep.yml \
      --config=p/python \
      --config=p/trailofbits \
      --config=p/owasp-top-ten \
      --config=p/security-audit \
      src/
```

Or for maximum coverage with auto-detection:
```yaml
- name: Run Semgrep security scan
  run: |
    uv pip install semgrep
    uv run semgrep scan \
      --config=.semgrep.yml \
      --config=auto \
      --config=p/trailofbits \
      src/
```

## Rule Severity Levels

| Level | When to Use |
|-------|-------------|
| `ERROR` | Critical security issues, should fail CI |
| `WARNING` | Important issues, review required |
| `INFO` | Best practice suggestions |

## Testing Rules Locally

```bash
# Test a specific ruleset
semgrep --config p/trailofbits src/

# Test with verbose output
semgrep --config p/python --verbose src/

# Test specific patterns
semgrep --config https://semgrep.dev/r/python.lang.security.deserialization.pickle.avoid-pickle src/

# Dry run to see what would be flagged
semgrep --config auto --dryrun src/
```

## Sources and References

- [Semgrep Registry - Python](https://registry.semgrep.dev/tag/python)
- [Semgrep OWASP Top 10 Ruleset](https://semgrep.dev/p/owasp-top-ten)
- [Trail of Bits Semgrep Rules](https://github.com/trailofbits/semgrep-rules)
- [Trail of Bits Blog: Secure your ML with Semgrep](https://blog.trailofbits.com/2022/10/03/semgrep-maching-learning-static-analysis/)
- [Semgrep Insecure Deserialization Docs](https://semgrep.dev/docs/learn/vulnerabilities/insecure-deserialization/python)
- [Semgrep Supply Chain](https://semgrep.dev/products/semgrep-supply-chain/)
- [Semgrep Framework-Native Python Analysis](https://semgrep.dev/blog/2024/redefining-security-coverage-for-python-with-framework-native-analysis/)
- [AWS: The Little Pickle Story](https://aws.amazon.com/blogs/security/enhancing-cloud-security-in-ai-ml-the-little-pickle-story/)

## Summary

For cite-right, the most valuable Semgrep community rules are:

1. **p/trailofbits** - ML/AI-specific security rules (pickle, PyTorch, NumPy)
2. **p/python** - General Python security patterns
3. **p/owasp-top-ten** - Web security fundamentals
4. **p/security-audit** - Deeper audit-level checks

The Trail of Bits rules are particularly important given cite-right's use of `sentence-transformers`, `transformers`, and `numpy`, all of which can have deserialization vulnerabilities when loading models from untrusted sources.

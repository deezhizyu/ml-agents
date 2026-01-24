# ML-Agents Runbooks

This directory contains operational runbooks for the ML-Agents Toolkit. These runbooks provide step-by-step procedures for handling common operational scenarios.

## Available Runbooks

### Training Issues

- **[Training Convergence Issues](./training-convergence.md)** - Troubleshooting when agents fail to learn
- **[GPU Memory Issues](./gpu-memory.md)** - Handling out-of-memory errors during training
- **[Connection Failures](./connection-failures.md)** - Resolving Unity environment connection issues

### Performance Issues

- **[Slow Training Performance](./slow-training.md)** - Diagnosing and fixing slow training
- **[Inference Performance](./inference-performance.md)** - Optimizing model inference speed

### Deployment Issues

- **[Model Export Failures](./model-export.md)** - Troubleshooting ONNX export issues
- **[Unity Integration Issues](./unity-integration.md)** - Resolving model loading in Unity

### CI/CD Issues

- **[Build Failures](./build-failures.md)** - Handling package build errors
- **[Test Failures](./test-failures.md)** - Investigating test failures in CI

## Quick Reference

### Training Not Converging

1. Check reward signal configuration
2. Verify observation space is correct
3. Adjust hyperparameters (learning rate, batch size)
4. Increase max_steps if needed
5. Review TensorBoard logs for patterns

### Unity Connection Timeout

1. Verify Unity build path is correct
2. Check port 5005 is not in use: `netstat -an | grep 5005` (Linux/Mac) or `netstat -an | findstr 5005` (Windows)
3. Ensure no firewall blocking
4. Try increasing timeout in UnityEnvironment initialization
5. Check Unity logs for errors

### GPU Not Detected

1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure correct PyTorch version for CUDA version
4. Set `CUDA_VISIBLE_DEVICES` environment variable
5. Reinstall PyTorch with CUDA support

### Out of Memory During Training

1. Reduce `batch_size` in training config
2. Reduce `buffer_size`
3. Decrease number of parallel environments
4. Use gradient checkpointing if available
5. Monitor GPU memory: `watch -n 1 nvidia-smi`

## External Resources

### Official Documentation
- [Unity ML-Agents Documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)
- [Training Configuration Reference](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Training-Configuration-File.html)

### Community Resources
- [Unity Discussions - ML-Agents](https://discussions.unity.com/tag/ml-agents)
- [GitHub Issues](https://github.com/Unity-Technologies/ml-agents/issues)
- [Discord Community](https://discord.com/channels/489222168727519232/1202574086115557446)

### Internal Resources
For Unity employees and maintainers:
- Internal Slack: `#machine-learning`
- Confluence: [ML-Agents Operations Guide](https://unity.atlassian.net/) *(internal only)*

## Creating New Runbooks

When creating a new runbook, follow this template:

```markdown
# [Issue Title]

## Symptoms
- Specific error messages
- Observable behavior
- Affected components

## Diagnosis Steps
1. Step-by-step investigation
2. What to check
3. How to verify the issue

## Resolution
1. Step-by-step fix
2. Configuration changes
3. Verification steps

## Prevention
- How to avoid this issue
- Monitoring recommendations
- Configuration best practices

## Related Issues
- Links to GitHub issues
- Similar problems
```

## Contributing

If you encounter an operational issue not covered here:
1. Document your troubleshooting steps
2. Create a runbook following the template
3. Submit a PR with the new runbook
4. Update this index

## Escalation

If runbooks don't resolve your issue:
1. Check [GitHub Issues](https://github.com/Unity-Technologies/ml-agents/issues) for similar problems
2. Search [Unity Discussions](https://discussions.unity.com/tag/ml-agents)
3. Create a new issue with detailed information
4. For urgent production issues (Unity internal): Contact @unity/behavior-authoring on Slack

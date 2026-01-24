---
name: Export Models
description: Export trained ML-Agents models to ONNX format for Unity deployment and push to HuggingFace Hub for sharing
---

# Export Models Skill

Export and deploy trained ML-Agents models.

## When to Use

- Training completed successfully
- Ready to deploy model to Unity
- Want to share model on HuggingFace
- Need to test model in Unity Editor
- Archiving trained models

## Automatic Export

Models are automatically exported during training to ONNX format:

```
results/<run-id>/
├── <BehaviorName>.onnx        # Exported ONNX model
├── <BehaviorName>/
│   └── checkpoint.pt          # PyTorch checkpoint
└── configuration.yaml         # Training config
```

## Manual ONNX Export

If you need to re-export:

```python
from mlagents.trainers.torch_entities.model_serialization import export_policy_model

# Load checkpoint and export
export_policy_model(
    checkpoint_path="results/MyRun/MyBehavior/checkpoint.pt",
    output_filepath="results/MyRun/MyBehavior.onnx"
)
```

## Load Model in Unity

### 1. Copy ONNX to Unity

```bash
# Copy model to Unity Assets
cp results/MyRun/MyBehavior.onnx Project/Assets/ML-Agents/Models/
```

### 2. Assign in Unity Editor

1. Select your Agent GameObject
2. In Behavior Parameters component:
   - Set **Model** to your `.onnx` file
   - Set **Behavior Type** to "Inference Only"
3. Play the scene to test

### 3. Verify Model Works

```csharp
// In Unity, check model is loaded:
var model = GetComponent<BehaviorParameters>().Model;
if (model != null)
{
    Debug.Log("Model loaded successfully!");
}
```

## Push to HuggingFace Hub

Share your trained model on HuggingFace:

```bash
# Set HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Push model
mlagents-push-to-hf \
  --run-id=MyTraining \
  --local-dir=results/MyTraining \
  --repo-id=username/my-agent-model \
  --commit-message="Trained PPO agent on CustomEnv"
```

### HuggingFace Model Card

The push command automatically generates a model card with:
- Training configuration
- Hyperparameters
- Environment details
- Usage instructions

## Load from HuggingFace

Download and use community models:

```bash
# Download model
mlagents-load-from-hf \
  --repo-id=username/my-agent-model \
  --local-dir=./downloaded_models

# Copy to Unity
cp downloaded_models/*.onnx Project/Assets/ML-Agents/Models/
```

## Model Validation

Verify exported model works correctly:

```python
import onnx

# Load ONNX model
model = onnx.load("results/MyRun/MyBehavior.onnx")

# Check the model
onnx.checker.check_model(model)
print("Model is valid!")

# Print model info
print(f"Inputs: {[input.name for input in model.graph.input]}")
print(f"Outputs: {[output.name for output in model.graph.output]}")
```

## Model Size Optimization

Reduce model size for deployment:

```yaml
# In training config, reduce network size:
network_settings:
  hidden_units: 64   # Down from 128
  num_layers: 2      # Down from 3
```

Smaller networks:
- ✅ Faster inference
- ✅ Less memory usage
- ✅ Smaller file size
- ⚠️ May reduce learning capacity

## Troubleshooting

### ONNX Export Fails

```
ModuleNotFoundError: No module named 'onnxscript'
```

**Solution:**
```bash
# Ensure correct torch version
pip install torch<=2.8.0
```

### Model Not Loading in Unity

1. Check Unity console for errors
2. Verify model architecture matches (observation/action spaces)
3. Ensure Unity ML-Agents package version matches Python package
4. Check model file is not corrupted (re-export if needed)

### Model Produces Wrong Actions

1. Verify inference vs training mode set correctly
2. Check observation normalization matches training
3. Ensure action space configuration matches
4. Test model in Python first before Unity

## Best Practices

1. **Version Control Models**: Tag training runs with git commits
2. **Model Registry**: Organize models by date and performance
3. **Test Before Deploy**: Validate in Unity Editor before builds
4. **Document Performance**: Note reward/success rate in model card
5. **Archive Checkpoints**: Keep PyTorch checkpoints for fine-tuning

## File Structure

```
results/
└── MyRun/
    ├── MyBehavior.onnx           # ← Deploy this to Unity
    ├── MyBehavior/
    │   └── checkpoint.pt         # ← Keep for resuming training
    ├── configuration.yaml        # ← Training config reference
    ├── events.out.tfevents.*     # ← TensorBoard logs
    └── run_logs/
        └── training_status.json  # ← Training metadata
```

## HuggingFace Integration

Browse ML-Agents models:
- https://huggingface.co/models?library=ml-agents

Share your models:
1. Create HuggingFace account
2. Generate API token
3. Push model with `mlagents-push-to-hf`
4. Model card auto-generated with training details

## Related Skills

- `train-ml-agent` - Train models before export
- `debug-training` - Fix issues before export
- `optimize-performance` - Optimize model size/speed

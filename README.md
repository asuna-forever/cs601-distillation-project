# Model Information: Distilled Qwen3-8B Variant

## Overview

This model is a distilled version derived from the original **`Qwen3-8B`**. Key characteristics include:

* The attention blocks from the original `Qwen3` architecture have been reused.
* A bottleneck mechanism has been applied specifically to the MLP (Multi-Layer Perceptron) layers.

## Parameters

The parameters are too big to be uploaded to gradescope, so please visit github repo:
https://github.com/asuna-forever/cs601-distillation-project.git
for the model parameters.

## Performance Testing

To evaluate the model's performance, please use the following command:

```bash
sbatch run_validation.sh
```
**Note**: Loading the model from files may take quite a long while (about 10 minutes).
## Denpendency

Requirment is attached, but generally, any environment installed with torch>2.5, CUDA>12.0, accelerate, bitsandbytes and transformers==4.52.0 can run the validation.

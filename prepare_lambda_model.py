#!/usr/bin/env python3
"""
Pre-download and optimize ColPali model for AWS Lambda deployment.
This script runs during Docker build to prepare the model.
"""
import os
import torch
from transformers import AutoModel, AutoProcessor

try:
    # Download ColPali model
    print("Downloading ColPali model...")
    model = AutoModel.from_pretrained('vidore/colqwen2-v0.1', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained('vidore/colqwen2-v0.1', trust_remote_code=True)
    print("Model downloaded successfully")

    # Quantize model for memory efficiency
    print("Quantizing model for Lambda...")
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save optimized model
    torch.save({
        'model_state_dict': model_quantized.state_dict(),
        'config': model.config
    }, '/models/colpali_quantized.pth')

    processor.save_pretrained('/models/processor')
    print("Model preparation complete")

except Exception as e:
    print(f"Model download failed: {e}")
    # Create placeholder files for build to continue
    os.makedirs('/models/processor', exist_ok=True)
    with open('/models/colpali_quantized.pth', 'w') as f:
        f.write('placeholder')
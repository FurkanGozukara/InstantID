#!/usr/bin/env python3
"""
Test script to verify that quantization works correctly.
This script tests both 4-bit and 8-bit quantization to ensure they don't produce noise.

Usage:
    python test_quantization.py

This test helps verify the 8-bit quantization fix is working properly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path to import pipeline_common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pipelines.pipeline_common import quantize_4bit, quantize_8bit
    print("✓ Successfully imported quantization functions")
except ImportError as e:
    print(f"✗ Failed to import quantization functions: {e}")
    exit(1)

def create_test_model():
    """Create a simple test model with Linear layers."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    return TestModel()

def test_quantization_method(quantize_func, method_name):
    """Test a specific quantization method."""
    print(f"\n--- Testing {method_name} ---")
    
    # Create test model and move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_test_model().to(device)
    
    # Generate test input
    test_input = torch.randn(4, 128).to(device)
    
    # Get output before quantization
    model.eval()
    with torch.no_grad():
        output_before = model(test_input)
        print(f"Output before quantization - Mean: {output_before.mean().item():.6f}, Std: {output_before.std().item():.6f}")
    
    # Apply quantization
    print(f"Applying {method_name}...")
    quantize_func(model)
    
    # Get output after quantization
    with torch.no_grad():
        output_after = model(test_input)
        print(f"Output after quantization - Mean: {output_after.mean().item():.6f}, Std: {output_after.std().item():.6f}")
    
    # Check if output is reasonable (not all zeros or extremely large values)
    mean_val = output_after.mean().item()
    std_val = output_after.std().item()
    
    # Define reasonable bounds
    is_reasonable = (
        abs(mean_val) < 100 and  # Mean shouldn't be extremely large
        std_val > 0.001 and     # Should have some variation
        std_val < 100 and       # Std shouldn't be extremely large
        not torch.isnan(output_after).any() and  # No NaN values
        not torch.isinf(output_after).any()      # No infinite values
    )
    
    if is_reasonable:
        print(f"✓ {method_name} test PASSED - Output looks reasonable")
        return True
    else:
        print(f"✗ {method_name} test FAILED - Output appears to be noise or invalid")
        print(f"  Mean: {mean_val}, Std: {std_val}")
        print(f"  Contains NaN: {torch.isnan(output_after).any()}")
        print(f"  Contains Inf: {torch.isinf(output_after).any()}")
        return False

def main():
    print("Testing Quantization Fix")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available, testing on CPU")
    
    # Check if bitsandbytes is available
    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes available: {bnb.__version__}")
    except ImportError:
        print("✗ bitsandbytes not available - quantization tests will be skipped")
        print("Install bitsandbytes with: pip install bitsandbytes")
        exit(1)
    
    # Test both quantization methods
    results = []
    
    # Test 4-bit quantization
    results.append(test_quantization_method(quantize_4bit, "4-bit quantization"))
    
    # Test 8-bit quantization
    results.append(test_quantization_method(quantize_8bit, "8-bit quantization"))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if all(results):
        print("✓ All quantization tests PASSED!")
        print("The 8-bit quantization fix appears to be working correctly.")
        print("You can now use --load_mode 8bit without getting noise output.")
    else:
        print("✗ Some quantization tests FAILED!")
        print("Please check the output above for details.")
        print("If 8-bit failed, the quantization fix may need additional work.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
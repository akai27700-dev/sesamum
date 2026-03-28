                      
"""
PyTorchモデルをONNXに変換するスクリプト
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--input', '-i', default='data/model_best.pth', 
                       help='Input PyTorch model path')
    parser.add_argument('--output', '-o', default='data/model_best.onnx', 
                       help='Output ONNX model path')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Force overwrite existing ONNX model')
    
    args = parser.parse_args()
    
                
    if not os.path.isabs(args.input):
        args.input = os.path.abspath(args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)
    
                 
    if not os.path.exists(args.input):
        print(f"Error: Input PyTorch model not found: {args.input}")
        return 1
    
               
    if os.path.exists(args.output) and not args.force:
        print(f"Error: Output ONNX model already exists: {args.output}")
        print("Use --force to overwrite")
        return 1
    
                 
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
                   
        from core.onnx_inference import create_onnx_model_from_pytorch
        print(f"Converting {args.input} to {args.output}")
        create_onnx_model_from_pytorch(args.input, args.output)
        print("Conversion completed successfully!")
        
                  
        from core.onnx_inference import ONNXInference
        onnx_engine = ONNXInference(args.output)
        info = onnx_engine.get_model_info()
        print("\nONNX Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'engine'))
try:
    import othello_engine
    print("Engine loaded successfully.")
    
    # Run benchmark
    results = othello_engine.benchmark_optimizations(10000)
    print("\nBenchmark Results (10000 iterations):")
    print(f"TT Probe/Store: {results[0]:.6f}s")
    print(f"Legal + Flip (Scalar): {results[1]:.6f}s")
    print(f"Full Eval (Scalar): {results[2]:.6f}s")
    if results[3] > 0:
        print(f"SIMD Batch Eval: {results[3]:.6f}s")
    elif results[3] == 0:
        print("SIMD Batch Eval: SKIPPED (Too few iterations)")
    else:
        print("SIMD Batch Eval: NOT SUPPORTED (No AVX2)")

except Exception as e:
    print(f"Error: {e}")

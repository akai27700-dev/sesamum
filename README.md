# Sesamum - Advanced Othello AI Engine

A high-performance Othello AI engine with neural network evaluation, Monte Carlo Tree Search, and advanced pruning techniques.

## Project Structure

```
Sesamum_C++/
├── sesamum.py            # Main entry point
├── convert_to_onnx.py     # PyTorch to ONNX conversion script
├── engine/                # C++ engine components
│   ├── othello_engine.cpp # Core C++ engine with pybind11
│   ├── othello_engine.pyi # Type hints
│   └── othello_engine.cp313-win_amd64.pyd # Compiled binary
├── core/                  # Python core logic
│   ├── othello_core.py   # Core game logic and utilities
│   └── onnx_inference.py  # ONNX inference engine
├── gui/                   # User interface
│   ├── othello_gui.py    # Main GUI application
│   ├── othello_gui_dialogs.py # Settings dialogs
│   └── othello_gui_search.py  # Search functionality
├── scripts/               # Utility scripts
│   ├── setup.py          # Build configuration
│   ├── benchmark_battle.py # Benchmarking tool
│   └── init_venv.ps1     # Virtual environment setup
├── data/                  # Data files
│   ├── best_weights.json # Evaluation weights
│   ├── model_best.pth    # PyTorch neural network model
│   ├── model_best.onnx   # ONNX neural network model (auto-generated)
│   ├── opening_book.json # Opening book
│   └── egaroucid_hash/   # Hash tables
└── backup/               # Backup files
```

## Features

- **Hybrid Search**: Alpha-beta pruning + MCTS with neural network evaluation
- **Advanced Pruning**: MPC, Null Move Pruning, Late Move Reductions
- **Optimized Bitboard Operations**: AVX2 SIMD support
- **Transposition Table**: Large hash table with replacement strategies
- **Neural Network**: ResNet-based position evaluation with ONNX acceleration
- **Opening Book**: Extensive opening book integration
- **Parallel Search**: Multi-threaded search with dynamic load balancing
- **Board-Only Mode**: Minimalist interface for focused gameplay

## Installation

1. Set up virtual environment:
   ```powershell
   .\scripts\init_venv.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build C++ engine:
   ```bash
   cd scripts
   python setup.py build_ext --inplace
   ```

## Usage

Run the application:
```bash
python sesamum.py
```

## Settings

The GUI supports various configuration options:
- **Time Control**: Fixed time settings (0.5s, 1s, 5s, 10s, 30s)
- **MCTS Influence**: Adjust MCTS vs Alpha-beta balance (0-100%)
- **Book Usage**: Opening book usage percentage
- **Engine Options**: C++ engine, Neural Network, Pondering

## Performance

- **Search Depth**: 18-20 plies reachable
- **Nodes per Second**: 1M+ NPS on modern hardware
- **Memory Usage**: ~500MB with default settings
- **CPU Usage**: Efficient multi-threading with hyper-threading support

## Requirements

- Python 3.8+
- C++ compiler with AVX2 support
- PyTorch
- NumPy
- Tkinter

import tkinter as tk
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from core.othello_core import print_system_info
    print_system_info()
except ImportError:
    print('Failed to import system info display', flush=True)
try:
    from core.othello_core import initialize_onnx_engine
    initialize_onnx_engine()
except ImportError:
    print('Failed to import ONNX initialization, using PyTorch fallback', flush=True)
from gui.othello_gui import UltimateOthello
if __name__ == '__main__':
    r = tk.Tk()
    app = UltimateOthello(r)
    r.mainloop()
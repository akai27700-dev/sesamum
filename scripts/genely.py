import tkinter as tk
import sys
import os

                                           
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.othello_gui import UltimateOthello

if __name__ == "__main__":
    r = tk.Tk()
    app = UltimateOthello(r)
    r.mainloop()

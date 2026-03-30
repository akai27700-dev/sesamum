import os, threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
from numba import njit, prange, set_num_threads

set_num_threads(6)

_FULL_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

@njit(cache=True, fastmath=True, nogil=True)
def count_bits(x):
    c = np.int64(0)
    while x:
        c += np.int64(1)
        x &= x - np.uint64(1)
    return c

@njit(cache=True, fastmath=True, nogil=True)
def get_legal_moves(P, O):
    m, e, l = np.uint64(0x7E7E7E7E7E7E7E7E), ~(P | O) & _FULL_MASK, np.uint64(0)
    t = (P << np.uint64(1)) & O & m
    t |= (t << np.uint64(1)) & O & m; t |= (t << np.uint64(1)) & O & m; t |= (t << np.uint64(1)) & O & m; t |= (t << np.uint64(1)) & O & m; l |= (t << np.uint64(1)) & e
    t = (P >> np.uint64(1)) & O & m
    t |= (t >> np.uint64(1)) & O & m; t |= (t >> np.uint64(1)) & O & m; t |= (t >> np.uint64(1)) & O & m; t |= (t >> np.uint64(1)) & O & m; l |= (t >> np.uint64(1)) & e
    t = (P << np.uint64(8)) & O
    t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O; l |= (t << np.uint64(8)) & e
    t = (P >> np.uint64(8)) & O
    t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O; l |= (t >> np.uint64(8)) & e
    t = (P << np.uint64(7)) & O & m
    t |= (t << np.uint64(7)) & O & m; t |= (t << np.uint64(7)) & O & m; t |= (t << np.uint64(7)) & O & m; t |= (t << np.uint64(7)) & O & m; l |= (t << np.uint64(7)) & e
    t = (P >> np.uint64(7)) & O & m
    t |= (t >> np.uint64(7)) & O & m; t |= (t >> np.uint64(7)) & O & m; t |= (t >> np.uint64(7)) & O & m; t |= (t >> np.uint64(7)) & O & m; l |= (t >> np.uint64(7)) & e
    t = (P << np.uint64(9)) & O & m
    t |= (t << np.uint64(9)) & O & m; t |= (t << np.uint64(9)) & O & m; t |= (t << np.uint64(9)) & O & m; t |= (t << np.uint64(9)) & O & m; l |= (t << np.uint64(9)) & e
    t = (P >> np.uint64(9)) & O & m
    t |= (t >> np.uint64(9)) & O & m; t |= (t >> np.uint64(9)) & O & m; t |= (t >> np.uint64(9)) & O & m; t |= (t >> np.uint64(9)) & O & m; l |= (t >> np.uint64(9)) & e
    return l & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def get_flip(P, O, idx):
    m, mk, f = np.uint64(1) << np.uint64(idx), np.uint64(0x7E7E7E7E7E7E7E7E), np.uint64(0)
    t = (m << np.uint64(1)) & O & mk
    t |= (t << np.uint64(1)) & O & mk; t |= (t << np.uint64(1)) & O & mk; t |= (t << np.uint64(1)) & O & mk; t |= (t << np.uint64(1)) & O & mk
    if (t << np.uint64(1)) & P: f |= t
    t = (m >> np.uint64(1)) & O & mk
    t |= (t >> np.uint64(1)) & O & mk; t |= (t >> np.uint64(1)) & O & mk; t |= (t >> np.uint64(1)) & O & mk; t |= (t >> np.uint64(1)) & O & mk
    if (t >> np.uint64(1)) & P: f |= t
    t = (m << np.uint64(8)) & O
    t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O; t |= (t << np.uint64(8)) & O
    if (t << np.uint64(8)) & P: f |= t
    t = (m >> np.uint64(8)) & O
    t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O; t |= (t >> np.uint64(8)) & O
    if (t >> np.uint64(8)) & P: f |= t
    t = (m << np.uint64(7)) & O & mk
    t |= (t << np.uint64(7)) & O & mk; t |= (t << np.uint64(7)) & O & mk; t |= (t << np.uint64(7)) & O & mk; t |= (t << np.uint64(7)) & O & mk
    if (t << np.uint64(7)) & P: f |= t
    t = (m >> np.uint64(7)) & O & mk
    t |= (t >> np.uint64(7)) & O & mk; t |= (t >> np.uint64(7)) & O & mk; t |= (t >> np.uint64(7)) & O & mk; t |= (t >> np.uint64(7)) & O & mk
    if (t >> np.uint64(7)) & P: f |= t
    t = (m << np.uint64(9)) & O & mk
    t |= (t << np.uint64(9)) & O & mk; t |= (t << np.uint64(9)) & O & mk; t |= (t << np.uint64(9)) & O & mk; t |= (t << np.uint64(9)) & O & mk
    if (t << np.uint64(9)) & P: f |= t
    t = (m >> np.uint64(9)) & O & mk
    t |= (t >> np.uint64(9)) & O & mk; t |= (t >> np.uint64(9)) & O & mk; t |= (t >> np.uint64(9)) & O & mk; t |= (t >> np.uint64(9)) & O & mk
    if (t >> np.uint64(9)) & P: f |= t
    return f & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def evaluate_board(P, O):
    my_moves = count_bits(get_legal_moves(P, O))
    opp_moves = count_bits(get_legal_moves(O, P))
    c1 = np.uint64(0x8100000000000081)
    c2 = np.uint64(0x0042000000004200)
    c3 = np.uint64(0x4281000000008142)
    c4 = np.uint64(0x3c0081818181003c)
    score = (my_moves - opp_moves) * 15.0
    score += (count_bits(P & c1) - count_bits(O & c1)) * 1000.0
    score -= (count_bits(P & c2) - count_bits(O & c2)) * 300.0
    score -= (count_bits(P & c3) - count_bits(O & c3)) * 100.0
    score += (count_bits(P & c4) - count_bits(O & c4)) * 30.0
    return score

@njit(cache=True, fastmath=True, nogil=True)
def exact_eval(P, O):
    return np.float64((count_bits(P) - count_bits(O)) * 10000.0)

@njit(cache=True, fastmath=True, nogil=True)
def alphabeta(P, O, depth, alpha, beta, passed, is_exact):
    valid = get_legal_moves(P, O)
    if not valid:
        if passed:
            return exact_eval(P, O)
        return -alphabeta(O, P, depth, -beta, -alpha, True, is_exact)
    
    if depth == 0:
        return evaluate_board(P, O)

    max_val = -1e18
    v_temp = valid
    while v_temp:
        bit = v_temp & (~v_temp + np.uint64(1))
        v_temp ^= bit
        idx = count_bits(bit - np.uint64(1))
        f = get_flip(P, O, idx)
        val = -alphabeta(O ^ f, P | bit | f, depth - 1, -beta, -alpha, False, is_exact)
        if val > max_val:
            max_val = val
        if max_val > alpha:
            alpha = max_val
        if alpha >= beta:
            break
    return max_val

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def search_root(P, O, depth, is_exact):
    valid = get_legal_moves(P, O)
    if not valid: return -1

    indices = np.zeros(64, dtype=np.int64)
    count = 0
    v_temp = valid
    while v_temp:
        bit = v_temp & (~v_temp + np.uint64(1))
        v_temp ^= bit
        indices[count] = count_bits(bit - np.uint64(1))
        count += 1

    vals = np.zeros(count, dtype=np.float64)
    
    for i in prange(count):
        idx = indices[i]
        bit = np.uint64(1) << np.uint64(idx)
        f = get_flip(P, O, idx)
        vals[i] = -alphabeta(O ^ f, P | bit | f, depth - 1, -1e18, 1e18, False, is_exact)

    best_a = -1
    max_val = -1e18
    for i in range(count):
        if vals[i] > max_val:
            max_val = vals[i]
            best_a = indices[i]

    return best_a

class SimpleStrongOthello:
    def __init__(self, r):
        self.rt = r
        self.rt.title("Simple Strong AI")
        self.rt.protocol("WM_DELETE_WINDOW", self.on_close)
        self.B, self.W = np.uint64(0x0000000810000000), np.uint64(0x0000001008000000)
        self.tn = 1
        self.hc = -1 
        self.ac = 1
        self.cv = tk.Canvas(r, width=480, height=480, bg="#2e7d32")
        self.cv.pack(pady=10)
        self.cv.bind("<Button-1>", self.clk)
        self.is_thinking = False
        self.running = True
        self.drw()
        self.chk()

    def on_close(self):
        self.running = False
        self.rt.quit()
        self.rt.destroy()

    def mc(self):
        return int(self.B | self.W).bit_count()

    def drw(self):
        if not self.running: return
        self.cv.delete("all")
        for r in range(8):
            for c in range(8):
                x, y, i = c * 60, r * 60, np.uint64(r * 8 + c)
                self.cv.create_rectangle(x, y, x + 60, y + 60, outline="#1b5e20")
                if (self.B >> i) & np.uint64(1):
                    self.cv.create_oval(x + 6, y + 6, x + 54, y + 54, fill="black")
                elif (self.W >> i) & np.uint64(1):
                    self.cv.create_oval(x + 6, y + 6, x + 54, y + 54, fill="white", outline="#ccc")
        if self.tn == self.hc:
            mB, mW = self.B if self.hc == 1 else self.W, self.W if self.hc == 1 else self.B
            tm = int(get_legal_moves(mB, mW))
            while tm:
                ix, t = 0, tm & -tm
                while t > 1:
                    t >>= 1
                    ix += 1
                self.cv.create_oval((ix % 8) * 60 + 25, (ix // 8) * 60 + 25, (ix % 8) * 60 + 35, (ix // 8) * 60 + 35, fill="#4caf50", outline="")
                tm &= tm - 1

    def clk(self, e):
        if not self.running or self.tn != self.hc: return
        ix = np.uint64((e.y // 60) * 8 + (e.x // 60))
        mB, mW = self.B if self.hc == 1 else self.W, self.W if self.hc == 1 else self.B
        if (int(get_legal_moves(mB, mW)) >> int(ix)) & 1:
            f = np.uint64(get_flip(mB, mW, np.int64(ix)))
            if self.hc == 1:
                self.B, self.W = (self.B | (np.uint64(1) << np.uint64(ix)) | f) & _FULL_MASK, (self.W ^ f) & _FULL_MASK
            else:
                self.W, self.B = (self.W | (np.uint64(1) << np.uint64(ix)) | f) & _FULL_MASK, (self.B ^ f) & _FULL_MASK
            self.tn = self.ac
            self.drw()
            self.rt.after(100, self.chk)

    def ai_r(self):
        try:
            if not self.running: return
            empty = 64 - self.mc()
            aB, oB = self.B if self.ac == 1 else self.W, self.W if self.ac == 1 else self.B
            
            if empty <= 18:
                bm = search_root(aB, oB, empty, True)
            else:
                bm = search_root(aB, oB, 10, False)

            if not self.running: return
            f = np.uint64(get_flip(aB, oB, np.int64(bm)))
            if self.ac == 1:
                self.B, self.W = (self.B | (np.uint64(1) << np.uint64(bm)) | f) & _FULL_MASK, (self.W ^ f) & _FULL_MASK
            else:
                self.W, self.B = (self.W | (np.uint64(1) << np.uint64(bm)) | f) & _FULL_MASK, (self.B ^ f) & _FULL_MASK
            self.tn = self.hc
            self.rt.after(0, self.drw)
            self.rt.after(0, self.chk)
        finally:
            self.is_thinking = False

    def chk(self):
        if not self.running: return
        bM = get_legal_moves(self.B, self.W)
        wM = get_legal_moves(self.W, self.B)
        if not bM and not wM:
            b, w = int(self.B).bit_count(), int(self.W).bit_count()
            msg = "勝ち" if (self.hc == 1 and b > w) or (self.hc == -1 and w > b) else ("負け" if (self.hc == 1 and b < w) or (self.hc == -1 and w < b) else "引分")
            messagebox.showinfo("終", f"黒: {b}  白: {w}\n{msg}")
            return
        if not get_legal_moves(self.B if self.tn == 1 else self.W, self.W if self.tn == 1 else self.B):
            self.tn = -self.tn
            self.rt.after(100, self.chk)
            return
        if self.tn == self.ac:
            if not self.is_thinking:
                self.is_thinking = True
                threading.Thread(target=self.ai_r, daemon=True).start()
        else:
            self.drw()

if __name__ == "__main__":
    r = tk.Tk()
    app = SimpleStrongOthello(r)
    r.mainloop()
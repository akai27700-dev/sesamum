import os
import time
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
import math
import random

try:
    import cupy as cp
    import numpy as np
    CUDA_AVAILABLE = True
    print(f"[CUDA] CuPy検出: runtime={cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    import numpy as np
    cp = np
    CUDA_AVAILABLE = False
    print("[CUDA] CuPy未検出 → CPUフォールバック")

_FULL_MASK = 0xFFFFFFFFFFFFFFFF

def popcount(x):
    return bin(x & _FULL_MASK).count("1")

def _mask64(x):
    return x & _FULL_MASK

def get_legal_moves(P, O):
    mask = 0x7E7E7E7E7E7E7E7E
    empty = _mask64(~(P | O))
    legal = 0
    t = _mask64((P << 1) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t << 1) & O & mask))
    legal |= _mask64((t << 1) & empty)
    t = _mask64((P >> 1) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t >> 1) & O & mask))
    legal |= _mask64((t >> 1) & empty)
    t = _mask64((P << 8) & O)
    for _ in range(5): t = _mask64(t | _mask64((t << 8) & O))
    legal |= _mask64((t << 8) & empty)
    t = _mask64((P >> 8) & O)
    for _ in range(5): t = _mask64(t | _mask64((t >> 8) & O))
    legal |= _mask64((t >> 8) & empty)
    t = _mask64((P << 7) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t << 7) & O & mask))
    legal |= _mask64((t << 7) & empty)
    t = _mask64((P >> 7) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t >> 7) & O & mask))
    legal |= _mask64((t >> 7) & empty)
    t = _mask64((P << 9) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t << 9) & O & mask))
    legal |= _mask64((t << 9) & empty)
    t = _mask64((P >> 9) & O & mask)
    for _ in range(5): t = _mask64(t | _mask64((t >> 9) & O & mask))
    legal |= _mask64((t >> 9) & empty)
    return legal & _FULL_MASK

def get_flip(P, O, move_idx):
    m = _mask64(1 << move_idx)
    mask = 0x7E7E7E7E7E7E7E7E
    flip = 0
    tmp = _mask64((m << 1) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp << 1) & O & mask))
    if _mask64((tmp << 1) & P): flip |= tmp
    tmp = _mask64((m >> 1) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp >> 1) & O & mask))
    if _mask64((tmp >> 1) & P): flip |= tmp
    tmp = _mask64((m << 8) & O)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp << 8) & O))
    if _mask64((tmp << 8) & P): flip |= tmp
    tmp = _mask64((m >> 8) & O)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp >> 8) & O))
    if _mask64((tmp >> 8) & P): flip |= tmp
    tmp = _mask64((m << 7) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp << 7) & O & mask))
    if _mask64((tmp << 7) & P): flip |= tmp
    tmp = _mask64((m >> 7) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp >> 7) & O & mask))
    if _mask64((tmp >> 7) & P): flip |= tmp
    tmp = _mask64((m << 9) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp << 9) & O & mask))
    if _mask64((tmp << 9) & P): flip |= tmp
    tmp = _mask64((m >> 9) & O & mask)
    for _ in range(5): tmp = _mask64(tmp | _mask64((tmp >> 9) & O & mask))
    if _mask64((tmp >> 9) & P): flip |= tmp
    return flip & _FULL_MASK

def idx_to_coord(idx):
    return (idx // 8 + 1, idx % 8 + 1)

# =====================================================================
# CUDAカーネル: curand不使用・stdint不使用（NVRTC互換）
# プレイアウトのみGPUで実行し、MCTSツリーはCPU側で管理
# =====================================================================
CUDA_KERNEL_CODE = r"""
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

__device__ __forceinline__ uint32_t xorshift_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (uint32_t)(x >> 32);
}

__device__ __forceinline__ uint64_t legal_moves_dev(uint64_t P, uint64_t O) {
    const uint64_t mask = 0x7E7E7E7E7E7E7EULL;
    uint64_t empty = ~(P | O);
    uint64_t legal = 0ULL, t;
    t=(P<<1)&O&mask; t|=(t<<1)&O&mask; t|=(t<<1)&O&mask; t|=(t<<1)&O&mask; t|=(t<<1)&O&mask; t|=(t<<1)&O&mask; legal|=(t<<1)&empty;
    t=(P>>1)&O&mask; t|=(t>>1)&O&mask; t|=(t>>1)&O&mask; t|=(t>>1)&O&mask; t|=(t>>1)&O&mask; t|=(t>>1)&O&mask; legal|=(t>>1)&empty;
    t=(P<<8)&O;      t|=(t<<8)&O;      t|=(t<<8)&O;      t|=(t<<8)&O;      t|=(t<<8)&O;      t|=(t<<8)&O;      legal|=(t<<8)&empty;
    t=(P>>8)&O;      t|=(t>>8)&O;      t|=(t>>8)&O;      t|=(t>>8)&O;      t|=(t>>8)&O;      t|=(t>>8)&O;      legal|=(t>>8)&empty;
    t=(P<<7)&O&mask; t|=(t<<7)&O&mask; t|=(t<<7)&O&mask; t|=(t<<7)&O&mask; t|=(t<<7)&O&mask; t|=(t<<7)&O&mask; legal|=(t<<7)&empty;
    t=(P>>7)&O&mask; t|=(t>>7)&O&mask; t|=(t>>7)&O&mask; t|=(t>>7)&O&mask; t|=(t>>7)&O&mask; t|=(t>>7)&O&mask; legal|=(t>>7)&empty;
    t=(P<<9)&O&mask; t|=(t<<9)&O&mask; t|=(t<<9)&O&mask; t|=(t<<9)&O&mask; t|=(t<<9)&O&mask; t|=(t<<9)&O&mask; legal|=(t<<9)&empty;
    t=(P>>9)&O&mask; t|=(t>>9)&O&mask; t|=(t>>9)&O&mask; t|=(t>>9)&O&mask; t|=(t>>9)&O&mask; t|=(t>>9)&O&mask; legal|=(t>>9)&empty;
    return legal;
}

__device__ __forceinline__ uint64_t do_flip_dev(uint64_t P, uint64_t O, int idx) {
    uint64_t m = 1ULL << idx;
    const uint64_t mask = 0x7E7E7E7E7E7E7EULL;
    uint64_t flip = 0ULL, tmp;
    tmp=(m<<1)&O&mask; tmp|=(tmp<<1)&O&mask; tmp|=(tmp<<1)&O&mask; tmp|=(tmp<<1)&O&mask; tmp|=(tmp<<1)&O&mask; tmp|=(tmp<<1)&O&mask; if((tmp<<1)&P) flip|=tmp;
    tmp=(m>>1)&O&mask; tmp|=(tmp>>1)&O&mask; tmp|=(tmp>>1)&O&mask; tmp|=(tmp>>1)&O&mask; tmp|=(tmp>>1)&O&mask; tmp|=(tmp>>1)&O&mask; if((tmp>>1)&P) flip|=tmp;
    tmp=(m<<8)&O;      tmp|=(tmp<<8)&O;      tmp|=(tmp<<8)&O;      tmp|=(tmp<<8)&O;      tmp|=(tmp<<8)&O;      tmp|=(tmp<<8)&O;      if((tmp<<8)&P) flip|=tmp;
    tmp=(m>>8)&O;      tmp|=(tmp>>8)&O;      tmp|=(tmp>>8)&O;      tmp|=(tmp>>8)&O;      tmp|=(tmp>>8)&O;      tmp|=(tmp>>8)&O;      if((tmp>>8)&P) flip|=tmp;
    tmp=(m<<7)&O&mask; tmp|=(tmp<<7)&O&mask; tmp|=(tmp<<7)&O&mask; tmp|=(tmp<<7)&O&mask; tmp|=(tmp<<7)&O&mask; tmp|=(tmp<<7)&O&mask; if((tmp<<7)&P) flip|=tmp;
    tmp=(m>>7)&O&mask; tmp|=(tmp>>7)&O&mask; tmp|=(tmp>>7)&O&mask; tmp|=(tmp>>7)&O&mask; tmp|=(tmp>>7)&O&mask; tmp|=(tmp>>7)&O&mask; if((tmp>>7)&P) flip|=tmp;
    tmp=(m<<9)&O&mask; tmp|=(tmp<<9)&O&mask; tmp|=(tmp<<9)&O&mask; tmp|=(tmp<<9)&O&mask; tmp|=(tmp<<9)&O&mask; tmp|=(tmp<<9)&O&mask; if((tmp<<9)&P) flip|=tmp;
    tmp=(m>>9)&O&mask; tmp|=(tmp>>9)&O&mask; tmp|=(tmp>>9)&O&mask; tmp|=(tmp>>9)&O&mask; tmp|=(tmp>>9)&O&mask; tmp|=(tmp>>9)&O&mask; if((tmp>>9)&P) flip|=tmp;
    return flip;
}

// results[i]: 1.0=AI勝ち, 0.0=AI負け, 0.5=引き分け
extern "C" __global__
void playout_kernel(
    uint64_t start_P,
    uint64_t start_O,
    int n_sims,
    float* results,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sims) return;

    // XorShift64でスレッドごとに独立したRNG
    uint64_t rng = seed ^ ((uint64_t)tid * 6364136223846793005ULL + 1442695040888963407ULL);
    xorshift_next(&rng);
    xorshift_next(&rng);

    uint64_t P = start_P;  // 現在のプレイヤー（AIとして開始）
    uint64_t O = start_O;
    int is_ai = 1;         // 現在のターンがAIかどうか
    int pass_count = 0;

    while (1) {
        uint64_t moves = legal_moves_dev(P, O);
        if (!moves) {
            if (++pass_count >= 2) break;  // 両者パスで終局
            // スワップ（パス）
            uint64_t tmp = P; P = O; O = tmp;
            is_ai ^= 1;
            continue;
        }
        pass_count = 0;

        // 合法手リストを構築
        int ml[64];
        int nm = 0;
        uint64_t mv = moves;
        while (mv) {
            ml[nm++] = __ffsll((long long)mv) - 1;
            mv &= mv - 1ULL;
        }

        // ランダムに手を選択
        int move = ml[(int)(xorshift_next(&rng) % (uint32_t)nm)];
        uint64_t f = do_flip_dev(P, O, move);
        uint64_t nP = P | (1ULL << move) | f;
        uint64_t nO = O & ~f;
        // 次のターンへ（盤面をスワップ）
        P = nO; O = nP;
        is_ai ^= 1;
    }

    // 終局: is_ai==1ならPがAI側の石
    uint64_t ai_bb  = is_ai ? P : O;
    uint64_t opp_bb = is_ai ? O : P;
    int ac = __popcll(ai_bb);
    int oc = __popcll(opp_bb);
    results[tid] = (ac > oc) ? 1.0f : (ac < oc) ? 0.0f : 0.5f;
}
"""

# =====================================================================
# GPUエンジン: プレイアウトのみGPU実行
# =====================================================================
class GPUEngine:
    def __init__(self):
        self.kernel = None
        self._results_buf = None
        self._buf_size = 0

        if not CUDA_AVAILABLE:
            print("[GPU] CuPy利用不可 → CPUフォールバック")
            return

        # アーキテクチャを試す順番（新しいものから）
        archs = ['sm_120', 'sm_100', 'sm_89', 'sm_86', 'sm_80', 'sm_75', 'sm_70']
        for arch in archs:
            try:
                k = cp.RawKernel(
                    CUDA_KERNEL_CODE,
                    'playout_kernel',
                    options=(f'--std=c++17', f'-arch={arch}'),
                    backend='nvrtc',
                )
                # 動作確認用ダミー実行
                dummy = cp.zeros(256, dtype=cp.float32)
                k((1,), (256,), (
                    np.uint64(0x0000000810000000),
                    np.uint64(0x0000001008000000),
                    np.int32(256),
                    dummy,
                    np.uint64(12345),
                ))
                cp.cuda.Stream.null.synchronize()
                self.kernel = k
                print(f"[GPU] コンパイル成功 ({arch})")
                break
            except Exception as e:
                print(f"[GPU] {arch} 失敗: {e}")

        if self.kernel is None:
            print("[GPU] 全アーキテクチャ失敗 → CPUフォールバック")

    def _ensure_buf(self, n):
        if n > self._buf_size:
            self._results_buf = cp.empty(n, dtype=cp.float32)
            self._buf_size = n

    def run(self, ai_bb, opp_bb, n_sims):
        """n_sims回のプレイアウトを実行し、AIの平均勝率を返す"""
        if self.kernel is None:
            return self._cpu_run(ai_bb, opp_bb, n_sims)

        self._ensure_buf(n_sims)
        seed = np.uint64(random.getrandbits(63) + 1)
        threads = 256
        blocks = (n_sims + threads - 1) // threads
        self.kernel(
            (blocks,), (threads,),
            (
                np.uint64(ai_bb),
                np.uint64(opp_bb),
                np.int32(n_sims),
                self._results_buf[:n_sims],
                seed,
            )
        )
        cp.cuda.Stream.null.synchronize()
        return float(cp.mean(self._results_buf[:n_sims]))

    def _cpu_run(self, ai_bb, opp_bb, n_sims):
        """GPUなし時のCPUフォールバック"""
        total = 0.0
        for _ in range(n_sims):
            P, O, is_ai, pc = ai_bb, opp_bb, True, 0
            while True:
                moves = get_legal_moves(P, O)
                if not moves:
                    pc += 1
                    if pc >= 2: break
                    P, O = O, P
                    is_ai = not is_ai
                    continue
                pc = 0
                ml = []
                t = moves
                while t:
                    ml.append((t & -t).bit_length() - 1)
                    t &= t - 1
                m = random.choice(ml)
                f = get_flip(P, O, m)
                nP = _mask64(P | (1 << m) | f)
                nO = _mask64(O & ~f)
                P, O = nO, nP
                is_ai = not is_ai
            ab = P if is_ai else O
            ob = O if is_ai else P
            ac, oc = popcount(ab), popcount(ob)
            total += 1.0 if ac > oc else 0.0 if ac < oc else 0.5
        return total / max(n_sims, 1)


# =====================================================================
# MCTSノード（CPU版と同一ロジック）
# =====================================================================
class MCTSNode:
    __slots__ = ['move', 'parent', 'P', 'O', 'is_ai', 'visits', 'wins', 'children', 'untried']

    def __init__(self, move, parent, P, O, is_ai):
        self.move = move
        self.parent = parent
        self.P = P
        self.O = O
        self.is_ai = is_ai
        self.visits = 0
        self.wins = 0.0
        self.children = []
        self.untried = []

        temp = get_legal_moves(P, O)
        while temp:
            idx = (temp & -temp).bit_length() - 1
            self.untried.append(idx)
            temp &= temp - 1

    def uct_select(self):
        best = None
        best_val = -1.0
        log_visits = math.log(self.visits)
        for c in self.children:
            if c.visits == 0:
                return c
            val = (c.wins / c.visits) + 1.414 * math.sqrt(log_visits / c.visits)
            if val > best_val:
                best_val = val
                best = c
        return best


# =====================================================================
# GPU-MCTSエンジン
# CPU版と同一のMCTSツリー構造 + プレイアウトのみGPUで大量実行
#
# バッチサイズ（N_PLAYOUT）を大きくするほどGPUの恩恵が大きいが、
# ツリーの更新頻度が下がる。1ノードにつきN_PLAYOUTを費やす設計。
# =====================================================================
N_PLAYOUT =  1 << 14 # ノード1つあたりのGPUプレイアウト数
time_limit = 0.1

class CUDAMCTSEngine:
    def __init__(self):
        self.gpu = GPUEngine()
        backend = "CUDA GPU" if self.gpu.kernel else "CPU fallback"
        print(f"[MCTS] バックエンド: {backend}  ノードあたり{N_PLAYOUT}プレイアウト")

    def search(self, ai_bb, opp_bb, time_limit):
        root = MCTSNode(None, None, ai_bb, opp_bb, True)
        if not root.untried:
            return None, 0.0, 0

        end_time = time.time() + time_limit
        total_sims = 0

        while time.time() < end_time:
            # ---- Select ----
            node = root
            while not node.untried and node.children:
                node = node.uct_select()

            # ---- Expand ----
            if node.untried:
                m = random.choice(node.untried)
                node.untried.remove(m)
                f = get_flip(node.P, node.O, m)
                nP = _mask64(node.P | (1 << m) | f)
                nO = _mask64(node.O & ~f)
                # 展開後は相手のターン（CPU版と同じ）
                child = MCTSNode(m, node, nO, nP, not node.is_ai)
                node.children.append(child)
                node = child

            # ---- Simulate (GPU大量プレイアウト) ----
            # node.is_ai=Trueならnode.PがAI, FalseならOがAI
            if node.is_ai:
                pP, pO = node.P, node.O
            else:
                pP, pO = node.O, node.P   # AIの視点に揃える

            win_rate = self.gpu.run(pP, pO, N_PLAYOUT)
            total_sims += N_PLAYOUT

            # ---- Backpropagate (CPU版と同一ロジック) ----
            # res = AIの勝率（0〜1）
            curr = node
            while curr is not None:
                curr.visits += N_PLAYOUT
                if curr.is_ai:
                    # このノードはAIが指した後の局面
                    # → CPU版: curr.wins += (1.0 - res) と同義
                    #   （子ノードのスコア＝相手の得点として記録するCPU版の流儀）
                    curr.wins += (1.0 - win_rate) * N_PLAYOUT
                else:
                    curr.wins += win_rate * N_PLAYOUT
                curr = curr.parent

        # ---- 最善手選択（CPU版と同じく勝率最大）----
        best_move = None
        best_win_rate = -1.0
        best_visits = -1
        rows = []

        for c in root.children:
            if c.visits > 0:
                # root.is_ai=True → c.is_ai=False → c.winsはAIの勝数
                wr = c.wins / c.visits
                rows.append((c.move, wr, c.visits))
                # 最多訪問数の手を最終選択（UCT探索済みのロバストな基準）
                if c.visits > best_visits:
                    best_visits = c.visits
                    best_move = c.move
                    best_win_rate = wr

        rows.sort(key=lambda x: x[2], reverse=True)
        for mv, rate, vis in rows:
            print(f"  手 {idx_to_coord(mv)}: 勝率 {rate*100:.1f}% ({vis:,} ノード)")

        return best_move, best_win_rate, total_sims


# =====================================================================
# GUI（CPU版ベースに統計表示を追加）
# =====================================================================
class OthelloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello GPU-MCTS")
        self.black_bb = 0x0000000810000000
        self.white_bb = 0x0000001008000000

        choice = simpledialog.askstring(
            "石の選択",
            "色を選んでください (black / white):",
            initialvalue="black"
        )
        self.human_color = 1 if (choice and choice.lower().startswith("b")) else -1
        self.ai_color = -self.human_color
        self.current_turn = 1
        self.engine = None

        self.canvas = tk.Canvas(root, width=480, height=480, bg="#2e7d32")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = tk.Label(root, text="エンジン初期化中...", font=("Consolas", 12))
        self.status.pack(pady=4)
        self.info = tk.Label(root, text="", font=("Consolas", 10), fg="#555")
        self.info.pack()

        self.draw()
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self):
        self.engine = CUDAMCTSEngine()
        self.root.after(0, lambda: self.status.config(text="あなたのターン"))
        self.root.after(0, self.check_turn)

    def draw(self):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                x, y = c * 60, r * 60
                self.canvas.create_rectangle(x, y, x+60, y+60, outline="#1b5e20")
                idx = r * 8 + c
                if (self.black_bb >> idx) & 1:
                    self.canvas.create_oval(x+6, y+6, x+54, y+54, fill="black")
                elif (self.white_bb >> idx) & 1:
                    self.canvas.create_oval(x+6, y+6, x+54, y+54, fill="white", outline="#ccc")

        if self.current_turn == self.human_color:
            my_bb = self.black_bb if self.human_color == 1 else self.white_bb
            op_bb = self.white_bb if self.human_color == 1 else self.black_bb
            moves = get_legal_moves(my_bb, op_bb)
            while moves:
                idx = (moves & -moves).bit_length() - 1
                x, y = (idx % 8) * 60, (idx // 8) * 60
                self.canvas.create_oval(x+25, y+25, x+35, y+35, fill="#4caf50", outline="")
                moves &= moves - 1

        b, w = popcount(self.black_bb), popcount(self.white_bb)
        self.root.title(f"Othello GPU-MCTS  ⚫{b} - ⚪{w}")

    def on_click(self, e):
        if self.current_turn != self.human_color or self.engine is None:
            return
        idx = (e.y // 60) * 8 + (e.x // 60)
        my_bb = self.black_bb if self.human_color == 1 else self.white_bb
        op_bb = self.white_bb if self.human_color == 1 else self.black_bb
        if (get_legal_moves(my_bb, op_bb) >> idx) & 1:
            f = get_flip(my_bb, op_bb, idx)
            if self.human_color == 1:
                self.black_bb = _mask64(self.black_bb | (1 << idx) | f)
                self.white_bb = _mask64(self.white_bb & ~f)
            else:
                self.white_bb = _mask64(self.white_bb | (1 << idx) | f)
                self.black_bb = _mask64(self.black_bb & ~f)
            self.current_turn = self.ai_color
            self.draw()
            self.root.after(100, self.check_turn)

    def ai_routine(self):
        color_str = "黒" if self.ai_color == 1 else "白"
        ply = popcount(self.black_bb | self.white_bb) - 4
        print(f"\n=== AI ({color_str}) {int(ply)}手目 ===")
        self.root.after(0, lambda: self.status.config(text="AI 思考中..."))

        ai_bb  = self.black_bb if self.ai_color == 1 else self.white_bb
        opp_bb = self.white_bb if self.ai_color == 1 else self.black_bb

        t0 = time.time()
        best, wr, total = self.engine.search(ai_bb, opp_bb, time_limit)
        elapsed = time.time() - t0

        if best is None:
            mv = get_legal_moves(ai_bb, opp_bb)
            best = (mv & -mv).bit_length() - 1
            wr = 0.5

        sps = total / elapsed if elapsed > 0 else 0
        print(f"=== 決定: {idx_to_coord(best)} | 勝率 {wr*100:.1f}% | {total//1000}K sims | {sps/1e6:.2f}M/s | {elapsed:.2f}s ===")

        self.root.after(0, lambda: self.info.config(
            text=f"手:{idx_to_coord(best)}  勝率:{wr*100:.1f}%  シミュ:{total//1000}K  速度:{sps/1e6:.2f}M/s"
        ))
        self.root.after(0, lambda: self.root.title(f"GPU-MCTS  勝率: {wr*100:.1f}%"))

        f = get_flip(ai_bb, opp_bb, best)
        if self.ai_color == 1:
            self.black_bb = _mask64(self.black_bb | (1 << best) | f)
            self.white_bb = _mask64(self.white_bb & ~f)
        else:
            self.white_bb = _mask64(self.white_bb | (1 << best) | f)
            self.black_bb = _mask64(self.black_bb & ~f)

        self.current_turn = self.human_color
        self.root.after(0, self.draw)
        self.root.after(0, lambda: self.status.config(text="Your turn"))
        self.root.after(0, self.check_turn)

    def check_turn(self):
        if self.engine is None:
            self.root.after(200, self.check_turn)
            return

        bm = get_legal_moves(self.black_bb, self.white_bb)
        wm = get_legal_moves(self.white_bb, self.black_bb)

        if not bm and not wm:
            b, w = popcount(self.black_bb), popcount(self.white_bb)
            b, w = int(b), int(w)
            if self.human_color == 1:
                result = "Win!" if b > w else "Lose!" if b < w else "Draw!"
            else:
                result = "Win!" if w > b else "Lose!" if w < b else "Draw!"
            messagebox.showinfo("ゲーム終了", f"黒: {b}  白: {w}\n{result}")
            return

        cur_bb  = self.black_bb if self.current_turn == 1 else self.white_bb
        opp_bb  = self.white_bb if self.current_turn == 1 else self.black_bb

        if not get_legal_moves(cur_bb, opp_bb):
            print("パス発生 → 手番反転")
            self.current_turn = -self.current_turn
            self.root.after(100, self.check_turn)
            return

        if self.current_turn == self.ai_color:
            threading.Thread(target=self.ai_routine, daemon=True).start()
        else:
            self.draw()
            self.status.config(text="Your turn")


if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloApp(root)
    root.mainloop()
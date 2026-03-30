import os, time, math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
from numba import njit as _numba_njit, set_num_threads
import torch 
import torch.nn as nn
import platform
import psutil

print("Loading files...", flush=True)

def njit(*args, **kwargs):
    kwargs["cache"] = False
    return _numba_njit(*args, **kwargs)

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import engine.othello_engine as cpp_engine
except ImportError:
    cpp_engine = None

if cpp_engine is not None:
    print("C++ engine: othello_engine loaded.", flush=True)
else:
    print("C++ engine: othello_engine unavailable. Python fallback will be used.", flush=True)

# ONNX推論エンジン
try:
    from .onnx_inference import ONNXInference, create_onnx_model_from_pytorch
    ONNX_AVAILABLE = True
    print("ONNX inference engine available.", flush=True)
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX inference engine unavailable.", flush=True)


_NN_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# PCスペックを自動検出して最適なスレッド数を設定
import os
import multiprocessing

def get_optimal_thread_count():
    """PCスペックに応じた最適なスレッド数を返す（Numba上限20を考慮）"""
    cpu_count = multiprocessing.cpu_count()
    
    # Intel Core Ultra 7 265KFは24スレッドだが、Numbaの上限は20
    if cpu_count >= 24:
        return 20  # Numbaの上限
    elif cpu_count >= 16:
        return 20  # Numbaの上限
    elif cpu_count >= 8:
        return min(20, cpu_count * 2)  # 中性能CPU：20スレッドまで
    else:
        return min(16, cpu_count * 2)  # 低性能CPU：16スレッドまで

optimal_threads = get_optimal_thread_count()
set_num_threads(optimal_threads)
print(f"CPU threads set to {optimal_threads} (detected {multiprocessing.cpu_count()} cores)", flush=True)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_WEIGHT = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "data", "best_weights.json")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "data", "model_best.pth")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "data", "model_best.onnx")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# ONNX推論エンジン
_onnx_inference_engine = None
_use_onnx = False

def initialize_onnx_engine(force_pytorch: bool = False):
    """ONNX推論エンジンを初期化"""
    global _onnx_inference_engine, _use_onnx
    
    if force_pytorch or not ONNX_AVAILABLE:
        _use_onnx = False
        print("Using PyTorch inference engine.", flush=True)
        return
    
    try:
        # ONNXモデルが存在しない場合は作成
        if not os.path.exists(ONNX_MODEL_PATH) and os.path.exists(BEST_MODEL_PATH):
            print(f"Converting PyTorch model to ONNX: {BEST_MODEL_PATH} -> {ONNX_MODEL_PATH}")
            create_onnx_model_from_pytorch(BEST_MODEL_PATH, ONNX_MODEL_PATH)
        
        # ONNX推論エンジンを初期化
        if os.path.exists(ONNX_MODEL_PATH):
            _onnx_inference_engine = ONNXInference(ONNX_MODEL_PATH, use_gpu=(DEVICE_STR == "cuda"))
            _use_onnx = True
            print("Using ONNX inference engine.", flush=True)
        else:
            _use_onnx = False
            print("ONNX model not found, using PyTorch inference engine.", flush=True)
    except Exception as e:
        _use_onnx = False
        print(f"Failed to initialize ONNX engine: {e}", flush=True)
        print("Falling back to PyTorch inference engine.", flush=True)

def get_inference_engine():
    """現在の推論エンジンを取得"""
    return _onnx_inference_engine if _use_onnx else None

def is_using_onnx():
    """ONNXを使用しているか確認"""
    return _use_onnx

def get_runtime_device_str():
    if _use_onnx and _onnx_inference_engine is not None:
        return _onnx_inference_engine.get_runtime_device()
    return DEVICE_STR

def print_system_info():
    """システム情報を表示"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # OS情報
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    
    # CPU情報
    cpu_count = multiprocessing.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # より詳細なCPU情報を取得
    try:
        import wmi
        c = wmi.WMI()
        cpu_info = c.Win32_Processor()[0]
        cpu_name = cpu_info.Name.strip()
    except:
        # WMIが使えない場合はpsutilの情報を使用
        cpu_name = platform.processor() or "Unknown CPU"
        if "Family" in cpu_name and "Model" in cpu_name:
            cpu_name = "Unknown CPU (see system properties)"
    
    print(f"\nCPU:")
    print(f"  Model: {cpu_name}")
    print(f"  Cores: {cpu_count} logical")
    print(f"  Frequency: {cpu_freq.current:.0f} MHz" if cpu_freq else "  Frequency: Unknown")
    print(f"  Usage: {psutil.cpu_percent()}%")
    
    # メモリ情報
    memory = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Usage: {memory.percent}%")
    
    # GPU情報
    print(f"\nGPU:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_name = gpu_props.name.replace("NVIDIA ", "")
            gpu_memory = gpu_props.total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
            print(f"    Compute: {gpu_props.major}.{gpu_props.minor}")
            print(f"    Cores: {gpu_props.multi_processor_count}")
        
        current_device = torch.cuda.current_device()
        print(f"  Current: GPU {current_device} ({torch.cuda.get_device_name(current_device)})")
        print(f"  Memory Usage: {torch.cuda.memory_allocated(current_device) / (1024**3):.1f} GB / {torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.1f} GB")
    else:
        print("  No CUDA GPU available")
    
    # 推論エンジン情報
    print(f"\nInference Engine:")
    if ONNX_AVAILABLE and _onnx_inference_engine is not None:
        info = _onnx_inference_engine.get_model_info()
        print(f"  Engine: ONNX Runtime")
        print(f"  Providers: {', '.join(info['providers'])}")
        print(f"  Input Shape: {info['input_shape']}")
    else:
        print(f"  Engine: PyTorch")
        print(f"  Device: {DEVICE_STR}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    
    print("=" * 60)
    print()

_FULL_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_NOT_A_FILE = np.uint64(0xFEFEFEFEFEFEFEFE)
_NOT_H_FILE = np.uint64(0x7F7F7F7F7F7F7F7F)
GENE_LEN = 243
TT_SIZE = int(2**23)

MASK_CORNER = np.uint64(0x8100000000000081)
MASK_X = np.uint64(0x0042000000004200)
MASK_C = np.uint64(0x4281000000008142)
MASK_B1 = np.uint64(0x0F0F0F0F00000000)
MASK_B2 = np.uint64(0xF0F0F0F000000000)
MASK_B3 = np.uint64(0x000000000F0F0F0F)
MASK_B4 = np.uint64(0x00000000F0F0F0F0)

def _generate_zobrist():
    zb = np.zeros(64, dtype=np.uint64)
    zw = np.zeros(64, dtype=np.uint64)
    state = np.uint64(123456789)
    a = np.uint64(2862933555777941757)
    c = np.uint64(3037000493)
    with np.errstate(over='ignore'):
        for i in range(64):
            state = state * a + c
            zb[i] = state
            state = state * a + c
            zw[i] = state
    return zb, zw

ZOBRIST_BLACK, ZOBRIST_WHITE = _generate_zobrist()

def make_board_perfect_seed():
    b = np.array([10.0,-5.0,3.0,2.0,2.0,3.0,-5.0,10.0,-5.0,-8.0,-1.0,-1.0,-1.0,-1.0,-8.0,-5.0,3.0,-1.0,1.5,1.0,1.0,1.5,-1.0,3.0,2.0,-1.0,1.0,0.2,0.2,1.0,-1.0,2.0,2.0,-1.0,1.0,0.2,0.2,1.0,-1.0,2.0,3.0,-1.0,1.5,1.0,1.0,1.5,-1.0,3.0,-5.0,-8.0,-1.0,-1.0,-1.0,-1.0,-8.0,-5.0,10.0,-5.0,3.0,2.0,2.0,3.0,-5.0,10.0], dtype=np.float64)
    w = np.zeros(GENE_LEN, dtype=np.float64)
    w[0:64] = b * 0.5
    w[80:144] = b * 1.2
    w[160:224] = b * 2.0
    w[64:80] = np.array([1.2,0.6,1.0,-0.8,2.5,0.2,-2.5,-1.5,0.2,0.2,0.2,0.2,0.3,0.3,0.3,0.3], dtype=np.float64)
    w[144:160] = np.array([0.8,0.4,2.5,-0.5,4.0,0.5,-3.5,-2.0,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8], dtype=np.float64)
    w[224:240] = np.array([0.2,0.1,5.0,-0.2,6.0,2.0,-1.0,-0.5,0.8,0.8,0.8,0.8,1.5,1.5,1.5,1.5], dtype=np.float64)
    w[240], w[241], w[242] = 0.45, 0.85, 0.05
    return w

inital_weights = make_board_perfect_seed()

def calculate_win_rate(ev, ie=False):
    if ie: 
        # exact solveでも控えめな勝率計算にする
        if ev > 1000:
            return 95.0 + min(5.0, (ev - 1000) / 1000.0)  # 最大100%まで
        elif ev > 0:
            return 50.0 + min(45.0, ev / 22.22)  # ev=1000で95%
        elif ev < -1000:
            return 5.0 - min(5.0, (-ev - 1000) / 1000.0)  # 最小0%まで
        else:
            return 50.0 + max(-45.0, ev / 22.22)  # ev=-1000で5%
    return 100.0 / (1.0 + math.exp(-max(-4000.0, min(4000.0, ev)) / 400.0))

def compute_blend_weights(
    mvs,
    empty,
    ab_score,
    mcts_score,
    resolved_flag,
    use_mcts,
    mcts_visits=0,
    max_ab_depth=2,
    mcts_sim_count=0,
    time_limit_sec=0.0,
    device_str="cpu",
):
    if empty <= 22 or resolved_flag or not use_mcts:
        return 1.0, 0.0
    if mcts_visits <= 0 and mcts_sim_count <= 0:
        return 1.0, 0.0

    opening_phase = min(1.0, max(0.0, (float(empty) - 20.0) / 24.0))
    endgame_phase = min(1.0, max(0.0, (44.0 - float(empty)) / 22.0))
    depth_reliability = 1.0 - math.exp(-max(0.0, float(max_ab_depth) - 2.0) / 4.5)
    visit_target = 60.0 + max(0.0, float(empty) - 20.0) * 10.0
    sim_target = 700.0 + max(0.0, float(empty) - 18.0) * 90.0
    visit_reliability = 1.0 - math.exp(-float(max(0, mcts_visits)) / max(1.0, visit_target))
    sim_reliability = 1.0 - math.exp(-float(max(0, mcts_sim_count)) / max(1.0, sim_target))
    time_reliability = 1.0 - math.exp(-max(0.0, float(time_limit_sec)) / 2.5)
    mcts_reliability = min(1.0, max(0.0, 0.50 * visit_reliability + 0.35 * sim_reliability + 0.15 * time_reliability))
    ab_reliability = min(1.0, max(0.0, 0.45 + 0.45 * depth_reliability + 0.10 * endgame_phase))

    ab_gap = abs(ab_score - 50.0)
    mcts_gap = abs(mcts_score - 50.0)
    ab_confidence = min(1.0, ab_gap / 35.0)
    mcts_confidence = min(1.0, mcts_gap / 35.0)

    ab_raw = (0.42 + 0.45 * endgame_phase) * ab_reliability * (0.75 + 0.35 * ab_confidence)
    mcts_raw = (0.28 + 0.55 * opening_phase) * mcts_reliability * (0.72 + 0.40 * mcts_confidence)

    if device_str == "cuda":
        mcts_raw *= 1.04
    if mcts_visits < max(32, int(10 + empty * 1.8)):
        mcts_raw *= 0.55
    if max_ab_depth <= 6:
        ab_raw *= 0.82

    if ab_gap > 25.0 and ab_gap > mcts_gap + 10.0:
        ab_raw *= 1.12
    elif mcts_gap > 20.0 and mcts_gap > ab_gap + 10.0 and mcts_visits >= 100:
        mcts_raw *= 1.12

    total = ab_raw + mcts_raw
    if total <= 0.0:
        return 1.0, 0.0
    w_ab = ab_raw / total
    if empty <= 26:
        w_ab = max(w_ab, 0.70)
    w_ab = max(0.12, min(0.95, w_ab))
    return w_ab, 1.0 - w_ab

def blend_search_scores(
    move_idx,
    mvs,
    empty,
    ab_score,
    mcts_score,
    resolved_flag,
    use_mcts,
    mcts_visits=0,
    max_ab_depth=2,
    mcts_sim_count=0,
    time_limit_sec=0.0,
    device_str="cpu",
):
    w_ab, w_mcts = compute_blend_weights(
        mvs,
        empty,
        ab_score,
        mcts_score,
        resolved_flag,
        use_mcts,
        mcts_visits,
        max_ab_depth,
        mcts_sim_count,
        time_limit_sec,
        device_str,
    )
    return ab_score * w_ab + mcts_score * w_mcts

@njit(cache=True, fastmath=True, nogil=True)
def get_rotated_bitboard(b, op):
    res = np.uint64(0)
    if op == 0: return b
    elif op == 1:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64(c * 8 + (7 - r)))
    elif op == 2:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64((7 - r) * 8 + (7 - c)))
    elif op == 3:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64((7 - c) * 8 + r))
    elif op == 4:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64(r * 8 + (7 - c)))
    elif op == 5:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64((7 - r) * 8 + c))
    elif op == 6:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64(c * 8 + r))
    elif op == 7:
        for i in range(np.int64(64)):
            if (b >> np.uint64(i)) & np.uint64(1):
                r, c = i // 8, i % 8
                res |= (np.uint64(1) << np.uint64((7 - c) * 8 + (7 - r)))
    return res

@njit(cache=True, fastmath=True, nogil=True)
def unrotate_move(bm_rot, op):
    if bm_rot < 0 or bm_rot > 63: return bm_rot
    r = bm_rot // 8
    c = bm_rot % 8
    nr, nc = r, c
    if op == 0: nr, nc = r, c
    elif op == 1: nr, nc = 7 - c, r
    elif op == 2: nr, nc = 7 - r, 7 - c
    elif op == 3: nr, nc = c, 7 - r
    elif op == 4: nr, nc = r, 7 - c
    elif op == 5: nr, nc = 7 - r, c
    elif op == 6: nr, nc = c, r
    elif op == 7: nr, nc = 7 - c, 7 - r
    return nr * 8 + nc

@njit(cache=True, fastmath=True, nogil=True)
def zobrist_hash(P, O): 
    h = np.uint64(0)
    for i in range(np.int64(64)):
        if (P >> np.uint64(i)) & np.uint64(1): h ^= ZOBRIST_BLACK[i]
        elif (O >> np.uint64(i)) & np.uint64(1): h ^= ZOBRIST_WHITE[i]
    return h

@njit(cache=True, fastmath=True, nogil=True)
def lsb(x): return x & (~x + np.uint64(1))

@njit(cache=True, fastmath=True, nogil=True)
def count_bits(x):
    c = np.int64(0)
    while x:
        c += np.int64(1)
        x &= x - np.uint64(1)
    return c

def get_legal_moves(P, O):
    if cpp_engine is not None:
        return np.uint64(cpp_engine.get_legal_moves(int(P), int(O)))
    return _get_legal_moves_numba(P, O)

@njit(cache=True, fastmath=True, nogil=True)
def _get_legal_moves_numba(P, O):
    occ = (P | O) & _FULL_MASK
    legal = np.uint64(0)
    for idx in range(np.int64(64)):
        bit = np.uint64(1) << np.uint64(idx)
        if occ & bit:
            continue
        r = idx // np.int64(8)
        c = idx % np.int64(8)
        found = False
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr = r + dr
            cc = c + dc
            seen_opponent = False
            while np.int64(0) <= rr < np.int64(8) and np.int64(0) <= cc < np.int64(8):
                nidx = rr * np.int64(8) + cc
                nbit = np.uint64(1) << np.uint64(nidx)
                if O & nbit:
                    seen_opponent = True
                    rr += dr
                    cc += dc
                    continue
                if seen_opponent and (P & nbit):
                    found = True
                break
            if found:
                legal |= bit
                break
    return legal & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def get_flip(P, O, idx):
    r = idx // np.int64(8)
    c = idx % np.int64(8)
    flips = np.uint64(0)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        rr = r + dr
        cc = c + dc
        captured = np.uint64(0)
        seen_opponent = False
        while np.int64(0) <= rr < np.int64(8) and np.int64(0) <= cc < np.int64(8):
            nidx = rr * np.int64(8) + cc
            nbit = np.uint64(1) << np.uint64(nidx)
            if O & nbit:
                seen_opponent = True
                captured |= nbit
                rr += dr
                cc += dc
                continue
            if seen_opponent and (P & nbit):
                flips |= captured
            break
    return flips & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def neighbor_union(bb):
    n = np.uint64(0)
    n |= (bb << np.uint64(1)) & _NOT_A_FILE; n |= (bb >> np.uint64(1)) & _NOT_H_FILE
    n |= (bb << np.uint64(8)) & _FULL_MASK; n |= (bb >> np.uint64(8)) & _FULL_MASK
    n |= (bb << np.uint64(7)) & _NOT_A_FILE; n |= (bb >> np.uint64(7)) & _NOT_H_FILE
    n |= (bb << np.uint64(9)) & _NOT_H_FILE; n |= (bb >> np.uint64(9)) & _NOT_A_FILE
    return n & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def compute_strict_stable(P, O):
    s = P & MASK_CORNER
    if not s: return np.uint64(0)
    for _ in range(np.int64(7)):
        ns = np.uint64(0)
        ns |= (s << np.uint64(1)) & _NOT_A_FILE; ns |= (s >> np.uint64(1)) & _NOT_H_FILE
        ns |= (s << np.uint64(8)) & _FULL_MASK; ns |= (s >> np.uint64(8)) & _FULL_MASK
        ns |= (s << np.uint64(7)) & _NOT_A_FILE; ns |= (s >> np.uint64(7)) & _NOT_H_FILE
        ns |= (s << np.uint64(9)) & _NOT_H_FILE; ns |= (s >> np.uint64(9)) & _NOT_A_FILE
        a = (ns & P) & ~s
        if not a: break
        s |= a
    ch, it, occ = True, np.int64(0), P | O
    while ch and it < np.int64(8):
        ch = False; b = np.uint64(1)
        for _ in range(np.int64(64)):
            if (P & b) and not(s & b):
                ld = np.int64(0)
                for d in range(np.int64(8)):
                    c, dl, ac = b, False, True
                    for _ in range(np.int64(8)):
                        if not ac: break
                        if d == np.int64(0): c = (c << np.uint64(1)) & _NOT_A_FILE
                        elif d == np.int64(1): c = (c >> np.uint64(1)) & _NOT_H_FILE
                        elif d == np.int64(2): c = (c << np.uint64(8)) & _FULL_MASK
                        elif d == np.int64(3): c = (c >> np.uint64(8)) & _FULL_MASK
                        elif d == np.int64(4): c = (c << np.uint64(7)) & _NOT_A_FILE
                        elif d == np.int64(5): c = (c >> np.uint64(7)) & _NOT_H_FILE
                        elif d == np.int64(6): c = (c << np.uint64(9)) & _NOT_H_FILE
                        elif d == np.int64(7): c = (c >> np.uint64(9)) & _NOT_A_FILE
                        if not c or not(c & occ): dl = True; break
                        if c & P:
                            if c & s: dl = True; break
                            else: ac = False; break
                        if c & O: ac = False; break
                    if dl: ld += np.int64(1)
                if ld >= np.int64(4): s |= b; ch = True
            b <<= np.uint64(1)
        it += np.int64(1)
    return s & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def eval_xc(p, o, cor, cx, cc):
    v = np.float64(0.0)
    if not (p & cor) and not (o & cor):
        if p & cx: v -= np.float64(50.0)
        if o & cx: v += np.float64(50.0)
        v -= np.float64(count_bits(p & cc)) * np.float64(20.0)
        v += np.float64(count_bits(o & cc)) * np.float64(20.0)
    elif p & cor:
        if p & cx: v += np.float64(15.0)
        v += np.float64(count_bits(p & cc)) * np.float64(10.0)
    else:
        if o & cx: v -= np.float64(15.0)
        v -= np.float64(count_bits(o & cc)) * np.float64(10.0)
    return v

@njit(cache=True, fastmath=True, nogil=True)
def evaluate_board_full(P, O, mvs, W):
    st = np.int64(0) if mvs <= np.int64(15) else (np.int64(80) if mvs <= np.int64(45) else np.int64(160))
    wp, we, sc = W[st:st+np.int64(64)], W[st+np.int64(64):st+np.int64(80)], np.float64(0.0)
    tp, to = P, O
    while tp:
        ix, t = np.int64(0), lsb(tp)
        while t > np.uint64(1): t >>= np.uint64(1); ix += np.int64(1)
        sc += wp[ix]; tp &= tp - np.uint64(1)
    while to:
        ix, t = np.int64(0), lsb(to)
        while t > np.uint64(1): t >>= np.uint64(1); ix += np.int64(1)
        sc -= wp[ix]; to &= to - np.uint64(1)
    lm, lo = count_bits(_get_legal_moves_numba(P, O)), count_bits(_get_legal_moves_numba(O, P))
    emp = ~(P | O) & _FULL_MASK; np_, no = neighbor_union(P), neighbor_union(O)
    sp, so = compute_strict_stable(P, O), compute_strict_stable(O, P); occ = P | O
    m_mult = np.float64(2.5) if np.int64(20) <= mvs <= np.int64(45) else np.float64(1.0)
    sc += np.float64(lm - lo) * m_mult * np.float64(4.0)
    if (np.int64(64) - mvs) % np.int64(2) == np.int64(0): sc += np.float64(10.0)
    else: sc -= np.float64(10.0)
    c1, x1, cc1 = np.uint64(0x1), np.uint64(0x200), np.uint64(0x102)
    c2, x2, cc2 = np.uint64(0x80), np.uint64(0x4000), np.uint64(0x8040)
    c3, x3, cc3 = np.uint64(0x100000000000000), np.uint64(0x20000000000000), np.uint64(0x201000000000000)
    c4, x4, cc4 = np.uint64(0x8000000000000000), np.uint64(0x400000000000000), np.uint64(0x4080000000000000)
    sc += eval_xc(P, O, c1, x1, cc1)
    sc += eval_xc(P, O, c2, x2, cc2)
    sc += eval_xc(P, O, c3, x3, cc3)
    sc += eval_xc(P, O, c4, x4, cc4)
    if mvs >= np.int64(30): sc += np.float64(count_bits(sp & P) - count_bits(so & O)) * np.float64(25.0)
    sc += (np.float64(lm - lo) / np.float64(20.0)) * we[np.int64(0)]
    sc += (np.float64(count_bits(emp & np_) - count_bits(emp & no)) / np.float64(64.0)) * we[np.int64(1)]
    sc += np.float64(count_bits(sp & P) - count_bits(so & O)) * we[np.int64(2)]
    sc += (np.float64(count_bits(P & np_) - count_bits(O & no)) / np.float64(64.0)) * we[np.int64(3)]
    sc += np.float64(count_bits(P & MASK_CORNER) - count_bits(O & MASK_CORNER)) * we[np.int64(4)]
    sc += np.float64(count_bits(occ) & np.int64(1)) * we[np.int64(5)]
    sc += np.float64(count_bits(P & MASK_X) - count_bits(O & MASK_X)) * we[np.int64(6)]
    sc += np.float64(count_bits(P & MASK_C) - count_bits(O & MASK_C)) * we[np.int64(7)]
    sc += (np.float64(count_bits(P & MASK_B1) - count_bits(O & MASK_B1)) / np.float64(16.0)) * we[np.int64(8)]
    sc += (np.float64(count_bits(P & MASK_B2) - count_bits(O & MASK_B2)) / np.float64(16.0)) * we[np.int64(9)]
    sc += (np.float64(count_bits(P & MASK_B3) - count_bits(O & MASK_B3)) / np.float64(16.0)) * we[np.int64(10)]
    sc += (np.float64(count_bits(P & MASK_B4) - count_bits(O & MASK_B4)) / np.float64(16.0)) * we[np.int64(11)]
    sc += np.float64(count_bits(occ & MASK_B1) & np.int64(1)) * we[np.int64(12)]
    sc += np.float64(count_bits(occ & MASK_B2) & np.int64(1)) * we[np.int64(13)]
    sc += np.float64(count_bits(occ & MASK_B3) & np.int64(1)) * we[np.int64(14)]
    sc += np.float64(count_bits(occ & MASK_B4) & np.int64(1)) * we[np.int64(15)]
    return sc

@njit(cache=True, fastmath=True, nogil=True)
def exact_eval(P, O):
    return np.float64((count_bits(P) - count_bits(O)) * 10000.0)

@njit(cache=True, fastmath=True, nogil=True)
def alphabeta(P, O, mvs, depth, alpha, beta, passed, is_exact, W, order_map, tk_arr, ti_arr, stop_flag):
    if stop_flag[0]: return 0.0, 0
    hv = zobrist_hash(P, O)
    tx0 = np.int64(hv % np.uint64(TT_SIZE)) * 2
    tx1 = tx0 + 1
    tm = -1
    oa = alpha
    
    if tk_arr[tx0] == hv:
        td, tf, tv, tm = np.int64(ti_arr[tx0, 0]), np.int64(ti_arr[tx0, 1]), ti_arr[tx0, 2], np.int64(ti_arr[tx0, 3])
        if td >= depth:
            if tf == 1 or (tf == 2 and tv >= beta) or (tf == 3 and tv <= alpha): return tv, 1
    elif tk_arr[tx1] == hv:
        td, tf, tv, tm = np.int64(ti_arr[tx1, 0]), np.int64(ti_arr[tx1, 1]), ti_arr[tx1, 2], np.int64(ti_arr[tx1, 3])
        if td >= depth:
            if tf == 1 or (tf == 2 and tv >= beta) or (tf == 3 and tv <= alpha): return tv, 1

    valid = get_legal_moves(P, O)
    if not valid:
        if passed:
            return exact_eval(P, O), 1
        v, n = alphabeta(O, P, mvs, depth, -beta, -alpha, True, is_exact, W, order_map, tk_arr, ti_arr, stop_flag)
        return -v, n + 1
    
    if depth <= 0:
        if is_exact: return exact_eval(P, O), 1
        return evaluate_board_full(P, O, mvs, W), 1

    count = count_bits(valid)
    moves = np.zeros(count, dtype=np.uint64)
    scores = np.zeros(count, dtype=np.float64)
    
    v_temp = valid
    idx = 0
    while v_temp:
        bit = v_temp & (~v_temp + np.uint64(1))
        v_temp ^= bit
        moves[idx] = bit
        sq_idx = count_bits(bit - np.uint64(1))
        scores[idx] = order_map[sq_idx]
        if sq_idx == tm:
            scores[idx] += 1e9
        idx += 1
        
    for i in range(1, count):
        key_m = moves[i]
        key_s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_s:
            moves[j + 1] = moves[j]
            scores[j + 1] = scores[j]
            j -= 1
        moves[j + 1] = key_m
        scores[j + 1] = key_s

    max_val = -1e18
    bm = -1
    nodes = 1
    
    for i in range(count):
        if stop_flag[0]: break
        bit = moves[i]
        sq_idx = count_bits(bit - np.uint64(1))
        f = get_flip(P, O, sq_idx)
        
        nP = (P | bit | f) & _FULL_MASK
        nO = (O ^ f) & _FULL_MASK

        if i == 0:
            val, n = alphabeta(nO, nP, mvs + 1, depth - 1, -beta, -alpha, False, is_exact, W, order_map, tk_arr, ti_arr, stop_flag)
            val = -val
            nodes += n
        else:
            val, n = alphabeta(nO, nP, mvs + 1, depth - 1, -alpha - 1.0, -alpha, False, is_exact, W, order_map, tk_arr, ti_arr, stop_flag)
            val = -val
            nodes += n
            if val > alpha and val < beta:
                val2, n2 = alphabeta(nO, nP, mvs + 1, depth - 1, -beta, -val, False, is_exact, W, order_map, tk_arr, ti_arr, stop_flag)
                val = -val2
                nodes += n2
        
        if val > max_val:
            max_val = val
            bm = sq_idx
        if max_val > alpha:
            alpha = max_val
        if alpha >= beta:
            break

    fl = 2 if max_val >= beta else (3 if max_val <= oa else 1)
    if not stop_flag[0]:
        r0 = False
        if tk_arr[tx0] == hv:
            if float(depth) >= ti_arr[tx0, 0]: r0 = True
        elif tk_arr[tx0] == 0 or float(depth) >= ti_arr[tx0, 0]: r0 = True
        
        if r0:
            tk_arr[tx0] = hv
            ti_arr[tx0, 0], ti_arr[tx0, 1], ti_arr[tx0, 2], ti_arr[tx0, 3] = float(depth), float(fl), max_val, float(bm)
        else:
            r1 = False
            if tk_arr[tx1] == hv:
                if float(depth) >= ti_arr[tx1, 0]: r1 = True
            elif tk_arr[tx1] == 0 or float(depth) >= ti_arr[tx1, 0]: r1 = True
            if r1:
                tk_arr[tx1] = hv
                ti_arr[tx1, 0], ti_arr[tx1, 1], ti_arr[tx1, 2], ti_arr[tx1, 3] = float(depth), float(fl), max_val, float(bm)

    return max_val, nodes

@njit(cache=True, fastmath=True, nogil=True)
def search_root_parallel(P, O, mvs, depth, is_exact, ordered_indices, W, order_map, tk_arr, ti_arr, stop_flag):
    count = len(ordered_indices)
    vals = np.zeros(count, dtype=np.float64)
    nodes = np.zeros(count, dtype=np.int64)
    
    for i in range(count):
        if stop_flag[0]:
            continue
        idx = ordered_indices[i]
        bit = np.uint64(1) << np.uint64(idx)
        f = get_flip(P, O, idx)
        
        nP = (P | bit | f) & _FULL_MASK
        nO = (O ^ f) & _FULL_MASK
        
        v, n = alphabeta(nO, nP, mvs + 1, depth - 1, np.float64(-1e18), np.float64(1e18), False, is_exact, W, order_map, tk_arr, ti_arr, stop_flag)
        vals[i] = -v
        nodes[i] = n

    return vals, nodes

_NUMBA_WARMED_UP = False
_NUMBA_WARMUP_STARTED = False

def _warmup_numba():
    P = np.uint64(0x0000000810000000)
    O = np.uint64(0x0000001008000000)
    get_legal_moves(P, O)
    get_flip(P, O, np.int64(19))
    evaluate_board_full(P, O, np.int64(4), inital_weights)
    exact_eval(P, O)
    W = inital_weights
    order_map = W[0:64]
    ordered_indices = np.array([19, 26, 37, 44], dtype=np.int64)
    
    tk_arr = np.zeros(TT_SIZE * 2, dtype=np.uint64)
    ti_arr = np.zeros((TT_SIZE * 2, 4), dtype=np.float64)
    sf_arr = np.zeros(1, dtype=np.uint8)
    
    # テストコードを無効化
    # search_root_parallel(P, O, np.int64(4), 1, False, ordered_indices, W, order_map, tk_arr, ti_arr, sf_arr)
    # search_root_parallel(P, O, np.int64(60), 1, True, ordered_indices, W, order_map, tk_arr, ti_arr, sf_arr)
    # get_rotated_bitboard(P, 1)
    # unrotate_move(19, 1)

def ensure_numba_warmup():
    global _NUMBA_WARMED_UP
    if _NUMBA_WARMED_UP:
        return False
    _warmup_numba()
    _NUMBA_WARMED_UP = True
    return True

@njit(cache=True, fastmath=True, nogil=True)
def states_to_tensor_numba(P_arr, O_arr, turn_arr):
    n = len(P_arr)
    t = np.zeros((n, 3, 8, 8), dtype=np.float32)
    for idx in range(n):
        P = P_arr[idx]
        O = O_arr[idx]
        turn = turn_arr[idx]
        for i in range(64):
            if (P >> np.uint64(i)) & np.uint64(1): t[idx, 0, i // 8, i % 8] = 1.0
            if (O >> np.uint64(i)) & np.uint64(1): t[idx, 1, i // 8, i % 8] = 1.0
        if turn == 1:
            for r in range(8):
                for c in range(8):
                    t[idx, 2, r, c] = 1.0
    return t

def board_to_tensor_batch(states):
    n = len(states)
    P_arr = np.empty(n, dtype=np.uint64)
    O_arr = np.empty(n, dtype=np.uint64)
    turn_arr = np.empty(n, dtype=np.int32)
    for i in range(n):
        P_arr[i] = np.uint64(states[i][0])
        O_arr[i] = np.uint64(states[i][1])
        turn_arr[i] = states[i][2]
    return states_to_tensor_numba(P_arr, O_arr, turn_arr)

def nn_infer_batch(model, tensor_batch):
    """バッチ推論を実行（実行時はONNX優先）"""
    if _use_onnx and _onnx_inference_engine is not None:
        if isinstance(tensor_batch, np.ndarray):
            input_batch = tensor_batch
        else:
            input_batch = tensor_batch.detach().cpu().numpy()
        return _onnx_inference_engine.infer_batch(input_batch)

    if model is None:
        raise RuntimeError("ONNX runtime is not initialized and no PyTorch model is available")

    try:
        with torch.inference_mode():
            if DEVICE_STR == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    p_b, v_b = model(tensor_batch)
                    torch.cuda.synchronize()
            else:
                p_b, v_b = model(tensor_batch)
        return torch.softmax(p_b, dim=1).float().cpu().numpy(), v_b.float().cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear GPU memory and retry once
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            with torch.inference_mode():
                if DEVICE_STR == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        p_b, v_b = model(tensor_batch)
                        torch.cuda.synchronize()
                else:
                    p_b, v_b = model(tensor_batch)
            return torch.softmax(p_b, dim=1).float().cpu().numpy(), v_b.float().cpu().numpy()
        else:
            raise e

def make_input_tensor(states):
    tensors = board_to_tensor_batch(states)
    if _use_onnx and _onnx_inference_engine is not None:
        return tensors
    cpu_tensor = torch.from_numpy(tensors)
    if DEVICE_STR == "cuda":
        cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
    return cpu_tensor.to(device=DEVICE, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

class OthelloNet(nn.Module):
    def __init__(self, num_blocks=15, channels=192):
        super().__init__()
        self.start_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.view(p.size(0), -1))
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = torch.relu(self.value_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.value_fc2(v))
        return p, v

def get_nn_activation_snapshot(model, P, O, turn, top_k=6):
    if model is None or _use_onnx or not hasattr(model, "eval"):
        return None
    model.eval()
    tensor_batch = torch.from_numpy(board_to_tensor_batch([(int(P), int(O), int(turn))])).to(device=DEVICE, dtype=torch.float32)
    with torch.inference_mode():
        x = torch.relu(model.bn(model.start_conv(tensor_batch)))
        for block in model.res_blocks:
            x = block(x)
        trunk = x[0].detach().float().cpu()
        channel_strength = trunk.mean(dim=(1, 2))
        k = max(1, min(int(top_k), int(trunk.shape[0])))
        top_indices = torch.topk(channel_strength, k=k).indices.tolist()
        policy_head = torch.relu(model.policy_bn(model.policy_conv(x)))[0].detach().float().cpu()
        value_head = torch.relu(model.value_bn(model.value_conv(x)))[0, 0].detach().float().cpu()
    trunk_maps = []
    for idx in top_indices:
        trunk_maps.append(
            {
                "label": f"C{int(idx)}",
                "strength": float(channel_strength[idx].item()),
                "grid": trunk[idx].numpy(),
            }
        )
    return {
        "policy_grid": policy_head.mean(dim=0).numpy(),
        "value_grid": value_head.numpy(),
        "trunk_maps": trunk_maps,
    }

def build_valid_mask(valid):
    mask = np.zeros(64, dtype=np.float32)
    valid_temp = int(valid)
    while valid_temp:
        a = (valid_temp & -valid_temp).bit_length() - 1
        valid_temp &= valid_temp - 1
        mask[a] = 1.0
    return mask


class GPUMCTS:
    def __init__(self, c_puct=2.0, virtual_loss=1.0):
        self.W_val = {}
        self.N = {}
        self.Ns = {}
        self.P_probs = {}
        self.visited = set()
        self.valid_masks = {}
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss

    def initialize_root(self, state, policy_logits, valid_mask, add_noise=False):
        priors = policy_logits * valid_mask
        sum_p = float(np.sum(priors))
        if sum_p > 0.0:
            priors /= sum_p
            if add_noise:
                noise = np.random.dirichlet([0.3] * int(np.sum(valid_mask)))
                idx = 0
                for i in range(64):
                    if valid_mask[i] > 0:
                        priors[i] = 0.75 * priors[i] + 0.25 * noise[idx]
                        idx += 1
        else:
            sum_mask = float(np.sum(valid_mask))
            priors = (valid_mask / sum_mask) if sum_mask > 0.0 else valid_mask
        self.visited.add(state)
        self.valid_masks[state] = valid_mask
        self.P_probs[state] = priors
        self.Ns[state] = 0

    def traverse(self, P, O, turn, sf_arr):
        path = []
        cP, cO, ct = int(P), int(O), turn
        max_depth = 200
        depth = 0
        
        while True:
            depth += 1
            if depth > max_depth or sf_arr[0]:
                return None, path,
            
            if sf_arr[0]:
                return None, [], 0.0
            state = (cP, cO, ct)
            valid = int(get_legal_moves(np.uint64(cP), np.uint64(cO)))
            if valid == 0:
                return state, path, None
            
            valid_moves = get_legal_moves(np.uint64(cP), np.uint64(cO))
            if valid_moves == 0:
                cP, cO, ct = cO, cP, -ct
                path.append((state, -1))
                continue
            
            best_U = -float('inf')
            best_move = -1
            
            valid_temp = valid_moves
            while valid_temp:
                move = (valid_temp & -valid_temp).bit_length() - 1
                valid_temp &= valid_temp - 1
                
                N_sum = self.Ns[state]
                N_sa = self.N.get((state, move), 0)
                W_sa = self.W_val.get((state, move), 0.0)
                P_sa = self.P_probs[state][move]
                
                if N_sum == 0:
                    U = float('inf')
                else:
                    Q_sa = W_sa / (N_sa + 1e-8)
                    U = Q_sa + self.c_puct * P_sa * math.sqrt(N_sum) / (1 + N_sa)
                
                if U > best_U:
                    best_U = U
                    best_move = move
            
            if best_move == -1:
                return None, path, 0
            
            path.append((state, best_move))
            
            f = get_flip(np.uint64(cP), np.uint64(cO), np.int64(best_move))
            nP = (cP | (1 << best_move) | f) & 0xFFFFFFFFFFFFFFFF
            nO = (cO ^ f) & 0xFFFFFFFFFFFFFFFF
            cP, cO, ct = nO, nP, -ct

    def apply_virtual_loss(self, path):
        for state, action in path:
            if action == -1:
                continue
            self.N[(state, action)] = self.N.get((state, action), 0) + 1
            self.Ns[state] = self.Ns.get(state, 0) + 1
            self.W_val[(state, action)] = self.W_val.get((state, action), 0.0) - self.virtual_loss

    def backpropagate_terminal(self, path, value):
        v = value
        for state, action in reversed(path):
            v = -v
            if action == -1:
                continue
            self.N[(state, action)] = self.N.get((state, action), 0) + 1
            self.Ns[state] = self.Ns.get(state, 0) + 1
            self.W_val[(state, action)] = self.W_val.get((state, action), 0.0) + v

    def expand_leaf(self, leaf, path, policy_logits, value, valid_mask):
        self.visited.add(leaf)
        self.valid_masks[leaf] = valid_mask
        priors = policy_logits * valid_mask
        sum_p = float(np.sum(priors))
        if sum_p > 0.0:
            priors /= sum_p
        else:
            sum_mask = float(np.sum(valid_mask))
            priors = (valid_mask / sum_mask) if sum_mask > 0.0 else valid_mask
        self.P_probs[leaf] = priors
        self.Ns.setdefault(leaf, 0)
        v = max(-1.0, min(1.0, float(value)))
        for state, action in reversed(path):
            v = -v
            if action == -1:
                continue
            self.W_val[(state, action)] = self.W_val.get((state, action), 0.0) + self.virtual_loss + v


def _current_mcts_batch_size(batch_size, remain):
    runtime_device = get_runtime_device_str()
    if runtime_device == "cuda":
        if remain >= 10.0:
            return min(batch_size, 4096)
        if remain >= 5.0:
            return min(batch_size, 3072)
        if remain >= 2.0:
            return min(batch_size, 2048)
        if remain >= 1.0:
            return min(batch_size, 1024)
        if remain >= 0.4:
            return min(batch_size, 512)
        return min(batch_size, 256)
    
    if remain >= 5.0:
        return min(batch_size, 1024)
    if remain >= 2.0:
        return min(batch_size, 512)
    if remain >= 1.0:
        return min(batch_size, 256)
    if remain >= 0.3:
        return min(batch_size, 128)
    return min(batch_size, 64)


def get_mcts_win_rates_time_batched(model, P, O, turn, time_limit, batch_size, sf_arr, add_root_noise=False):
    effective_time_limit = float(time_limit)
    if get_runtime_device_str() != "cuda":
        effective_time_limit *= 0.62
        effective_time_limit = max(0.15, effective_time_limit)
    state = (int(P), int(O), turn)
    start_t = time.perf_counter()
    deadline = start_t + effective_time_limit
    simulation_count = 0
    nn_batch_count = 0
    nn_leaf_count = 0

    valid = int(get_legal_moves(np.uint64(P), np.uint64(O)))

    if cpp_engine is not None and hasattr(cpp_engine, "SearchSession"):
        def infer_leaves(leaves):
            tensor_batch_local = make_input_tensor(leaves)
            p_leaf_local, v_leaf_local = nn_infer_batch(model, tensor_batch_local)
            return np.asarray(p_leaf_local, dtype=np.float32), np.asarray(v_leaf_local[:, 0], dtype=np.float32)
        session = cpp_engine.SearchSession()
        result = session.run(
            int(P),
            int(O),
            int(turn),
            int((int(P) | int(O)).bit_count() - 4),
            2,
            False,
            [],
            [],
            False,
            True,
            float(effective_time_limit),
            int(batch_size),
            0.0,
            0,
            60,
            bool(add_root_noise),
            sf_arr,
            infer_leaves,
            None,
        )
        mcts_out = dict(result["mcts"])
        move_win_rates = {int(k): float(v) for k, v in dict(mcts_out["move_win_rates"]).items()}
        root_visits = {int(k): int(v) for k, v in dict(mcts_out["root_visits"]).items()}
        best_wr = float(mcts_out.get("best_wr", 50.0))
        simulation_count = int(mcts_out.get("simulation_count", 0))
        nn_batch_count = int(mcts_out.get("nn_batch_count", 0))
        nn_leaf_count = int(mcts_out.get("nn_leaf_count", 0))
        return move_win_rates, best_wr, simulation_count, nn_batch_count, nn_leaf_count, root_visits

    tensor_batch = make_input_tensor([state])
    p_b, _ = nn_infer_batch(model, tensor_batch)
    nn_batch_count += 1
    nn_leaf_count += 1

    mcts = GPUMCTS()
    valid_mask = build_valid_mask(valid)
    mcts.initialize_root(state, p_b[0], valid_mask, add_noise=add_root_noise)

    while time.perf_counter() < deadline:
        if sf_arr[0]:
            break
        remain = deadline - time.perf_counter()
        if remain <= 0.0:
            break
        curr_batch = _current_mcts_batch_size(batch_size, remain)

        leaves = []
        paths = []
        leaf_masks = []
        for _ in range(curr_batch):
            if sf_arr[0] or time.perf_counter() >= deadline:
                break
            leaf, path, terminal_value = mcts.traverse(P, O, turn, sf_arr)
            simulation_count += 1
            if leaf is None:
                if path:
                    mcts.backpropagate_terminal(path, terminal_value)
                continue
            mcts.apply_virtual_loss(path)
            leaves.append(leaf)
            paths.append(path)
            valid_leaf = int(get_legal_moves(np.uint64(leaf[0]), np.uint64(leaf[1])))
            leaf_masks.append(build_valid_mask(valid_leaf))

        if not leaves:
            continue

        nn_batch_count += 1
        nn_leaf_count += len(leaves)
        tensor_batch = make_input_tensor(leaves)
        p_leaf, v_leaf = nn_infer_batch(model, tensor_batch)

        for idx, leaf in enumerate(leaves):
            if sf_arr[0] or time.perf_counter() >= deadline:
                break
            mcts.expand_leaf(leaf, paths[idx], p_leaf[idx], v_leaf[idx][0], leaf_masks[idx])

    move_win_rates = {}
    root_visits = {}
    best_a = -1
    max_n = -1
    total_root_n = 0
    valid_temp = valid
    while valid_temp:
        a = (valid_temp & -valid_temp).bit_length() - 1
        valid_temp &= valid_temp - 1
        n = mcts.N.get((state, a), 0)
        root_visits[a] = n
        total_root_n += n
        if n > max_n:
            max_n = n
            best_a = a

    valid_temp = valid
    while valid_temp:
        a = (valid_temp & -valid_temp).bit_length() - 1
        valid_temp &= valid_temp - 1
        n = root_visits[a]
        if n > 0:
            avg_q = max(-1.0, min(1.0, mcts.W_val.get((state, a), 0.0) / n))
            q_wr = (avg_q + 1.0) / 2.0 * 100.0
            visit_wr = (n / total_root_n) * 100.0 if total_root_n > 0 else 50.0
            move_win_rates[a] = 0.70 * q_wr + 0.30 * visit_wr
        else:
            move_win_rates[a] = 50.0

    best_wr = move_win_rates.get(best_a, 50.0) if best_a != -1 else 50.0
    return move_win_rates, best_wr, simulation_count, nn_batch_count, nn_leaf_count, root_visits


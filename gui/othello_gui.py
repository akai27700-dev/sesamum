import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog
import numpy as np
try:
    import torch
except (ImportError, OSError) as e:
    print(f"PyTorch import failed: {e}")
    print("Using ONNX-only mode")
    torch = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.othello_core as core
cpp_engine = core.cpp_engine
get_endgame_best_move = core.get_endgame_best_move
from gui.othello_gui_dialogs import StartupSettingsDialog
from gui.othello_gui_search import OthelloSearchMixin
BASE_DIR = core.BASE_DIR
BEST_MODEL_PATH = core.BEST_MODEL_PATH
BLEND_CALIBRATION_LOG_PATH = os.path.join(BASE_DIR, 'data', 'blend_calibration_samples.jsonl')
DEVICE = core.DEVICE
DEVICE_STR = core.DEVICE_STR
ENDGAME_SOLVER_AVAILABLE = core.ENDGAME_SOLVER_AVAILABLE
GENE_LEN = core.GENE_LEN
TT_SIZE = core.TT_SIZE
USE_WEIGHT = core.USE_WEIGHT
WEIGHTS_PATH = core.WEIGHTS_PATH
_FULL_MASK = core._FULL_MASK
_NN_EXECUTOR = core._NN_EXECUTOR
blend_search_scores = core.blend_search_scores
calculate_blend_weights = core.compute_blend_weights
calculate_win_rate = core.calculate_win_rate
count_bits = core.count_bits
ensure_numba_warmup = core.ensure_numba_warmup
evaluate_board_full = core.evaluate_board_full
get_flip = core.get_flip
get_legal_moves = core.get_legal_moves
get_mcts_win_rates_time_batched = core.get_mcts_win_rates_time_batched
get_nn_activation_snapshot = core.get_nn_activation_snapshot
get_rotated_bitboard = core.get_rotated_bitboard
inital_weights = core.inital_weights
neighbor_union = core.neighbor_union
OthelloNet = core.OthelloNet
search_root_parallel = core.search_root_parallel
compute_strict_stable = core.compute_strict_stable
unrotate_move = core.unrotate_move
zobrist_hash = core.zobrist_hash
_EGAROUCID_BOOK_PATH = os.path.join(BASE_DIR, 'data', 'book.egbk3')
_EGAROUCID_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'opening_book_egbk3_cache.npz')
_EGAROUCID_RECORD_DTYPE = np.dtype([('player', '<u8'), ('opponent', '<u8'), ('value', 'i1'), ('level', 'i1'), ('n_lines', '<u4'), ('leaf_value', 'i1'), ('leaf_move', 'i1'), ('leaf_level', 'i1')])
_ROTATE_MOVE_TABLE = [[0] * 64 for _ in range(8)]

def _build_rotate_move_table():
    """Build the rotation move lookup table"""
    for _op in range(8):
        for _rot_idx in range(64):
            _ROTATE_MOVE_TABLE[_op][int(unrotate_move(_rot_idx, _op))] = _rot_idx
_build_rotate_move_table()

def rotate_move(move_idx, op):
    """Rotate a move index by operation"""
    move = int(move_idx)
    if move < 0 or move > 63:
        return move
    return int(_ROTATE_MOVE_TABLE[int(op)][move])

def _legal_moves_mask_raw(player_board, opp_board):
    """Get legal moves mask using C++ engine or Numba fallback"""
    if cpp_engine is not None:
        try:
            return int(cpp_engine.get_legal_moves(int(player_board), int(opp_board)))
        except Exception:
            pass
    return int(get_legal_moves(np.uint64(player_board), np.uint64(opp_board)))

def _apply_move_pair_raw(player_board, opp_board, move_idx):
    """Apply a move and return both boards"""
    if cpp_engine is not None:
        try:
            next_player, next_opp = cpp_engine.apply_move(int(player_board), int(opp_board), int(move_idx))
            return (int(next_player), int(next_opp))
        except Exception:
            pass
    player_board = np.uint64(player_board)
    opp_board = np.uint64(opp_board)
    move_idx = int(move_idx)
    flip = np.uint64(get_flip(player_board, opp_board, np.int64(move_idx)))
    next_player = (player_board | np.uint64(1) << np.uint64(move_idx) | flip) & _FULL_MASK
    next_opp = (opp_board ^ flip) & _FULL_MASK
    return (int(next_player), int(next_opp))

class EgaroucidOpeningBook:
    """Egaroucid .egbk3 opening book loader with caching"""

    def __init__(self, book_path, cache_path=None):
        self.book_path = os.path.abspath(book_path)
        self.cache_path = cache_path or _EGAROUCID_CACHE_PATH
        self.keys_p = np.zeros(0, dtype=np.uint64)
        self.keys_o = np.zeros(0, dtype=np.uint64)
        self.values = np.zeros(0, dtype=np.int16)
        self.levels = np.zeros(0, dtype=np.int8)
        self.leaf_moves = np.full(0, -1, dtype=np.int16)
        self.state_count = 0
        self._move_cache = {}
        self._load()

    def __len__(self):
        return int(self.state_count)

    def __bool__(self):
        return self.state_count > 0

    def _get_cache_meta(self):
        """Get metadata for cache validation"""
        if os.path.exists(self.book_path):
            stat = os.stat(self.book_path)
            return np.array([1, int(stat.st_mtime_ns), int(stat.st_size)], dtype=np.int64)
        return None

    def _load_cache(self):
        """Load from cache file if valid"""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                meta = data['meta']
                cache_meta = self._get_cache_meta()
                if cache_meta is not None and (meta.shape != (3,) or not np.array_equal(meta.astype(np.int64), cache_meta)):
                    return False
                self.keys_p = data['p'].astype(np.uint64, copy=False)
                self.keys_o = data['o'].astype(np.uint64, copy=False)
                self.values = data['value'].astype(np.int16, copy=False)
                self.levels = data['level'].astype(np.int8, copy=False)
                self.leaf_moves = data['leaf_move'].astype(np.int16, copy=False)
        except Exception as e:
            print(f'Cache load error: {e}')
            return False
        self.state_count = int(self.keys_p.size)
        self._move_cache.clear()
        return self.state_count > 0

    def _save_cache(self):
        """Save cache to npz file"""
        if not self.cache_path or self.state_count <= 0:
            return
        try:
            np.savez(self.cache_path, meta=self._get_cache_meta(), p=self.keys_p, o=self.keys_o, value=self.values, level=self.levels, leaf_move=self.leaf_moves)
        except Exception:
            pass

    def _load(self):
        """Load book from cache or egbk3 file"""
        if self._load_cache():
            return
        self._load_egbk3()
        self._save_cache()

    def _load_egbk3(self):
        """Load Egaroucid .egbk3 book file"""
        with open(self.book_path, 'rb') as f:
            header = f.read(14)
            if len(header) < 14:
                raise ValueError('Egaroucid book header is truncated')
            if header[:9] != b'DICUORAGE':
                raise ValueError('Not an Egaroucid .egbk3 book')
            if header[9] != 3:
                raise ValueError(f'Unsupported Egaroucid book version: {header[9]}')
            n_boards = int.from_bytes(header[10:14], byteorder='little', signed=False)
            records = np.fromfile(f, dtype=_EGAROUCID_RECORD_DTYPE, count=n_boards)
        if int(records.size) != n_boards:
            raise ValueError('Egaroucid book data is truncated')
        valid_mask = (records['value'] >= -64) & (records['value'] <= 64) & (records['player'] & records['opponent'] == 0)
        records = records[valid_mask]
        if records.size == 0:
            self.keys_p = np.zeros(0, dtype=np.uint64)
            self.keys_o = np.zeros(0, dtype=np.uint64)
            self.values = np.zeros(0, dtype=np.int16)
            self.levels = np.zeros(0, dtype=np.int8)
            self.leaf_moves = np.full(0, -1, dtype=np.int16)
            self.state_count = 0
            return
        order = np.lexsort((records['opponent'], records['player']))
        keys_p = records['player'][order]
        keys_o = records['opponent'][order]
        values = records['value'][order].astype(np.int16, copy=False)
        levels = records['level'][order].astype(np.int8, copy=False)
        leaf_moves = records['leaf_move'][order].astype(np.int16, copy=False)
        self.keys_p, self.keys_o, self.values, self.levels, self.leaf_moves = self._dedupe_sorted(keys_p, keys_o, values, levels, leaf_moves)
        self.state_count = int(self.keys_p.size)
        self._move_cache.clear()

    @staticmethod
    def _dedupe_sorted(keys_p, keys_o, values, levels, leaf_moves):
        """Remove duplicate records, keeping best level"""
        n = int(keys_p.size)
        keep = np.empty(n, dtype=np.int64)
        write = 0
        i = 0
        while i < n:
            best = i
            best_level = int(levels[i])
            j = i + 1
            while j < n and keys_p[j] == keys_p[i] and (keys_o[j] == keys_o[i]):
                if int(levels[j]) >= best_level:
                    best = j
                    best_level = int(levels[j])
                j += 1
            keep[write] = best
            write += 1
            i = j
        keep = keep[:write]
        return (keys_p[keep], keys_o[keep], values[keep], levels[keep], leaf_moves[keep])

    def _find_index_exact(self, player_board, opp_board):
        """Binary search for exact board state"""
        if self.state_count <= 0:
            return -1
        player_board = np.uint64(player_board)
        opp_board = np.uint64(opp_board)
        left = int(np.searchsorted(self.keys_p, player_board, side='left'))
        right = int(np.searchsorted(self.keys_p, player_board, side='right'))
        if left >= right:
            return -1
        rel = int(np.searchsorted(self.keys_o[left:right], opp_board, side='left'))
        idx = left + rel
        if idx < right and int(self.keys_o[idx]) == int(opp_board):
            return idx
        return -1

    def _lookup_value_rotated(self, player_board, opp_board):
        """Look up board value considering rotations - searches both player orientations"""
        for op in range(8):
            rotated_player = int(get_rotated_bitboard(np.uint64(player_board), op))
            rotated_opp = int(get_rotated_bitboard(np.uint64(opp_board), op))
            idx = self._find_index_exact(rotated_player, rotated_opp)
            if idx >= 0:
                return int(self.values[idx])
        for op in range(8):
            rotated_player = int(get_rotated_bitboard(np.uint64(opp_board), op))
            rotated_opp = int(get_rotated_bitboard(np.uint64(player_board), op))
            idx = self._find_index_exact(rotated_player, rotated_opp)
            if idx >= 0:
                return -int(self.values[idx])
        return None

    def _score_child(self, player_board, opp_board, move_idx):
        next_player, next_opp = _apply_move_pair_raw(player_board, opp_board, move_idx)
        child_player = next_opp
        child_opp = next_player
        child_legal = _legal_moves_mask_raw(child_player, child_opp)
        if child_legal == 0:
            passed_player = child_opp
            passed_opp = child_player
            passed_legal = _legal_moves_mask_raw(passed_player, passed_opp)
            if passed_legal == 0:
                end_value = self._lookup_value_rotated(child_player, child_opp)
                if end_value is not None:
                    return -int(end_value)
            passed_value = self._lookup_value_rotated(passed_player, passed_opp)
            if passed_value is not None:
                return int(passed_value)
            return None
        child_value = self._lookup_value_rotated(child_player, child_opp)
        if child_value is None:
            return None
        return -int(child_value)

    def _resolve_rotated_state(self, player_board, opp_board):
        cache_key = (int(player_board), int(opp_board))
        cached = self._move_cache.get(cache_key)
        if cached is not None:
            return cached
        legal_mask = _legal_moves_mask_raw(player_board, opp_board)
        candidates = []
        remaining = legal_mask
        while remaining:
            move_idx = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            score = self._score_child(player_board, opp_board, move_idx)
            if score is not None:
                candidates.append((int(move_idx), int(score)))
        if not candidates:
            idx = self._find_index_exact(player_board, opp_board)
            if idx >= 0:
                leaf_move = int(self.leaf_moves[idx])
                if 0 <= leaf_move < 64 and legal_mask >> leaf_move & 1:
                    candidates.append((leaf_move, int(self.values[idx])))
        self._move_cache[cache_key] = candidates
        return candidates

    def get_prior_info(self, player_board, opp_board, legal_moves):
        legal_set = {int(move) for move in legal_moves}
        if self.state_count <= 0 or not legal_set:
            return ({}, None, False)
        found_state = False
        best_prior_map = {}
        best_primary_move = None
        orientations = [(player_board, opp_board, 1), (opp_board, player_board, -1)]
        for p_board, o_board, sign in orientations:
            for op in range(8):
                rotated_player = int(get_rotated_bitboard(np.uint64(p_board), op))
                rotated_opp = int(get_rotated_bitboard(np.uint64(o_board), op))
                if self._find_index_exact(rotated_player, rotated_opp) < 0:
                    continue
                found_state = True
                candidates = self._resolve_rotated_state(rotated_player, rotated_opp)
                if not candidates:
                    continue
                best_score = max((score for _, score in candidates))
                best_moves = [move_idx for move_idx, score in candidates if score == best_score]
                prior_map = {}
                for move_idx in best_moves:
                    move = int(unrotate_move(int(move_idx), op))
                    if move in legal_set:
                        prior_map[move] = 1.0
                if not prior_map:
                    continue
                norm = 1.0 / float(len(prior_map))
                for move in list(prior_map.keys()):
                    prior_map[move] = norm
                primary_move = max(prior_map.items(), key=lambda item: item[1])[0]
                if len(prior_map) > len(best_prior_map):
                    best_prior_map = prior_map
                    best_primary_move = primary_move
                elif len(prior_map) == len(best_prior_map) and best_primary_move is None:
                    best_prior_map = prior_map
                    best_primary_move = primary_move
        if best_prior_map:
            return (best_prior_map, best_primary_move, True)
        return ({}, None, found_state)

class UltimateOthello(OthelloSearchMixin):

    def __init__(self, r):
        self.rt = r
        self.ui_thread = threading.current_thread()
        self.ui_queue = queue.Queue()
        self.game_id = 0
        self.B = np.uint64(34628173824)
        self.W = np.uint64(68853694464)
        self.tn = 1
        self.bc = '#2e7d32'
        self.is_thinking = False
        self.running = True
        self.btn_pass = None
        self.last_win_rate = 50.0
        self.last_mcts_win_rate = None
        self._last_ab_completed_depth = 2
        self.blend_history = []
        self.pending_blend_samples = []
        self.win_rate_history = []
        self.activation_snapshot = None
        self.last_activation_key = None
        self.module_activity = {}
        self.connection_activity = {}
        self.ponder_thread = None
        self.ponder_mcts_thread = None
        self.ponder_ab_thread = None
        self.ponder_token = 0
        self.ponder_active_token = -1
        self.ponder_sf = np.zeros(1, dtype=np.uint8)
        self.ponder_cache = {}
        self.ponder_lock = threading.Lock()
        self.exact_cache = {}
        self.mcts_sim_history = []
        self.position_history = []
        self.history_view_index = 0
        if not core._NUMBA_WARMED_UP:
            print('Numba kernelを準備中です...')
            warmed = ensure_numba_warmup()
            if warmed:
                print('Numba kernelの準備が完了しました。')
        self.sp_var = tk.BooleanVar(value=True)
        self.board_only_var = tk.BooleanVar(value=False)
        self.use_mcts = tk.BooleanVar(value=True)
        self.use_nn = True
        self.auto_battle = False
        self.game_over_shown = False
        self.search_mode = 'hybrid'
        self.light_mode = False
        self.super_light_mode = False
        self.use_cpp_engine = cpp_engine is not None
        self.use_tt_resume = True
        self.opening_book_rate = 1.0
        self.time_limit_sec = 5.0
        self.auto_time = True
        self.auto_mode_type = 'normal'
        self.readout_empty_threshold = 22
        self.exact_auto = True
        self.mcts_batch_size = 256
        self.ab_time_limit_ms = 5000
        self.stop_flag = np.zeros(1, dtype=np.uint8)
        self.nn_infer_batch = []
        self.rt.update_idletasks()
        nn_available = self._detect_nn_runtime_available()
        nn_reason = ''
        if not nn_available:
            if core.ONNX_AVAILABLE and os.path.exists(core.ONNX_MODEL_PATH):
                nn_reason = 'PyTorch はありませんが ONNX モデルも見つかりませんでした。'
            elif torch is None:
                nn_reason = 'PyTorch / ONNX の推論環境が未設定です。'
            else:
                nn_reason = '利用可能な NN モデルが見つかりません。'
        settings_dialog = StartupSettingsDialog(self.rt, cpp_engine is not None, nn_available=nn_available, nn_reason=nn_reason)
        settings = settings_dialog.result or {
            'use_cpp': cpp_engine is not None, 
            'use_nn': nn_available, 
            'use_mcts': nn_available, 
            'search_mode': 'hybrid' if nn_available else 'ab_only', 
            'use_mcts_only': False, 
            'mcts_influence': 50, 
            'use_tt': True, 
            'book_usage': 85, 
            'time_limit': 5.0, 
            'auto_time': True, 
            'auto_mode_type': 'normal', 
            'player_color': 'black', 
            'use_pondering': True,
            # 枝刈り設定（デフォルト）
            'pruning_enabled': True,
            'mcts_pruning_enabled': False,
            'mcts_pruning_time': 2.0,
            'mcts_pruning_branches': 3,
            'ab_pruning_time': 3.0,
            'traditional_pruning_enabled': True,
            'multi_cut_enabled': True,
            'multi_cut_threshold': 3,
            'multi_cut_depth': 8
        }
        self.use_cpp_engine = settings['use_cpp']
        self.use_nn = settings['use_nn'] and nn_available
        self.use_mcts.set(settings['use_mcts'])
        self.search_mode = settings.get('search_mode', 'hybrid')
        self.auto_time = settings.get('auto_time', True)
        self.use_mcts_enabled = self.use_nn and self.search_mode != 'ab_only'
        self.use_mcts_only = self.use_nn and self.search_mode == 'mcts_only'
        self.mcts_influence_pct = 0 if self.search_mode == 'ab_only' else 100 if self.search_mode == 'mcts_only' else 50
        self.auto_time = settings.get('auto_time', False)
        self.time_limit_sec = settings.get('time_limit', 5.0)
        self.auto_mode_type = 'normal'
        self.use_pondering = settings.get('use_pondering', True)
        self.use_book = settings.get('use_book', True)
        book_source = settings.get('book_source', 'egaroucid')
        book_usage_ratio = max(0.0, min(1.0, float(settings.get('book_usage', 85)) / 100.0))
        
        # 枝刈り設定を読み込み
        self.pruning_enabled = settings.get('pruning_enabled', False)
        self.mcts_pruning_enabled = settings.get('mcts_pruning_enabled', False)
        self.mcts_pruning_time = settings.get('mcts_pruning_time', 2.0)
        self.mcts_pruning_branches = settings.get('mcts_pruning_branches', 3)
        self.ab_pruning_time = settings.get('ab_pruning_time', 3.0)
        self.traditional_pruning_enabled = settings.get('traditional_pruning_enabled', True)
        self.multi_cut_enabled = settings.get('multi_cut_enabled', False)
        self.multi_cut_threshold = settings.get('multi_cut_threshold', 3)
        self.multi_cut_depth = settings.get('multi_cut_depth', 8)
        if not self.use_book:
            self.opening_book_rate = 0.0
        elif book_source == 'json':
            self.opening_book_rate = book_usage_ratio
        else:
            self.opening_book_rate = 1.0
        self.board_only_mode = False
        self.readout_empty_threshold = int(settings.get('readout_empty', 22))
        self.exact_auto = True
        self.ui_scale = 1.0
        self.cell_size = int(60 * self.ui_scale)
        self.board_size = self.cell_size * 8
        self.graph_width = 700
        self.graph_height = 190
        self.W_ary = np.round(np.array(inital_weights, dtype=np.float64), 3)
        self.log(f'{WEIGHTS_PATH}')
        if os.path.exists(WEIGHTS_PATH) and USE_WEIGHT:
            try:
                with open(WEIGHTS_PATH, 'r') as f:
                    w = json.load(f)
                    self.W_ary = np.round(np.array([float(x) for x in w + [0.0] * (GENE_LEN - len(w))], dtype=np.float64), 3)
                self.log(f'遺伝子ファイルを読み込みました。遺伝子長: {len(self.W_ary)}')
            except Exception:
                self.log('遺伝子ファイルの読み込みに失敗しました。デフォルト重みを使用します。')
        else:
            self.log('遺伝子ファイルが見つからないか無効です。デフォルト重みを使用します。')
        self.order_map = self.W_ary[0:64]
        self.weights_list = self.W_ary.tolist()
        self.order_map_list = self.order_map.tolist()
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                cpp_engine.set_eval_data(self.W_ary, self.order_map)
                self.log('C++ 評価データを転送しました。')
            except Exception:
                self.log('C++ 評価データを転送できないため、python 評価に切り替えます。')
                self.use_cpp_engine = False
        self.nn_model = None
        if self.use_nn:
            self.mark_modules_active('NN-IN', 'TRUNK', 'POLICY', 'VALUE')
            self.mark_connections_active(('BOARD', 'NN-IN'), ('NN-IN', 'TRUNK'), ('TRUNK', 'POLICY'), ('TRUNK', 'VALUE'))
            self.log(f'{BEST_MODEL_PATH} | device={DEVICE_STR}')
            if torch is not None:
                try:
                    self.nn_model = OthelloNet().to(DEVICE)
                    if os.path.exists(BEST_MODEL_PATH):
                        self.nn_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
                    else:
                        self.log('ニューラルネットワークモデルが見つからないため、未学習初期値を使用します。')
                    self.nn_model.eval()
                    try:
                        d_st = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device=DEVICE)
                        with torch.inference_mode():
                            if DEVICE_STR == 'cuda':
                                with torch.amp.autocast('cuda', dtype=torch.float16):
                                    _ = self.nn_model(d_st)
                            else:
                                _ = self.nn_model(d_st)
                        self.log('ニューラルネットワークのウォームアップが完了しました。')
                    except Exception:
                        self.log('ニューラルネットワークのウォームアップをスキップしました。')
                except Exception as e:
                    self.nn_model = None
                    self.log(f'PyTorch NN の初期化に失敗したため ONNX/αβ にフォールバックします: {e}')
            else:
                self.log('PyTorch が見つからないため、NN は ONNX が使える場合のみ有効です。')
            if self.nn_model is None and core.is_using_onnx():
                self.log('ONNX 推論を使用します。')
            elif self.nn_model is None:
                self.use_nn = False
                self.use_mcts.set(False)
                self.use_mcts_enabled = False
                self.use_mcts_only = False
                self.search_mode = 'ab_only'
                self.mcts_influence_pct = 0
                self.log('利用可能な NN 推論環境がないため、αβ のみで動作します。')
        else:
            self.use_mcts.set(False)
            self.use_mcts_enabled = False
            self.search_mode = 'ab_only'
            self.mcts_influence_pct = 0
            self.log('ニューラルネットワークは無効です。')
        self.opening_book = None
        npz_path = _EGAROUCID_CACHE_PATH
        egbk3_path = _EGAROUCID_BOOK_PATH
        if not self.use_book:
            self.log('定石を使用しません。')
        elif os.path.exists(npz_path):
            try:
                self.log(f'npz キャッシュから定石を読み込み中: {npz_path}')
                self.opening_book = EgaroucidOpeningBook.__new__(EgaroucidOpeningBook)
                self.opening_book.book_path = egbk3_path
                self.opening_book.cache_path = npz_path
                self.opening_book.keys_p = np.zeros(0, dtype=np.uint64)
                self.opening_book.keys_o = np.zeros(0, dtype=np.uint64)
                self.opening_book.values = np.zeros(0, dtype=np.int16)
                self.opening_book.levels = np.zeros(0, dtype=np.int8)
                self.opening_book.leaf_moves = np.full(0, -1, dtype=np.int16)
                self.opening_book.state_count = 0
                self.opening_book._move_cache = {}
                if self.opening_book._load_cache():
                    self.mark_modules_active('BOOK')
                    self.mark_connections_active(('BOARD', 'BOOK'))
                    self.log(f'[OK] Egaroucid npz キャッシュ読み込み完了: 局面数 {len(self.opening_book)}')
                else:
                    self.log('[WARN] npz キャッシュが無効です。')
                    self.opening_book = None
            except Exception as e:
                self.log(f'[ERROR] npz キャッシュの読み込み失敗: {e}')
                self.opening_book = None
        if self.opening_book is None and os.path.exists(egbk3_path):
            try:
                self.log(f'egbk3 ファイルから定石を読み込み中: {egbk3_path}')
                self.opening_book = EgaroucidOpeningBook(egbk3_path)
                if self.opening_book:
                    self.mark_modules_active('BOOK')
                    self.mark_connections_active(('BOARD', 'BOOK'))
                    self.log(f'[OK] Egaroucid egbk3 読み込み完了: 局面数 {len(self.opening_book)}')
                else:
                    self.log('[WARN] egbk3 ファイルが空です。')
                    self.opening_book = None
            except Exception as e:
                self.log(f'[ERROR] egbk3 ファイルの読み込み失敗: {e}')
                self.opening_book = None
        if self.opening_book is None and (not self.use_book):
            pass
        elif self.opening_book is None:
            self.log('[WARN] 定石データが利用できません。定石なしで動作します。')
        self.tk_arr = np.zeros(TT_SIZE * 2, dtype=np.uint64)
        self.ti_arr = np.zeros((TT_SIZE * 2, 4), dtype=np.float64)
        self.sf_arr = np.zeros(1, dtype=np.uint8)
        time_str = f'max {self.time_limit_sec:g}s'
        mode_str = {'hybrid': 'αβ+MCTS', 'ab_only': 'αβのみ', 'mcts_only': 'MCTSのみ'}.get(self.search_mode, self.search_mode)
        self.log(f"探索設定: TT size={TT_SIZE * 2}, C++={('ON' if self.use_cpp_engine else 'OFF')}, NN={('ON' if self.use_nn else 'OFF')}, モード={mode_str}, book={int(self.opening_book_rate * 100)}%, TT reuse={('ON' if self.use_tt_resume else 'OFF')}, time limit={time_str}, exact solve starts=always-auto, pondering={('ON' if self.use_pondering else 'OFF')}")
        self.rt.title(f'Sesamum')
        self.rt.protocol('WM_DELETE_WINDOW', self.on_close)
        self.rt.bind('<KeyPress-z>', self.on_viewer_step_back)
        self.rt.bind('<KeyPress-Z>', self.on_viewer_step_back)
        self.rt.bind('<KeyPress-y>', self.on_viewer_step_forward)
        self.rt.bind('<KeyPress-Y>', self.on_viewer_step_forward)
        mb = tk.Menu(r)
        um = tk.Menu(mb, tearoff=0)
        um.add_command(label='盤面カラー', command=self.s_col)
        um.add_checkbutton(label='確率表示', variable=self.sp_var, command=self.drw)
        um.add_checkbutton(label='盤面のみ表示', variable=self.board_only_var, command=self.toggle_board_only_mode)
        self.auto_battle_var = tk.BooleanVar(value=self.auto_battle)
        um.add_checkbutton(label='自動対戦モード', variable=self.auto_battle_var, command=self.toggle_auto_battle_mode)
        um.add_separator()
        um.add_command(label='先後交代', command=self.c_col)
        um.add_separator()
        um.add_command(label='新ゲーム', command=self.ng)
        mb.add_cascade(label='設定', menu=um)
        r.config(menu=mb)
        self.rt.minsize(1450, 760)
        self.main_frame = tk.Frame(r, bg='#eef2f7')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        if self.board_only_mode:
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.board_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.board_frame.grid(row=0, column=0, sticky='nsew')
        else:
            self.main_frame.grid_columnconfigure(0, weight=0)
            self.main_frame.grid_columnconfigure(1, weight=1)
            self.main_frame.grid_columnconfigure(2, weight=0)
            self.main_frame.grid_rowconfigure(0, weight=0)
            self.main_frame.grid_rowconfigure(1, weight=0)
            self.left_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.left_frame.grid(row=0, column=0, sticky='nw')
            self.graphs_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.graphs_frame.grid(row=0, column=1, sticky='nw', padx=(12, 0))
            self.activation_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.activation_frame.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(12, 0))
            self.log_host_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.log_host_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 0))
        if self.board_only_mode:
            self.cv = tk.Canvas(self.board_frame, width=self.board_size, height=self.board_size, bg=self.bc)
            self.cv.pack(expand=True)
        else:
            self.cv = tk.Canvas(self.left_frame, width=self.board_size, height=self.board_size, bg=self.bc)
            self.cv.pack()
        self.cv.bind('<Button-1>', self.clk)
        if not self.board_only_mode:
            tk.Label(self.graphs_frame, text='勝率グラフ', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.win_graph = tk.Canvas(self.graphs_frame, width=self.graph_width, height=self.graph_height, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.win_graph.pack()
            tk.Label(self.graphs_frame, text='判定割合', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w', pady=(8, 0))
            self.diff_graph = tk.Canvas(self.graphs_frame, width=self.graph_width, height=self.graph_height + 50, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.diff_graph.pack()
            tk.Label(self.activation_frame, text='発火図', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.activation_graph = tk.Canvas(self.activation_frame, width=300, height=self.graph_height + 200, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.activation_graph.pack()
            tk.Label(self.activation_frame, text='接続図', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w', pady=(8, 0))
            self.connection_graph = tk.Canvas(self.activation_frame, width=300, height=self.graph_height + 100, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.connection_graph.pack()
            tk.Label(self.log_host_frame, text='ログ', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.log_frame = tk.Frame(self.log_host_frame, height=220, bg='#eef2f7')
            self.log_frame.pack(fill='x', pady=(0, 0))
            self.log_frame.pack_propagate(False)
            self.log_text = tk.Text(self.log_frame, width=90, height=5, bg='#ffffff', fg='#111111', insertbackground='#111111')
            self.log_text.pack(fill='both', expand=True)
            self.log_text.config(state='disabled')
            self.log_text.tag_config('green', foreground='#00aa00')
            self.log_text.tag_config('blue', foreground='#0066cc')
        self.rt.after(16, self.process_ui_queue)
        if self.board_only_mode:
            self.rt.minsize(self.board_size + 40, self.board_size + 40)
        else:
            self.rt.minsize(1450, 760)
        self.rt.update()
        c = settings['player_color']
        if c == 'white':
            self.hc = -1
            self.ac = 1
        else:
            self.hc = 1
            self.ac = -1
        self.reset_position_history()
        self.reset_graph_history()
        self.redraw_graphs()
        self.drw()
        self.chk()

    def _detect_nn_runtime_available(self):
        if torch is not None:
            return True
        return bool(core.ONNX_AVAILABLE and os.path.exists(core.ONNX_MODEL_PATH))

    def get_exact_base_threshold(self):
        if not self.exact_auto:
            return int(self.readout_empty_threshold)
        base_time = float(self.time_limit_sec)
        if base_time <= 1.0:
            return 22
        if base_time <= 5.0:
            return 26
        if base_time <= 10.0:
            return 28
        return 30

    def get_exact_gap_budget(self, legal_count, time_limit):
        budget = 7
        if time_limit >= 10.0:
            budget += 4
        elif time_limit >= 5.0:
            budget += 3
        elif time_limit >= 2.0:
            budget += 2
        if legal_count <= 4:
            budget += 4
        elif legal_count <= 8:
            budget += 3
        elif legal_count <= 12:
            budget += 2
        elif legal_count >= 16:
            budget -= 1
        return max(3, budget)

    def get_endgame_solver_threshold(self, legal_count, time_limit):
        limit = float(time_limit)
        if limit <= 1.0:
            threshold = 24
        elif limit <= 5.0:
            threshold = 28
        elif limit <= 10.0:
            threshold = 29
        else:
            threshold = 30
        if legal_count <= 6:
            threshold += 2
        elif legal_count <= 10:
            threshold += 1
        elif legal_count >= 16:
            threshold -= 2
        elif legal_count >= 13:
            threshold -= 1
        return max(22, min(32, threshold))

    def get_endgame_solver_time_limit_ms(self, empty, legal_count, time_limit):
        base_ms = float(time_limit) * 1000.0
        bonus = 0.0
        if empty <= 20:
            bonus += 2200.0
        elif empty <= 24:
            bonus += 1500.0
        elif empty <= 28:
            bonus += 1000.0
        elif empty <= 32:
            bonus += 700.0
        if legal_count <= 4:
            bonus += 1400.0
        elif legal_count <= 8:
            bonus += 1100.0
        elif legal_count <= 12:
            bonus += 700.0
        return int(max(1800.0, min(14000.0, base_ms + bonus)))

    def get_auto_time_limit(self, empty, legal_count, mvs):
        if not self.auto_time:
            return self.time_limit_sec
        if self.time_limit_sec == 0.5:
            return 0.5
        elif self.time_limit_sec == 1.0:
            base_time = 0.9
            max_time = 1.0
        elif self.time_limit_sec == 5.0:
            base_time = 3.8
            max_time = 5.0
        elif self.time_limit_sec == 10.0:
            base_time = 8.0
            max_time = 10.0
        elif self.time_limit_sec == 30.0:
            base_time = 18.0
            max_time = 30.0
        else:
            base_time = self.time_limit_sec * 0.78
            max_time = self.time_limit_sec
        if legal_count <= 4:
            move_factor = 1.30
        elif legal_count <= 8:
            move_factor = 1.18
        elif legal_count <= 12:
            move_factor = 1.05
        elif legal_count <= 16:
            move_factor = 0.96
        else:
            move_factor = 0.90
        if empty >= 44:
            phase_factor = 0.90
        elif empty >= 32:
            phase_factor = 1.0
        elif empty >= 22:
            phase_factor = 1.08
        elif empty >= 14:
            phase_factor = 1.18
        else:
            phase_factor = 1.28
        final_time = base_time * move_factor * phase_factor
        return max(base_time * 0.65, min(max_time, final_time))

    def get_exact_cached_move(self, aB, oB, tn):
        if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'probe_exact_cache'):
            try:
                empty = 64 - int((aB | oB).bit_count())
                cache_result = cpp_engine.probe_exact_cache(int(aB), int(oB), int(tn), empty)
                if cache_result.get('found') and cache_result.get('is_resolved'):
                    best_move = cache_result.get('best_move', -1)
                    if best_move >= 0:
                        return best_move
            except Exception:
                pass
        key = f'{int(aB)}_{int(oB)}_{int(tn)}'
        cached = self.exact_cache.get(key)
        if cached:
            moves = cached.get('moves', [])
            if moves:
                return moves[0][0]
        return None

    def cache_exact_result(self, aB, oB, tn, ab_val_res, ab_res):
        key = f'{int(aB)}_{int(oB)}_{int(tn)}'
        moves = []
        move_values = {}
        move_win_rates = {}
        for m, v in ab_val_res.items():
            wr = ab_res.get(m, 50.0)
            moves.append((m, v, wr))
            move_values[int(m)] = float(v)
            move_win_rates[int(m)] = float(wr)
        moves.sort(key=lambda x: x[2], reverse=True)
        self.exact_cache[key] = {'moves': moves, 'timestamp': time.time()}
        if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'store_exact_cache'):
            try:
                empty = 64 - int((aB | oB).bit_count())
                best_move = moves[0][0] if moves else -1
                best_value = moves[0][1] if moves else 0.0
                best_win_rate = moves[0][2] if moves else 50.0
                cpp_engine.store_exact_cache(int(aB), int(oB), int(tn), int(best_move), float(best_value), float(best_win_rate), move_values, move_win_rates, int(empty))
            except Exception:
                pass
        return moves[0][0] if moves else None

    def count_empty_regions(self, empty_mask):
        empty_mask_int = int(empty_mask) & int(_FULL_MASK)
        remaining = empty_mask_int
        regions = 0
        small_regions = 0
        while remaining:
            region = remaining & -remaining
            frontier = region
            while frontier:
                expanded = int(neighbor_union(np.uint64(frontier))) & empty_mask_int & ~region
                region |= expanded
                frontier = expanded
            if region.bit_count() <= 2:
                small_regions += 1
            remaining &= ~region
            regions += 1
        return (regions, small_regions)

    def get_search_time_profile(self, empty, legal_count, time_limit, is_exact, ponder_has_mcts):
        if is_exact:
            return {'use_mcts': False, 'ab_delay': 0.0, 'ab_budget': None, 'mcts_priority': 0.0, 'max_depth': 60}
        if not (self.use_mcts_enabled and self.use_nn and (self.nn_model is not None)):
            return {'use_mcts': False, 'ab_delay': 0.0, 'ab_budget': time_limit, 'mcts_priority': 0.0, 'max_depth': 60}
        
        opening_phase = min(1.0, max(0.0, (float(empty) - 18.0) / 26.0))
        branch_factor = min(1.0, max(0.0, (float(legal_count) - 3.0) / 10.0))
        ponder_bonus = 0.10 if ponder_has_mcts else 0.0
        gpu_bonus = 0.04 if DEVICE_STR == 'cuda' else 0.0
        mcts_priority = min(1.0, max(0.0, 0.10 + 0.32 * opening_phase + 0.22 * branch_factor + ponder_bonus + gpu_bonus))
        
        max_depth = 60
        
        if time_limit <= 1.5:
            ab_delay = time_limit * (0.03 * mcts_priority)
            ab_budget_ratio = 0.72 - 0.05 * mcts_priority
        elif time_limit <= 5.0:
            ab_delay = time_limit * (0.012 * mcts_priority)
            ab_budget_ratio = 0.84 - 0.05 * mcts_priority
        else:
            ab_delay = time_limit * (0.006 * mcts_priority)
            ab_budget_ratio = 0.90 - 0.04 * mcts_priority
        if empty <= 32:
            ab_delay *= 0.15
            ab_budget_ratio = max(ab_budget_ratio, 0.85)
        if empty <= 26:
            ab_delay = 0.0
            ab_budget_ratio = max(ab_budget_ratio, 0.93)
        if empty <= 20:
            ab_delay = 0.0
            ab_budget_ratio = max(ab_budget_ratio, 0.97)
        ab_delay = max(0.0, min(time_limit * 0.2, ab_delay))
        ab_budget = max(0.20, min(time_limit, time_limit * ab_budget_ratio))
        return {'use_mcts': True, 'ab_delay': ab_delay, 'ab_budget': ab_budget, 'mcts_priority': mcts_priority, 'max_depth': max_depth}

    def adjust_mcts_influence_auto(self, mcts_sim_count, empty):
        if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'calculate_mcts_influence'):
            try:
                is_auto = bool(self.auto_time)
                influence_ratio = cpp_engine.calculate_mcts_influence(int(mcts_sim_count), int(empty), is_auto)
                return int(influence_ratio * 100.0)
            except Exception as e:
                print(f'MCTS influence calculation error: {e}')
                pass
        
        if self.auto_time:
            time_limit = self.time_limit_sec
            if time_limit <= 1.0:
                base_influence = 68
            elif time_limit <= 3.0:
                base_influence = 58
            elif time_limit <= 10.0:
                base_influence = 48
            else:
                base_influence = 40
            if empty <= 16:
                base_influence -= 22
            elif empty <= 24:
                base_influence -= 14
            elif empty <= 32:
                base_influence -= 8
            elif empty <= 44:
                base_influence -= 2
            else:
                base_influence += 6
            if mcts_sim_count >= 300000:
                base_influence += 6
            elif mcts_sim_count >= 150000:
                base_influence += 3
            elif mcts_sim_count < 50000:
                base_influence -= 18
            elif mcts_sim_count < 100000:
                base_influence -= 10
            return max(10, min(80, base_influence))
        else:
            return self.mcts_influence_pct

    def should_use_opening_book(self, mvs, empty, ab_result=None, mcts_result=None, return_details=False):
        """動的定石使用判定 - AB/MCTSの一致度が高い場合だけ定石を使わない"""
        base_rate = float(self.opening_book_rate)
        if base_rate >= 0.999:
            details = {'rate': 1.0, 'reason': 'forced 100%'}
            return details if return_details else details['rate']
        use_book = True
        reasons = []
        ab_best_move = None
        ab_best_wr = None
        ab_second_wr = None
        mcts_best_move = None
        mcts_best_wr = None
        mcts_second_wr = None
        ab_moves = list(ab_result.get('moves', [])) if ab_result else []
        ab_win_rates = list(ab_result.get('win_rates', [])) if ab_result else []
        if len(ab_moves) >= 2 and len(ab_win_rates) >= 2:
            ab_pairs = sorted(zip(ab_moves, ab_win_rates), key=lambda x: x[1], reverse=True)
            ab_best_move, ab_best_wr = (int(ab_pairs[0][0]), float(ab_pairs[0][1]))
            ab_second_wr = float(ab_pairs[1][1])
        mcts_move_win_rates = dict(mcts_result.get('move_win_rates', {})) if mcts_result else {}
        if len(mcts_move_win_rates) >= 2:
            mcts_pairs = sorted(((int(move), float(rate)) for move, rate in mcts_move_win_rates.items()), key=lambda x: x[1], reverse=True)
            mcts_best_move, mcts_best_wr = mcts_pairs[0]
            mcts_second_wr = float(mcts_pairs[1][1])
        if ab_best_move is not None and mcts_best_move is not None and (ab_best_move == mcts_best_move):
            ab_gap = ab_best_wr - ab_second_wr if ab_second_wr is not None else 0.0
            mcts_gap = mcts_best_wr - mcts_second_wr if mcts_second_wr is not None else 0.0
            if ab_gap >= 5.0 and ab_best_wr >= 52.0 and (mcts_gap >= 6.0) and (mcts_best_wr >= 55.0):
                use_book = False
                reasons.append(f'AB/MCTS一致_強 (AB:{ab_gap:.1f}pt@{ab_best_wr:.0f}%, MCTS:{mcts_gap:.1f}pt@{mcts_best_wr:.0f}%)')
            else:
                reasons.append(f'AB/MCTS一致_弱 (AB:{ab_gap:.1f}pt@{ab_best_wr:.0f}%, MCTS:{mcts_gap:.1f}pt@{mcts_best_wr:.0f}%)')
        adjusted_rate = 1.0 if use_book else 0.0
        details = {'rate': min(1.0, max(0.0, adjusted_rate)), 'reason': ' / '.join(reasons) if reasons else 'standard (AB/MCTS not analyzed or mismatch)'}
        return details if return_details else details['rate']

    def get_book_prior_info(self, aB, oB, legal_moves, dynamic_book_rate):
        """定石から事前確率情報を取得"""
        legal_set = {int(m) for m in legal_moves}
        prior_map = {}
        primary_move = None
        book_roll_used = False
        if not self.opening_book or dynamic_book_rate <= 0.0 or (not legal_set):
            return (prior_map, primary_move, book_roll_used)
        if isinstance(self.opening_book, EgaroucidOpeningBook):
            return self.opening_book.get_prior_info(int(aB), int(oB), legal_moves)
        for swap in (False, True):
            p, o = (aB, oB) if not swap else (oB, aB)
            for op in range(8):
                rp = get_rotated_bitboard(p, op)
                ro = get_rotated_bitboard(o, op)
                state_key = f'{int(rp)}_{int(ro)}'
                if state_key not in self.opening_book:
                    continue
                book_roll_used = True
                book_entry = self.opening_book[state_key]
                if isinstance(book_entry, dict) and 'moves' in book_entry:
                    moves_list = list(book_entry['moves'])
                    probs_list = list(book_entry.get('probs', []))
                elif isinstance(book_entry, list):
                    moves_list = list(book_entry)
                    probs_list = []
                elif isinstance(book_entry, dict):
                    moves_list = [list(book_entry.values())[0]]
                    probs_list = []
                else:
                    moves_list = [int(book_entry)]
                    probs_list = []
                if not moves_list:
                    break
                if len(probs_list) != len(moves_list):
                    probs_list = [1.0] * len(moves_list)
                total_prob = float(sum((max(0.0, float(pv)) for pv in probs_list)))
                if total_prob <= 0.0:
                    probs_list = [1.0] * len(moves_list)
                    total_prob = float(len(moves_list))
                for bm_rot, prob in zip(moves_list, probs_list):
                    move = int(unrotate_move(int(bm_rot), op))
                    if move in legal_set:
                        normalized = float(prob) / total_prob
                        prior_map[move] = max(prior_map.get(move, 0.0), normalized)
                if prior_map:
                    primary_move = max(prior_map.items(), key=lambda x: x[1])[0]
                break
            if book_roll_used:
                break
        return (prior_map, primary_move, book_roll_used)

    def should_start_exact_early(self, aB, oB, empty, legal_count, start_depth=2, time_limit=None):
        base_threshold = int(self.get_exact_base_threshold())
        time_budget = float(self.time_limit_sec if time_limit is None else time_limit)
        if empty > base_threshold + 3:
            return False
        if empty > base_threshold and legal_count >= 12:
            return False
        gap_budget = self.get_exact_gap_budget(int(legal_count), time_budget)
        remaining_gap = max(0, int(empty) - max(2, int(start_depth)))
        if remaining_gap > gap_budget:
            return False
        if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'should_use_early_exact'):
            try:
                heuristic_exact = bool(cpp_engine.should_use_early_exact(int(aB), int(oB), int(empty), base_threshold))
                if heuristic_exact and legal_count >= 15 and remaining_gap >= max(5, gap_budget):
                    return False
                return heuristic_exact
            except Exception:
                pass
        # フォールバック: C++ヒューリスティックが使えない場合のみ簡易判定
        if empty <= min(20, base_threshold):
            return True
        if empty <= base_threshold:
            threshold = int(max(4, 11 + (base_threshold - empty)))
            if legal_count <= threshold:
                return True
        return False

    def update_title(self):
        t = f'Sesamum - αβ: {max(0.0, min(100.0, self.last_win_rate)):.1f}%'
        if self.use_mcts_enabled and self.last_mcts_win_rate is not None:
            t += f' | MCTS: {max(0.0, min(100.0, self.last_mcts_win_rate)):.1f}%'
        self.rt.title(t)

    def on_toggle_mcts(self):
        self.use_mcts_enabled = bool(self.use_mcts.get())
        self.update_title()

    def call_on_ui_thread(self, func, *args, **kwargs):
        if threading.current_thread() is self.ui_thread:
            return func(*args, **kwargs)
        self.ui_queue.put((func, args, kwargs))
        return None

    def process_ui_queue(self):
        try:
            while True:
                func, args, kwargs = self.ui_queue.get_nowait()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f'UI thread error: {type(e).__name__}: {e}', flush=True)
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        if self.running:
            self.rt.after(16, self.process_ui_queue)

    def append_log_message(self, message):
        if self.log_text is None or not hasattr(self, 'log_text'):
            return
        try:
            self.log_text.config(state='normal')
            has_green = '[GREEN]' in message and '[/GREEN]' in message
            has_blue = '[BLUE]' in message and '[/BLUE]' in message
            if has_green or has_blue:
                if has_green:
                    parts = message.split('[GREEN]')
                    for i, part in enumerate(parts):
                        if '[/GREEN]' in part:
                            green_part, rest = part.split('[/GREEN]', 1)
                            if green_part:
                                self.log_text.insert('end', green_part, 'green')
                            if rest:
                                if '[BLUE]' in rest and '[/BLUE]' in rest:
                                    blue_parts = rest.split('[BLUE]')
                                    for j, blue_part in enumerate(blue_parts):
                                        if '[/BLUE]' in blue_part:
                                            blue_content, blue_rest = blue_part.split('[/BLUE]', 1)
                                            if blue_content:
                                                self.log_text.insert('end', blue_content, 'blue')
                                            if blue_rest:
                                                self.log_text.insert('end', blue_rest)
                                        elif j > 0:
                                            self.log_text.insert('end', blue_part)
                                else:
                                    self.log_text.insert('end', rest)
                        elif i > 0:
                            self.log_text.insert('end', part)
                elif has_blue:
                    parts = message.split('[BLUE]')
                    for i, part in enumerate(parts):
                        if '[/BLUE]' in part:
                            blue_part, rest = part.split('[/BLUE]', 1)
                            if blue_part:
                                self.log_text.insert('end', blue_part, 'blue')
                            if rest:
                                self.log_text.insert('end', rest)
                        elif i > 0:
                            self.log_text.insert('end', part)
                self.log_text.insert('end', '\n')
            else:
                self.log_text.insert('end', message + '\n')
            self.log_text.see('end')
            self.log_text.config(state='disabled')
        except Exception:
            pass

    def log(self, message, mirror=True):
        if mirror:
            print(message, flush=True)
        if hasattr(self, 'log_text'):
            self.call_on_ui_thread(self.append_log_message, message)

    def log_section(self, title):
        self.log('')
        self.log(f'[{title}]')

    def format_move_label(self, move_idx):
        return f'({move_idx // 8 + 1},{move_idx % 8 + 1})'

    def format_top_moves(self, pairs, limit=5, score_fmt='{:.1f}%'):
        out = []
        for move, score in list(pairs)[:limit]:
            out.append(f'{self.format_move_label(int(move))} {score_fmt.format(float(score))}')
        return ' | '.join(out)

    def format_log_columns(self, parts, widths=None):
        cols = []
        for idx, part in enumerate(parts):
            text = str(part)
            width = None if widths is None or idx >= len(widths) else widths[idx]
            if width is not None:
                cols.append(text.ljust(width))
            else:
                cols.append(text)
        return ' | '.join(cols)

    def legal_moves_mask(self, player_board, opp_board):
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                return int(cpp_engine.get_legal_moves(int(player_board), int(opp_board)))
            except Exception:
                pass
        return int(get_legal_moves(player_board, opp_board))

    def evaluate_position(self, player_board, opp_board, mvs):
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                return float(cpp_engine.evaluate_board_cached(int(player_board), int(opp_board), int(mvs)))
            except Exception:
                pass
        return float(evaluate_board_full(player_board, opp_board, np.int64(mvs), self.W_ary))

    def reset_graph_history(self):
        self.win_rate_history = [(0, 50.0, 50.0)]
        self.blend_history = []
        self.pending_blend_samples = []

    def flush_blend_calibration_samples(self, result_value, disc_diff):
        if not self.pending_blend_samples:
            return
        try:
            with open(BLEND_CALIBRATION_LOG_PATH, 'a', encoding='utf-8') as fh:
                for sample in self.pending_blend_samples:
                    row = dict(sample)
                    row['result'] = int(result_value)
                    row['disc_diff'] = int(disc_diff)
                    row['timestamp'] = time.time()
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception:
            pass
        self.pending_blend_samples = []

    def push_graph_point(self, ply, ab_wr, mcts_wr):
        if self.win_rate_history and self.win_rate_history[-1][0] == ply:
            self.win_rate_history[-1] = (ply, ab_wr, mcts_wr)
        else:
            self.win_rate_history.append((ply, ab_wr, mcts_wr))
        self.redraw_graphs()

    def redraw_graphs(self):
        if not self.board_only_mode:
            if self.win_graph is not None:
                self.draw_line_graph(self.win_graph, self.win_rate_history, [(1, '#2563eb'), (2, '#dc2626')], 0.0, 100.0, ['αβ', 'MCTS'])
            if self.diff_graph is not None:
                self.draw_recent_blends(self.diff_graph)
            if hasattr(self, 'activation_graph') and self.activation_graph is not None:
                self.draw_activation_maps(self.activation_graph)
            if hasattr(self, 'connection_graph') and self.connection_graph is not None:
                self.draw_connection_graph(self.connection_graph)

    def draw_recent_blends(self, canvas):
        if canvas is None:
            return
        try:
            canvas.delete('all')
        except Exception:
            return
        self.module_activity['BLEND'] = time.time()
        w = int(canvas['width'])
        h = int(canvas['height'])
        canvas.create_rectangle(10, 10, w - 10, h - 10, outline='#cbd5e1', fill='#ffffff')
        if not self.blend_history:
            canvas.create_text(w // 2, h // 2, text='まだ着手はありません', fill='#64748b', font=('Arial', 11))
            return
        slot_w = 220
        gap = 8
        pie_size = 96
        latest_three = self.blend_history[-3:]
        start_x = w - 16 - slot_w
        y0 = 44
        for idx, info in enumerate(reversed(latest_three)):
            x0 = start_x - idx * (slot_w + gap)
            move_idx = info.get('move')
            ply = info.get('ply')
            ab_ratio = max(0.0, min(1.0, float(info.get('ab_weight', 1.0))))
            mcts_ratio = max(0.0, min(1.0, float(info.get('mcts_weight', 0.0))))
            pie_x0 = x0 + 16
            pie_y0 = y0 + 18
            pie_x1 = pie_x0 + pie_size
            pie_y1 = pie_y0 + pie_size
            start = 90.0
            extent_ab = -360.0 * ab_ratio
            extent_mcts = -360.0 * mcts_ratio
            canvas.create_oval(pie_x0, pie_y0, pie_x1, pie_y1, fill='#e2e8f0', outline='#cbd5e1')
            if ab_ratio > 0.0:
                canvas.create_arc(pie_x0, pie_y0, pie_x1, pie_y1, start=start, extent=extent_ab, fill='#2563eb', outline='#2563eb')
            if mcts_ratio > 0.0:
                canvas.create_arc(pie_x0, pie_y0, pie_x1, pie_y1, start=start + extent_ab, extent=extent_mcts, fill='#dc2626', outline='#dc2626')
            label = '最新' if idx == 0 else f'{idx}手前'
            if ply is not None:
                label = f'{int(ply)}手目'
            canvas.create_text(x0 + 12, y0 + 6, text=label, anchor='w', fill='#475569', font=('Arial', 9))
            if move_idx is not None:
                canvas.create_text(x0 + 16, pie_y1 + 18, text=f'({move_idx // 8 + 1},{move_idx % 8 + 1})', anchor='w', fill='#0f172a', font=('Arial', 11, 'bold'))
        legend_y = h - 28
        canvas.create_rectangle(24, legend_y - 6, 34, legend_y + 4, fill='#2563eb', outline='#2563eb')
        canvas.create_text(40, legend_y - 1, text='αβ', anchor='w', fill='#0f172a', font=('Arial', 9))
        canvas.create_rectangle(88, legend_y - 6, 98, legend_y + 4, fill='#dc2626', outline='#dc2626')
        canvas.create_text(104, legend_y - 1, text='MCTS', anchor='w', fill='#0f172a', font=('Arial', 9))
        for idx, info in enumerate(reversed(latest_three)):
            x0 = start_x - idx * (slot_w + gap)
            ab_ratio = max(0.0, min(1.0, float(info.get('ab_weight', 1.0))))
            mcts_ratio = max(0.0, min(1.0, float(info.get('mcts_weight', 0.0))))
            canvas.create_text(x0 + 124, y0 + 52, text=f'{ab_ratio * 100.0:.0f}%', anchor='w', fill='#2563eb', font=('Arial', 13, 'bold'))
            canvas.create_text(x0 + 124, y0 + 84, text=f'{mcts_ratio * 100.0:.0f}%', anchor='w', fill='#dc2626', font=('Arial', 13, 'bold'))

    def activation_color(self, value, max_value):
        denom = max(1e-06, float(max_value))
        t = max(0.0, min(1.0, float(value) / denom))
        r = int(20 + 235 * t)
        g = int(28 + 200 * t ** 0.8)
        b = int(40 + 90 * (1.0 - t))
        return f'#{r:02x}{g:02x}{b:02x}'

    def mark_modules_active(self, *names):
        now = time.time()
        for name in names:
            if name:
                self.module_activity[str(name)] = now
        if hasattr(self, 'connection_graph'):
            self.draw_connection_graph(self.connection_graph)

    def mark_connections_active(self, *edges):
        now = time.time()
        for edge in edges:
            if edge:
                self.connection_activity[tuple(edge)] = now
        if hasattr(self, 'connection_graph'):
            self.draw_connection_graph(self.connection_graph)

    def draw_connection_graph(self, canvas):
        if canvas is None:
            return
        try:
            canvas.delete('all')
        except Exception:
            return
        w = int(canvas['width'])
        h = int(canvas['height'])
        canvas.create_rectangle(10, 10, w - 10, h - 10, outline='#cbd5e1', fill='#ffffff')
        canvas.create_text(18, 24, text='関数の接続図', anchor='w', fill='#0f172a', font=('Arial', 10, 'bold'))
        canvas.create_text(18, 42, text='黒線=待機  赤線=直近にデータ通過', anchor='w', fill='#64748b', font=('Arial', 8))
        nodes = {'BOARD': (48, 82), 'BOOK': (48, 136), 'TT': (48, 190), 'NN-IN': (120, 82), 'TRUNK': (120, 136), 'POLICY': (120, 190), 'VALUE': (120, 244), 'αβ': (192, 82), 'MCTS': (192, 136), 'PONDER': (192, 190), 'BLEND': (192, 244)}
        edges = [('BOARD', 'BOOK'), ('BOARD', 'TT'), ('BOARD', 'NN-IN'), ('NN-IN', 'TRUNK'), ('TRUNK', 'POLICY'), ('TRUNK', 'VALUE'), ('POLICY', 'MCTS'), ('VALUE', 'MCTS'), ('TT', 'αβ'), ('BOOK', 'BLEND'), ('αβ', 'BLEND'), ('MCTS', 'BLEND'), ('PONDER', 'αβ'), ('PONDER', 'MCTS'), ('PONDER', 'TT')]
        now = time.time()
        for src, dst in edges:
            x0, y0 = nodes[src]
            x1, y1 = nodes[dst]
            age = max(0.0, now - float(self.connection_activity.get((src, dst), 0.0)))
            if age <= 1.6:
                fade = max(0.0, 1.0 - age / 1.6)
                red = int(120 + 135 * fade)
                color = f'#{red:02x}1010'
                width = 3
            else:
                color = '#111111'
                width = 2
            canvas.create_line(x0, y0, x1, y1, fill=color, width=width)
        for name, (x, y) in nodes.items():
            age = max(0.0, now - float(self.module_activity.get(name, 0.0)))
            if age <= 1.6:
                fade = max(0.0, 1.0 - age / 1.6)
                red = int(120 + 135 * fade)
                fill = f'#{red:02x}1010'
                outline = '#ff4d4d'
            else:
                fill = '#111111'
                outline = '#444444'
            canvas.create_oval(x - 18, y - 12, x + 18, y + 12, fill=fill, outline=outline, width=2)
            canvas.create_text(x, y, text=name, fill='#f8fafc', font=('Arial', 7, 'bold'))

    def draw_activation_heatmap(self, canvas, grid, x0, y0, cell, title, subtitle=''):
        grid = np.asarray(grid, dtype=np.float32)
        max_value = float(np.max(grid)) if grid.size else 1.0
        canvas.create_text(x0, y0 - 10, text=title, anchor='w', fill='#0f172a', font=('Arial', 9, 'bold'))
        if subtitle:
            canvas.create_text(x0 + 44, y0 - 10, text=subtitle, anchor='w', fill='#64748b', font=('Arial', 8))
        for r in range(8):
            for c in range(8):
                val = float(grid[r, c])
                color = self.activation_color(val, max_value)
                cx0 = x0 + c * cell
                cy0 = y0 + r * cell
                canvas.create_rectangle(cx0, cy0, cx0 + cell, cy0 + cell, fill=color, outline='#0f172a')

    def refresh_activation_view(self):
        if self.board_only_mode:
            return
        if not hasattr(self, 'activation_graph') or self.activation_graph is None:
            return
        try:
            if not self.use_nn or self.nn_model is None:
                self.activation_snapshot = None
                self.last_activation_key = None
                self.draw_activation_maps(self.activation_graph)
                return
            p_board = self.B if self.tn == 1 else self.W
            o_board = self.W if self.tn == 1 else self.B
            state_key = self.make_state_key(p_board, o_board, self.tn)
            if state_key != self.last_activation_key:
                try:
                    self.activation_snapshot = get_nn_activation_snapshot(self.nn_model, p_board, o_board, self.tn, top_k=6)
                except Exception:
                    self.activation_snapshot = None
                self.last_activation_key = state_key
            self.draw_activation_maps(self.activation_graph)
        except Exception:
            pass

    def draw_activation_maps(self, canvas):
        if canvas is None:
            return
        try:
            canvas.delete('all')
        except Exception:
            return
        self.module_activity['ACTIV'] = time.time()
        self.mark_connections_active(('TRUNK', 'POLICY'), ('TRUNK', 'VALUE'))
        w = int(canvas['width'])
        h = int(canvas['height'])
        canvas.create_rectangle(10, 10, w - 10, h - 10, outline='#cbd5e1', fill='#ffffff')
        if not self.use_nn or self.nn_model is None:
            canvas.create_text(w // 2, h // 2, text='NNが無効です', fill='#64748b', font=('Arial', 11))
            return
        if not self.activation_snapshot:
            canvas.create_text(w // 2, h // 2, text='活性データを準備中です', fill='#64748b', font=('Arial', 11))
            return
        maps = [('Policy', self.activation_snapshot.get('policy_grid'), ''), ('Value', self.activation_snapshot.get('value_grid'), '')]
        for info in self.activation_snapshot.get('trunk_maps', []):
            maps.append((info.get('label', 'C'), info.get('grid'), f"{float(info.get('strength', 0.0)):.3f}"))
        maps = maps[:8]
        map_count = max(1, len(maps))
        cols = 2
        rows = max(1, int(math.ceil(map_count / cols)))
        top_margin = 34
        bottom_margin = 28
        gap_x = 40
        gap_y = 20
        usable_w = w - 28 - gap_x * (cols - 1)
        usable_h = h - top_margin - bottom_margin - gap_y * (rows - 1)
        cell_w = usable_w // (cols * 8)
        cell_h = usable_h // (rows * 8)
        cell = max(5, min(12, cell_w, cell_h))
        grid_size = cell * 8
        total_w = cols * grid_size + gap_x * (cols - 1)
        x_start = max(14, (w - total_w) // 2)
        y0 = 28
        for idx, (title, grid, subtitle) in enumerate(maps):
            if grid is None:
                continue
            row = idx // cols
            col = idx % cols
            curr_x = x_start + col * (grid_size + gap_x)
            curr_y = y0 + row * (grid_size + gap_y)
            self.draw_activation_heatmap(canvas, grid, curr_x, curr_y, cell, title, subtitle)
        canvas.create_text(16, h - 28, text='最大活性=1.0に正規化', anchor='w', fill='#64748b', font=('Arial', 8))

    def draw_line_graph(self, canvas, history, specs, y_min, y_max, labels):
        if canvas is None:
            return
        try:
            canvas.delete('all')
        except Exception:
            return
        self.module_activity['GRAPH'] = time.time()
        w = int(canvas['width'])
        h = int(canvas['height'])
        ml, mr, mt, mb = (42, 16, 12, 52)
        legend_y = h - 18
        canvas.create_rectangle(ml, mt, w - mr, h - mb, outline='#94a3b8')
        axis_color = '#334155'
        grid_color = '#cbd5e1'
        tick_color = '#334155'
        label_color = '#334155'
        canvas.create_line(ml, mt, ml, h - mb, fill=axis_color, width=2)
        canvas.create_line(ml, h - mb, w - mr, h - mb, fill=axis_color, width=2)
        for step in range(11):
            frac = step / 10.0
            y = mt + (h - mt - mb) * frac
            canvas.create_line(ml, y, w - mr, y, fill=grid_color)
            canvas.create_line(ml - 5, y, ml, y, fill=tick_color, width=2)
            val = y_max - (y_max - y_min) * frac
            canvas.create_text(ml - 7, y, text=f'{val:.0f}', anchor='e', fill=label_color, font=('Arial', 8))
        if not history:
            canvas.create_text(w // 2, h // 2, text='データなし', fill='#64748b')
            return
        min_ply = 0
        max_ply = history[-1][0]
        display_max_ply = min(64, max(4, int(math.ceil(max_ply / 0.75 / 4.0) * 4.0)))
        ply_span = max(1, display_max_ply - min_ply)
        xs = [ml + (w - ml - mr) * ((entry[0] - min_ply) / ply_span) for entry in history]
        start_tick = 0
        tick_candidates = []
        if min_ply not in tick_candidates:
            tick_candidates.append(min_ply)
        for ply in range(start_tick + 4, display_max_ply + 1, 4):
            if ply != min_ply and ply != display_max_ply:
                tick_candidates.append(ply)
        if display_max_ply not in tick_candidates:
            tick_candidates.append(display_max_ply)
        for ply in tick_candidates:
            x = ml + (w - ml - mr) * ((ply - min_ply) / ply_span)
            canvas.create_line(x, mt, x, h - mb, fill=grid_color)
            canvas.create_line(x, h - mb, x, h - mb + 5, fill=tick_color, width=2)
            canvas.create_text(x, h - mb + 14, text=str(int(ply)), anchor='n', fill=label_color, font=('Arial', 8))
        for idx, color in specs:
            segment_points = []
            for x, entry in zip(xs, history):
                val = entry[idx]
                if val is None:
                    if len(segment_points) >= 4:
                        canvas.create_line(*segment_points, fill=color, width=2)
                    elif len(segment_points) == 2:
                        canvas.create_oval(segment_points[0] - 2, segment_points[1] - 2, segment_points[0] + 2, segment_points[1] + 2, fill=color, outline=color)
                    segment_points = []
                    continue
                y = mt + (y_max - val) / (y_max - y_min) * (h - mt - mb)
                segment_points.extend([x, y])
            if len(segment_points) >= 4:
                canvas.create_line(*segment_points, fill=color, width=2)
            elif len(segment_points) == 2:
                canvas.create_oval(segment_points[0] - 2, segment_points[1] - 2, segment_points[0] + 2, segment_points[1] + 2, fill=color, outline=color)
            for x, entry in zip(xs, history):
                val = entry[idx]
                if val is None:
                    continue
                y = mt + (y_max - val) / (y_max - y_min) * (h - mt - mb)
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
        if len(specs) >= 2 and len(history) > 0:
            midpoints = []
            for x, entry in zip(xs, history):
                val1 = entry[specs[0][0]] if len(entry) > specs[0][0] else None
                val2 = entry[specs[1][0]] if len(entry) > specs[1][0] else None
                if val1 is not None and val2 is not None:
                    midpoint = (val1 + val2) / 2.0
                    y = mt + (y_max - midpoint) / (y_max - y_min) * (h - mt - mb)
                    midpoints.extend([x, y])
                else:
                    if len(midpoints) >= 4:
                        for i in range(0, len(midpoints) - 2, 2):
                            canvas.create_line(midpoints[i], midpoints[i + 1], midpoints[i + 2], midpoints[i + 3], fill='#00aa00', width=2, dash=(4, 2))
                    midpoints = []
            if len(midpoints) >= 4:
                for i in range(0, len(midpoints) - 2, 2):
                    canvas.create_line(midpoints[i], midpoints[i + 1], midpoints[i + 2], midpoints[i + 3], fill='#00aa00', width=2, dash=(4, 2))
        legend_x = ml
        for color, label in zip([s[1] for s in specs], labels):
            canvas.create_rectangle(legend_x, legend_y - 5, legend_x + 10, legend_y + 5, fill=color, outline=color)
            canvas.create_text(legend_x + 14, legend_y, text=label, anchor='w', fill='#334155', font=('Arial', 8))
            legend_x += 90
        if len(specs) >= 2:
            canvas.create_line(legend_x, legend_y, legend_x + 10, legend_y, fill='#00aa00', width=2, dash=(4, 2))
            canvas.create_text(legend_x + 14, legend_y, text=f' ({labels[0]}+{labels[1]}) / 2', anchor='w', fill='#334155', font=('Arial', 8))

    def on_close(self):
        self.running = False
        self.ponder_token += 1
        self.ponder_sf[0] = 1
        with self.ponder_lock:
            self.ponder_cache.clear()
        self.sf_arr[0] = 1
        try:
            _NN_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.rt.quit()
        self.rt.destroy()

    def s_col(self):
        c = colorchooser.askcolor(initialcolor=self.bc)[1]
        if c and hasattr(self, 'cv') and (self.cv is not None):
            self.bc = c
            try:
                self.cv.config(bg=c)
            except Exception:
                pass
            self.drw()

    def toggle_board_only_mode(self):
        """盤面のみ表示モードを切り替え"""
        self.ponder_token += 1
        self.ponder_sf[0] = 1
        with self.ponder_lock:
            self.ponder_cache.clear()
        self.board_only_mode = self.board_only_var.get()
        if self.position_history:
            self.history_view_index = len(self.position_history) - 1
        self.rebuild_layout()
        mode_text = '盤面のみ表示モード: ON' if self.board_only_mode else '盤面のみ表示モード: OFF'
        print(mode_text, flush=True)
        if hasattr(self, 'log_text') and (not self.board_only_mode):
            self.call_on_ui_thread(self.append_log_message, mode_text)

    def toggle_auto_battle_mode(self):
        self.auto_battle = bool(self.auto_battle_var.get())
        if self.auto_battle:
            self.ponder_token += 1
            self.ponder_sf[0] = 1
            with self.ponder_lock:
                self.ponder_cache.clear()
            if self.btn_pass:
                self.btn_pass.destroy()
                self.btn_pass = None
            self.log('自動対戦モード: ON')
        else:
            self.ponder_sf[0] = 1
            self.log('自動対戦モード: OFF')
        self.drw()
        self.chk()

    def rebuild_layout(self):
        """UIレイアウトを再構築"""
        self.cv = None
        self.win_graph = None
        self.diff_graph = None
        self.activation_graph = None
        self.connection_graph = None
        self.log_text = None
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        if self.board_only_mode:
            self.rt.minsize(self.board_size + 40, self.board_size + 40)
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.board_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.board_frame.grid(row=0, column=0, sticky='nsew')
            self.cv = tk.Canvas(self.board_frame, width=self.board_size, height=self.board_size, bg=self.bc)
            self.cv.pack(expand=True)
        else:
            self.rt.minsize(1450, 760)
            self.main_frame.grid_columnconfigure(0, weight=0)
            self.main_frame.grid_columnconfigure(1, weight=1)
            self.main_frame.grid_columnconfigure(2, weight=0)
            self.main_frame.grid_rowconfigure(0, weight=0)
            self.main_frame.grid_rowconfigure(1, weight=0)
            self.left_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.left_frame.grid(row=0, column=0, sticky='nw')
            self.graphs_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.graphs_frame.grid(row=0, column=1, sticky='nw', padx=(12, 0))
            self.activation_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.activation_frame.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(12, 0))
            self.log_host_frame = tk.Frame(self.main_frame, bg='#eef2f7')
            self.log_host_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 0))
            self.cv = tk.Canvas(self.left_frame, width=self.board_size, height=self.board_size, bg=self.bc)
            self.cv.pack()
            tk.Label(self.graphs_frame, text='勝率グラフ', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.win_graph = tk.Canvas(self.graphs_frame, width=self.graph_width, height=self.graph_height, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.win_graph.pack()
            tk.Label(self.graphs_frame, text='判定割合', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w', pady=(8, 0))
            self.diff_graph = tk.Canvas(self.graphs_frame, width=self.graph_width, height=self.graph_height + 50, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.diff_graph.pack()
            tk.Label(self.log_host_frame, text='ログ', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.log_frame = tk.Frame(self.log_host_frame, height=220, bg='#eef2f7')
            self.log_frame.pack(fill='x', pady=(0, 0))
            self.log_frame.pack_propagate(False)
            self.log_text = tk.Text(self.log_frame, width=90, height=5, bg='#ffffff', fg='#111111', insertbackground='#111111')
            self.log_text.pack(fill='both', expand=True)
            self.log_text.config(state='disabled')
            self.log_text.tag_config('green', foreground='#00aa00')
            self.log_text.tag_config('blue', foreground='#0066cc')
            tk.Label(self.activation_frame, text='NN活性マップ', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w')
            self.activation_graph = tk.Canvas(self.activation_frame, width=250, height=340, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.activation_graph.pack()
            tk.Label(self.activation_frame, text='関数の接続図', bg='#eef2f7', fg='#0f172a', font=('Yu Gothic UI', 11, 'bold')).pack(anchor='w', pady=(10, 0))
            self.connection_graph = tk.Canvas(self.activation_frame, width=250, height=340, bg='#f8fafc', highlightthickness=1, highlightbackground='#cbd5e1')
            self.connection_graph.pack()
        self.cv.bind('<Button-1>', self.clk)
        self.drw()
        if not self.board_only_mode:
            self.redraw_graphs()

    def c_col(self):
        c = simpledialog.askstring('色', 'black/white:', initialvalue='black' if self.hc == 1 else 'white')
        if c:
            self.game_id += 1
            self.game_over_shown = False
            self.ponder_token += 1
            self.ponder_sf[0] = 1
            with self.ponder_lock:
                self.ponder_cache.clear()
            self.is_thinking = False
            self.hc = 1 if c.lower().startswith('b') or c == '黒' else -1
            self.ac = -self.hc
            self.last_win_rate = 50.0
            self.last_mcts_win_rate = None
            self.reset_graph_history()
            self.redraw_graphs()
            self.update_title()
            self.drw()
            self.chk()

    def ng(self):
        self.game_id += 1
        self.game_over_shown = False
        self.ponder_token += 1
        self.ponder_sf[0] = 1
        with self.ponder_lock:
            self.ponder_cache.clear()
        self.is_thinking = False
        self.B = np.uint64(34628173824)
        self.W = np.uint64(68853694464)
        self.tn = 1
        self.reset_position_history()
        self.last_win_rate = 50.0
        self.last_mcts_win_rate = None
        self.reset_graph_history()
        self.redraw_graphs()
        self.update_title()
        if self.btn_pass:
            self.btn_pass.destroy()
            self.btn_pass = None
        self.drw()
        self.chk()

    def mc(self):
        return int(self.B | self.W).bit_count() - 4

    def get_search_depth_cap(self, empty=None, is_exact=False):
        cap = 60
        if is_exact and empty is not None:
            cap = min(cap, max(2, int(empty)))
        return cap

    def clamp_search_depth(self, depth, empty=None, is_exact=False):
        return max(2, min(self.get_search_depth_cap(empty, is_exact), int(depth)))

    def get_live_position_snapshot(self):
        return {'B': int(self.B), 'W': int(self.W), 'tn': int(self.tn)}

    def reset_position_history(self):
        self.position_history = [self.get_live_position_snapshot()]
        self.history_view_index = 0

    def record_position_history(self):
        snapshot = self.get_live_position_snapshot()
        if not self.position_history:
            self.position_history = [snapshot]
            self.history_view_index = 0
            return
        latest_index = len(self.position_history) - 1
        was_at_latest = self.history_view_index >= latest_index
        last_snapshot = self.position_history[-1]
        if snapshot != last_snapshot:
            self.position_history.append(snapshot)
            if (not self.board_only_mode) or was_at_latest:
                self.history_view_index = len(self.position_history) - 1
        elif (not self.board_only_mode) and self.history_view_index != latest_index:
            self.history_view_index = latest_index
        elif self.history_view_index > latest_index:
            self.history_view_index = latest_index

    def is_viewing_historical_position(self):
        return bool(self.board_only_mode and self.position_history and self.history_view_index < (len(self.position_history) - 1))

    def get_display_position_state(self):
        self.record_position_history()
        if self.board_only_mode and self.position_history:
            index = max(0, min(int(self.history_view_index), len(self.position_history) - 1))
            snapshot = self.position_history[index]
            return (np.uint64(snapshot['B']), np.uint64(snapshot['W']), int(snapshot['tn']))
        return (self.B, self.W, int(self.tn))

    def navigate_position_history(self, delta):
        if (not self.board_only_mode) or (not self.position_history):
            return
        latest_index = len(self.position_history) - 1
        next_index = max(0, min(latest_index, int(self.history_view_index) + int(delta)))
        if next_index == self.history_view_index:
            return
        was_historical = self.is_viewing_historical_position()
        self.history_view_index = next_index
        if self.btn_pass:
            self.btn_pass.destroy()
            self.btn_pass = None
        self.drw()
        if was_historical and (not self.is_viewing_historical_position()):
            self.rt.after(0, self.chk)

    def _should_handle_viewer_shortcut(self, event=None):
        if not self.board_only_mode:
            return False
        widget = getattr(event, 'widget', None)
        if widget is not None:
            widget_class = str(widget.winfo_class())
            if widget_class in {'Entry', 'Text', 'TEntry', 'Spinbox', 'TCombobox'}:
                return False
        return True

    def on_viewer_step_back(self, event=None):
        if not self._should_handle_viewer_shortcut(event):
            return None
        self.navigate_position_history(-1)
        return 'break'

    def on_viewer_step_forward(self, event=None):
        if not self._should_handle_viewer_shortcut(event):
            return None
        self.navigate_position_history(1)
        return 'break'

    def make_state_key(self, p_board, o_board, turn):
        return f'{int(p_board)}_{int(o_board)}_{int(turn)}'

    def apply_move_pair(self, player_board, opp_board, move_idx):
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                next_player, next_opp = cpp_engine.apply_move(int(player_board), int(opp_board), int(move_idx))
                return (np.uint64(next_player), np.uint64(next_opp))
            except Exception:
                pass
        flip = np.uint64(get_flip(player_board, opp_board, np.int64(move_idx)))
        next_player = (player_board | np.uint64(1) << np.uint64(move_idx) | flip) & _FULL_MASK
        next_opp = (opp_board ^ flip) & _FULL_MASK
        return (next_player, next_opp)

    def should_stop_pondering(self, current_game_id, ponder_token, ponder_sf):
        return not self.running or self.game_id != current_game_id or self.ponder_token != ponder_token or (self.tn != self.hc) or bool(ponder_sf[0])

    def get_legal_index_list(self, player_board, opp_board):
        if self.use_cpp_engine and cpp_engine is not None:
            return list(cpp_engine.legal_move_indices(int(player_board), int(opp_board)))
        valid = self.legal_moves_mask(player_board, opp_board)
        legal_indices = []
        while valid:
            move = (valid & -valid).bit_length() - 1
            valid &= valid - 1
            legal_indices.append(move)
        return legal_indices

    def build_ponder_candidates(self, current_game_id, ponder_token):
        if not self.running or self.game_id != current_game_id or self.ponder_token != ponder_token or (self.tn != self.hc):
            return []
        hB = self.B if self.hc == 1 else self.W
        hW = self.W if self.hc == 1 else self.B
        valid = self.legal_moves_mask(hB, hW)
        if valid == 0:
            return []
        cpp_move_analysis = None
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                cpp_move_analysis = cpp_engine.analyze_legal_moves_cached(int(hB), int(hW), int(self.mc()))
            except Exception:
                cpp_move_analysis = None
        policy_scores = {}
        if self.use_nn and self.nn_model is not None:
            try:
                tensor_batch = make_input_tensor([(int(hB), int(hW), int(self.tn))])
                p_b, _ = nn_infer_batch(self.nn_model, tensor_batch)
                root_policy = p_b[0]
                valid_temp = valid
                while valid_temp:
                    move = (valid_temp & -valid_temp).bit_length() - 1
                    valid_temp &= valid_temp - 1
                    policy_scores[move] = float(root_policy[move])
            except Exception:
                policy_scores = {}
        candidates = []
        if cpp_move_analysis is not None:
            root_moves = list(cpp_move_analysis.get('moves', []))
            next_p_list = list(cpp_move_analysis.get('next_p', []))
            next_o_list = list(cpp_move_analysis.get('next_o', []))
            iter_items = zip(root_moves, next_p_list, next_o_list)
        else:
            valid_temp = valid
            raw_items = []
            while valid_temp:
                move = (valid_temp & -valid_temp).bit_length() - 1
                valid_temp &= valid_temp - 1
                human_after, ai_after = self.apply_move_pair(hB, hW, move)
                raw_items.append((move, human_after, ai_after))
            iter_items = raw_items
        for move, human_after, ai_after in iter_items:
            if self.use_cpp_engine and cpp_engine is not None:
                try:
                    static_score = -self.evaluate_position(ai_after, human_after, self.mc() + 1)
                except Exception:
                    static_score = -self.evaluate_position(ai_after, human_after, self.mc() + 1)
            else:
                static_score = -self.evaluate_position(ai_after, human_after, self.mc() + 1)
            total_score = static_score + 300.0 * policy_scores.get(move, 0.0)
            aB = ai_after
            oB = human_after
            legal_indices = self.get_legal_index_list(aB, oB)
            if not legal_indices:
                continue
            mvs = int((human_after | ai_after).bit_count() - 4)
            empty = 64 - (mvs + 4)
            is_exact = self.should_start_exact_early(aB, oB, empty, len(legal_indices))
            root_policy_vector = [0.0] * 64
            if self.use_nn and self.nn_model is not None:
                try:
                    tensor_batch = make_input_tensor([(int(aB), int(oB), int(self.ac))])
                    p_b, _ = nn_infer_batch(self.nn_model, tensor_batch)
                    root_policy = p_b[0]
                    for idx in legal_indices:
                        root_policy_vector[idx] = float(root_policy[idx])
                except Exception:
                    root_policy_vector = [0.0] * 64
            candidates.append({'move': move, 'human_after': human_after, 'ai_after': ai_after, 'score': total_score, 'αβ': aB, 'oB': oB, 'legal_indices': legal_indices, 'mvs': mvs, 'empty': empty, 'is_exact': is_exact, 'root_policy_vector': root_policy_vector, 'cache_key': self.make_state_key(aB, oB, self.ac)})
        candidates.sort(key=lambda x: -x['score'])
        return candidates[:3]

    def drw(self, probs_map=None):
        if not self.running or not hasattr(self, 'cv'):
            return
        try:
            self.cv.delete('all')
        except Exception:
            return
        display_B, display_W, display_tn = self.get_display_position_state()
        viewing_history = self.is_viewing_historical_position()
        cs = self.cell_size
        piece_pad = max(6, int(cs * 0.1))
        marker_r = max(5, int(cs * 0.08))
        text_y = max(15, int(cs * 0.25))
        font_sz = max(9, int(cs * 0.15))
        for r in range(8):
            for c in range(8):
                x, y, i = (c * cs, r * cs, np.uint64(r * 8 + c))
                self.cv.create_rectangle(x, y, x + cs, y + cs, outline='#1b5e20')
                if display_B >> i & np.uint64(1):
                    self.cv.create_oval(x + piece_pad, y + piece_pad, x + cs - piece_pad, y + cs - piece_pad, fill='black')
                elif display_W >> i & np.uint64(1):
                    self.cv.create_oval(x + piece_pad, y + piece_pad, x + cs - piece_pad, y + cs - piece_pad, fill='white', outline='#ccc')
        if (not viewing_history) and display_tn == self.hc:
            mB, mW = (display_B if self.hc == 1 else display_W, display_W if self.hc == 1 else display_B)
            tm = self.legal_moves_mask(mB, mW)
            ue_list = []
            while tm:
                ix, t = (0, tm & -tm)
                while t > 1:
                    t >>= 1
                    ix += 1
                if self.sp_var.get():
                    nP, nO = self.apply_move_pair(mB, mW, ix)
                    if self.use_cpp_engine and cpp_engine is not None:
                        try:
                            val = -float(cpp_engine.evaluate_board_cached(int(nO), int(nP), int(self.mc() + 1)))
                        except Exception:
                            val = -self.evaluate_position(nO, nP, self.mc() + 1)
                    else:
                        val = -self.evaluate_position(nO, nP, self.mc() + 1)
                    ue_list.append((ix, val))
                else:
                    cx = ix % 8 * cs + cs // 2
                    cy = ix // 8 * cs + cs // 2
                    self.cv.create_oval(cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r, fill='#4caf50', outline='')
                tm &= tm - 1
            if self.sp_var.get() and ue_list:
                mx = max([v[1] for v in ue_list])
                evs = [math.exp(max(-20.0, (v[1] - mx) / 400.0)) for v in ue_list]
                se = sum(evs) if sum(evs) > 0 else 1.0
                for i, (ix, _) in enumerate(ue_list):
                    cx = ix % 8 * cs + cs // 2
                    cy = ix // 8 * cs + cs // 2
                    self.cv.create_oval(cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r, fill='#4caf50', outline='')
                    self.cv.create_text(cx, ix // 8 * cs + text_y, text=f'{evs[i] / se * 100.0:.1f}%', fill='yellow', font=('Arial', font_sz, 'bold'))
        elif (not viewing_history) and display_tn == self.ac and self.sp_var.get() and probs_map and (not self.board_only_mode):
            for ix, prob in probs_map.items():
                cx = ix % 8 * cs + cs // 2
                cy = ix // 8 * cs + cs // 2
                self.cv.create_oval(cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r, fill='#4caf50', outline='')
                self.cv.create_text(cx, ix // 8 * cs + text_y, text=f'{prob:.1f}%', fill='cyan', font=('Arial', font_sz, 'bold'))
        self.refresh_activation_view()

    def clk(self, e):
        if self.auto_battle or not self.running or self.tn != self.hc or self.is_viewing_historical_position() or (self.btn_pass and (not self.board_only_mode)):
            return
        ix = np.uint64(e.y // self.cell_size * 8 + e.x // self.cell_size)
        mB, mW = (self.B if self.hc == 1 else self.W, self.W if self.hc == 1 else self.B)
        if self.legal_moves_mask(mB, mW) >> int(ix) & 1:
            self.ponder_token += 1
            self.ponder_sf[0] = 1
            next_player, next_opp = self.apply_move_pair(mB, mW, ix)
            if self.hc == 1:
                self.B, self.W = (next_player, next_opp)
            else:
                self.W, self.B = (next_player, next_opp)
            self.tn = self.ac
            self.drw()
            self.rt.after(100, self.chk)

    def show_pass_btn(self):
        parent_frame = getattr(self, 'left_frame', None) or getattr(self, 'board_frame', None)
        if not parent_frame or not parent_frame.winfo_exists() or (not self.cv.winfo_exists()):
            return
        self.btn_pass = tk.Button(parent_frame, text='pass', font=('Arial', 12, 'bold'), bg='#cfd8dc', fg='#37474f', relief='flat', padx=20, pady=10, command=self.do_pass)
        self.btn_pass.place(in_=self.cv, x=self.board_size // 2, y=self.board_size // 2, anchor='center')

    def advance_forced_pass_state(self):
        current_player = self.B if self.tn == 1 else self.W
        current_opp = self.W if self.tn == 1 else self.B
        if self.legal_moves_mask(current_player, current_opp):
            return False
        next_turn = -self.tn
        next_player = self.B if next_turn == 1 else self.W
        next_opp = self.W if next_turn == 1 else self.B
        if self.legal_moves_mask(next_player, next_opp):
            self.tn = next_turn
            return True
        return False

    def resolve_forced_pass(self, delay_ms=100, stop_ponder=False):
        if self.btn_pass:
            self.btn_pass.destroy()
            self.btn_pass = None
        if stop_ponder:
            self.ponder_token += 1
            self.ponder_sf[0] = 1
        if self.legal_moves_mask(self.B if self.tn == 1 else self.W, self.W if self.tn == 1 else self.B):
            self.drw()
            self.rt.after(delay_ms, self.chk)
            return
        self.advance_forced_pass_state()
        self.drw()
        self.rt.after(delay_ms, self.chk)

    def do_pass(self):
        self.resolve_forced_pass(delay_ms=50, stop_ponder=True)

    def ai_r(self, current_game_id):
        follow_up_handled = False
        try:
            if not self.running or self.game_id != current_game_id:
                return
            if (not self.auto_battle) and self.tn != self.ac:
                return
            ai_start_time = time.time()
            is_quick_mode = self.time_limit_sec <= 0.5
            mvs = 0
            empty = 64
            self.sf_arr[0] = 0
            acting_side = self.tn if self.auto_battle else self.ac
            aB, oB = (self.B if acting_side == 1 else self.W, self.W if acting_side == 1 else self.B)
            mvs = int((aB | oB).bit_count()) - 4
            empty = 64 - int((aB | oB).bit_count())
            initial_legal_mask = self.legal_moves_mask(aB, oB)
            initial_legal_count = int(initial_legal_mask.bit_count())
            if initial_legal_count == 0:
                self.log('search: no legal moves -> pass')
                self.advance_forced_pass_state()
                self.call_on_ui_thread(self.resolve_forced_pass)
                follow_up_handled = True
                return
            initial_time_limit = self.get_auto_time_limit(empty, initial_legal_count, mvs)
            solver_threshold = self.get_endgame_solver_threshold(initial_legal_count, initial_time_limit)
            solver_time_limit_ms = self.get_endgame_solver_time_limit_ms(empty, initial_legal_count, initial_time_limit)
            if empty <= solver_threshold and ENDGAME_SOLVER_AVAILABLE and (not is_quick_mode):
                self.log(f'endgame: C++ solver activated (empties={empty})')
                start_time = time.time()
                try:
                    current_player = int(acting_side)
                    best_move_idx = -1
                    completed_depth = 0
                    fully_solved = False
                    if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'solve_endgame_status'):
                        status = cpp_engine.solve_endgame_status(int(self.B), int(self.W), current_player, empty, int(solver_time_limit_ms))
                        best_move_idx = int(status.get('best_move', -1))
                        completed_depth = int(status.get('completed_depth', 0))
                        fully_solved = bool(status.get('fully_solved', False))
                    else:
                        best_move_idx = get_endgame_best_move(int(self.B), int(self.W), current_player, empty, int(solver_time_limit_ms))
                        fully_solved = (best_move_idx >= 0) or (best_move_idx == -1)  # -1も有効な結果（pass）
                        completed_depth = empty if fully_solved else 0
                    if fully_solved and best_move_idx >= 0:
                        # 通常の手を指す場合
                        if self.tn == 1:
                            self.B, self.W = self.apply_move_pair(self.B, self.W, best_move_idx)
                        else:
                            self.W, self.B = self.apply_move_pair(self.W, self.B, best_move_idx)
                        elapsed = time.time() - start_time
                        move_label = self.format_move_label(best_move_idx)
                        self.log(f'endgame: exact solved in {elapsed:.2f}s | depth={completed_depth}/{empty} | best move=({move_label})')
                        self.tn = -self.tn
                        self.drw()
                        self.rt.after(100, self.chk)
                        follow_up_handled = True
                        return
                    elif fully_solved and best_move_idx == -1:
                        # passの場合
                        elapsed = time.time() - start_time
                        self.log(f'endgame: exact solved in {elapsed:.2f}s | depth={completed_depth}/{empty} | pass')
                        self.advance_forced_pass_state()
                        self.call_on_ui_thread(self.resolve_forced_pass, 100, False)
                        follow_up_handled = True
                        return
                    elif best_move_idx >= 0:
                        self.log(f'endgame: partial solve depth={completed_depth}/{empty} -> falling back to normal search')
                    else:
                        self.log('endgame: no valid moves found, falling back to normal search')
                except Exception as e:
                    self.log(f'endgame: solver error ({e}), falling back to normal search')
            book_move = None
            book_roll_used = False
            prior_map = {}
            primary_move = None
            dynamic_book_rate = 1.0
            if self.opening_book:
                valid_moves = self.legal_moves_mask(aB, oB)
                legal_list = []
                temp = valid_moves
                while temp:
                    bit = temp & -temp
                    temp ^= bit
                    legal_list.append((bit - 1).bit_count())
                prior_map, primary_move, book_roll_used = self.get_book_prior_info(aB, oB, legal_list, dynamic_book_rate)
                if book_roll_used:
                    self.log(f'book: found in opening book | primary_move={primary_move} | prior_map={prior_map}')
                if primary_move is not None:
                    book_move = primary_move
                    self.log(f'book: move available primary_move={primary_move}')
            empty = 64 - (self.mc() + 4)
            mvs = self.mc()
            start_dp = 2
            is_resumed = False
            if self.use_tt_resume and self.use_cpp_engine and (cpp_engine is not None):
                tt_info = cpp_engine.probe_tt(int(aB), int(oB))
                if bool(tt_info['found']):
                    self.mark_modules_active('TT')
                    self.mark_connections_active(('BOARD', 'TT'))
                    td = self.clamp_search_depth(tt_info['depth'])
                    if td >= 4:
                        start_dp = max(start_dp, td + 1)
                        is_resumed = True
            elif self.use_tt_resume:
                hv_r = zobrist_hash(aB, oB)
                tx_r = np.int64(hv_r % np.uint64(TT_SIZE)) * 2
                if self.tk_arr[tx_r] == hv_r:
                    td = self.clamp_search_depth(self.ti_arr[tx_r, 0])
                    if td >= 4:
                        start_dp = max(start_dp, td + 1)
                        is_resumed = True
                elif self.tk_arr[tx_r + 1] == hv_r:
                    td = self.clamp_search_depth(self.ti_arr[tx_r + 1, 0])
                    if td >= 4:
                        start_dp = max(start_dp, td + 1)
                        is_resumed = True
            if self.use_cpp_engine and cpp_engine is not None:
                try:
                    analysis = cpp_engine.analyze_legal_moves_cached(int(aB), int(oB), int(mvs))
                    legal_indices = list(analysis.get('moves', []))
                    evals = list(analysis.get('evals', []))
                except Exception:
                    legal_indices = cpp_engine.legal_move_indices(int(aB), int(oB))
                    evals = cpp_engine.evaluate_moves(int(aB), int(oB), int(mvs), legal_indices, self.weights_list)
                lm = list(zip(legal_indices, evals))
                valid = self.legal_moves_mask(aB, oB)
            else:
                valid = self.legal_moves_mask(aB, oB)
                lm = []
                v_temp = valid
                while v_temp:
                    bit = v_temp & -v_temp
                    v_temp ^= bit
                    idx = (bit - 1).bit_count()
                    nP, nO = self.apply_move_pair(aB, oB, idx)
                    lm.append((idx, self.evaluate_position(nP, nO, mvs + 1)))
            policy_scores = {}
            root_policy_vector = [0.0] * 64
            if self.use_nn and self.nn_model is not None and lm:
                try:
                    self.mark_modules_active('NN-IN', 'TRUNK', 'POLICY')
                    self.mark_connections_active(('BOARD', 'NN-IN'), ('NN-IN', 'TRUNK'), ('TRUNK', 'POLICY'))
                    tensor_batch = make_input_tensor([(int(aB), int(oB), int(self.tn))])
                    p_b, _ = nn_infer_batch(self.nn_model, tensor_batch)
                    root_policy = p_b[0]
                    for idx, _ in lm:
                        policy_value = float(root_policy[idx])
                        policy_scores[idx] = policy_value
                        root_policy_vector[idx] = policy_value
                except Exception:
                    policy_scores = {}
                    root_policy_vector = [0.0] * 64
            if policy_scores:
                lm.sort(key=lambda x: -(x[1] + 400.0 * policy_scores.get(x[0], 0.0)))
            else:
                lm.sort(key=lambda x: -x[1])
            rms = [m[0] for m in lm]
            if not rms:
                self.log('search: no legal moves -> pass')
                self.advance_forced_pass_state()
                self.call_on_ui_thread(self.resolve_forced_pass)
                follow_up_handled = True
                return
            ponder_key = self.make_state_key(aB, oB, self.tn)
            with self.ponder_lock:
                ponder_entry = self.ponder_cache.pop(ponder_key, None)
            if ponder_entry:
                self.mark_modules_active('PONDER')
                self.mark_connections_active(('PONDER', 'αβ'), ('PONDER', 'MCTS'), ('PONDER', 'TT'))
                cached_order = [move for move in ponder_entry.get('ordered_moves', []) if move in rms]
                if cached_order:
                    cached_set = set(cached_order)
                    rms = cached_order + [move for move in rms if move not in cached_set]
                cached_depth = self.clamp_search_depth(ponder_entry.get('completed_depth', 2))
                start_dp = max(start_dp, cached_depth + 1)
                self.log(f'Ponder hit: reused cached ordering. cached depth={cached_depth}, start depth={start_dp}')
            ponder_mcts_res = dict(ponder_entry.get('mcts_res', {})) if ponder_entry else {}
            ponder_best_mcts_wr = float(ponder_entry.get('best_mcts_wr', 50.0)) if ponder_entry else 50.0
            ponder_root_visits = dict(ponder_entry.get('root_visits', {})) if ponder_entry else {}
            ponder_mcts_sim_count = int(ponder_entry.get('mcts_sim_count', 0)) if ponder_entry else 0
            cached_best = self.get_exact_cached_move(aB, oB, self.tn)
            if cached_best is not None and cached_best in rms:
                self.log(f'exact cache hit: {self.format_move_label(cached_best)}')
                best_move = cached_best
                final_scores = [(best_move, 100.0)]
                next_player, next_opp = self.apply_move_pair(aB, oB, best_move)
                if acting_side == 1:
                    self.B, self.W = (next_player, next_opp)
                else:
                    self.W, self.B = (next_player, next_opp)
                self.tn = -self.tn
                self.call_on_ui_thread(self.drw)
                self.call_on_ui_thread(self.chk)
                return
            time_limit = self.get_auto_time_limit(empty, len(rms), mvs)
            is_exact = self.should_start_exact_early(aB, oB, empty, len(rms), start_dp, time_limit)
            start_dp = self.clamp_search_depth(start_dp, empty, is_exact)
            if is_exact:
                self.stop_pondering_explicitly()
                self.log('ponder: stopped for exact solve')
            best_move = -1
            if book_move is not None:
                if valid >> book_move & 1:
                    best_move = book_move
                else:
                    for idx in rms:
                        if idx == book_move:
                            best_move = idx
                            break
                    if best_move == -1 and rms:
                        if mvs == 0:
                            best_move = rms[0]
            if best_move == -1 and mvs == 0 and rms:
                best_move = rms[0]
            mcts_res = {}
            best_mcts_wr = 50.0
            root_visits = {}
            mcts_sim_count = 0
            ab_res = {}
            ab_val_res = {}
            resolved_flag = False
            # 枝刈り設定に基づいてAB探索を使用するか決定
            if is_exact:
                use_ab = True
            elif self.use_mcts_only:
                use_ab = False
            else:
                # MCTS有効時は枝刈り設定に従う
                if self.use_mcts_enabled and self.mcts_pruning_enabled:
                    # MCTS枝刈り有効時はMCTS事前探索後にABを実行
                    use_ab = True
                elif self.pruning_enabled:
                    # 通常枝刈り有効時はABを実行
                    use_ab = True
                else:
                    use_ab = True  # デフォルトはAB実行
            setup_overhead = time.time() - ai_start_time
            overhead_threshold = 0.05 if is_quick_mode else 0.1
            actual_time_limit = max(0.05, time_limit - setup_overhead - overhead_threshold)
            auto_time_str = ''
            weights_list = self.weights_list
            order_map_list = self.order_map_list
            search_profile = self.get_search_time_profile(empty, len(rms), actual_time_limit, is_exact, bool(ponder_mcts_res))
            use_mcts_enabled = bool(search_profile['use_mcts'])
            self.log(self.format_log_columns([f'ply={mvs}', f'empties={empty}', f"exact={('ON' if is_exact else 'OFF')}", f"C++={('ON' if self.use_cpp_engine else 'OFF')}", f"NN={('ON' if self.use_nn else 'OFF')}", f"MCTS={('ON' if use_mcts_enabled else 'OFF')}", f"MCTS influence={('auto%' if self.auto_time else f'{self.mcts_influence_pct}%')}", f"TT={('ON' if self.use_tt_resume else 'OFF')}", f'limit={time_limit:.1f}s'], [8, 12, 10, 7, 6, 8, 20, 6, 11]))
            if book_move is not None:
                self.mark_modules_active('BOOK')
                self.mark_connections_active(('BOARD', 'BOOK'), ('BOOK', 'BLEND'))
                self.log(f'book: use {self.format_move_label(book_move)}')
            elif book_roll_used:
                base_rate = int(self.opening_book_rate * 100)
                dynamic_rate = int(dynamic_book_rate * 100)
                if dynamic_rate < base_rate:
                    reason = 'evaluation bias detected'
                elif dynamic_rate > base_rate:
                    reason = 'evaluation consensus'
                else:
                    reason = 'standard'
                self.log(f'book: hit but skipped by {dynamic_rate}% rule ({reason})')
            if is_resumed:
                self.log(f'tt: hit | start depth={start_dp}')
            else:
                self.log('tt: miss | start depth=2')
            self.log(f'legal moves={len(rms)}')
            if is_exact:
                self.log('scheduler: exact solve -> MCTS skipped, AB runs with time limit')
            else:
                self.log(self.format_log_columns([f"scheduler: mcts_priority={search_profile['mcts_priority']:.2f}", f"ab_delay={search_profile['ab_delay']:.2f}s", f"ab_budget={search_profile['ab_budget']:.2f}s"], [32, 18, 19]))

            def watchdog():
                nonlocal ai_start_time, actual_time_limit
                check_interval = 0.05
                while time.time() - ai_start_time < actual_time_limit:
                    if self.sf_arr[0] or not self.running or self.game_id != current_game_id:
                        return
                    time.sleep(check_interval)
                if time.time() - ai_start_time >= actual_time_limit:
                    self.sf_arr[0] = 1
            threading.Thread(target=watchdog, daemon=True).start()

            def run_mcts():
                nonlocal mcts_res, best_mcts_wr, root_visits, mcts_sim_count
                if use_mcts_enabled:
                    seed_mcts_res = dict(ponder_mcts_res)
                    seed_best_mcts_wr = ponder_best_mcts_wr
                    seed_root_visits = dict(ponder_root_visits)
                    seed_mcts_sim_count = ponder_mcts_sim_count
                    if seed_mcts_res:
                        self.mark_modules_active('PONDER', 'MCTS')
                        self.mark_connections_active(('PONDER', 'MCTS'), ('POLICY', 'MCTS'), ('VALUE', 'MCTS'))
                    
                    # MCTS枝刈りが有効な場合は事前探索時間を使用
                    if self.use_mcts_enabled and self.mcts_pruning_enabled:
                        mcts_time_limit = self.mcts_pruning_time
                        self.log(self.format_log_columns(['mcts: pruning start', f'pre-time={mcts_time_limit:.1f}s', f'branches={self.mcts_pruning_branches}'], [18, 16, 14]))
                    else:
                        mcts_time_limit = time_limit
                        self.log(self.format_log_columns(['mcts: start', f'limit={mcts_time_limit:.1f}s'], [11, 12]))
                    
                    mcts_batch_size = self._get_live_mcts_batch_size(mcts_time_limit)
                    self.mark_modules_active('MCTS', 'NN-IN', 'TRUNK', 'POLICY', 'VALUE')
                    self.mark_connections_active(('BOARD', 'NN-IN'), ('NN-IN', 'TRUNK'), ('TRUNK', 'POLICY'), ('TRUNK', 'VALUE'), ('POLICY', 'MCTS'), ('VALUE', 'MCTS'))
                    current_color = 'black' if self.tn == 1 else 'white'
                    self.call_on_ui_thread(self.rt.title, f'Sesamum - {current_color}')
                    mcts_res, best_mcts_wr, sim_count, nn_batch_count, nn_leaf_count, root_visits = get_mcts_win_rates_time_batched(self.nn_model, aB, oB, self.tn, mcts_time_limit, mcts_batch_size, self.sf_arr)
                    mcts_sim_count = sim_count
                    
                    # MCTS枝刈りが有効な場合は上位の手だけを選択
                    if self.use_mcts_enabled and self.mcts_pruning_enabled and mcts_res:
                        sorted_moves = sorted(mcts_res.items(), key=lambda x: x[1], reverse=True)
                        top_moves = sorted_moves[:self.mcts_pruning_branches]
                        mcts_res = dict(top_moves)
                        self.log(self.format_log_columns(['mcts: pruning applied', f'top moves={len(mcts_res)}', f'best={best_mcts_wr:.1f}%'], [20, 13, 12]))
                    if seed_mcts_res:
                        merged_scores = {}
                        merged_visits = {}
                        all_moves = set(seed_mcts_res.keys()) | set(mcts_res.keys()) | set(seed_root_visits.keys()) | set(root_visits.keys())
                        for move in all_moves:
                            live_visit = int(root_visits.get(move, 0))
                            seed_visit = int(seed_root_visits.get(move, 0))
                            total_visit = live_visit + seed_visit
                            merged_visits[move] = total_visit
                            if total_visit > 0:
                                merged_scores[move] = (mcts_res.get(move, 50.0) * live_visit + seed_mcts_res.get(move, 50.0) * seed_visit) / total_visit
                            elif move in mcts_res:
                                merged_scores[move] = mcts_res.get(move, 50.0)
                            else:
                                merged_scores[move] = seed_mcts_res.get(move, 50.0)
                        mcts_res = merged_scores
                        root_visits = merged_visits
                        mcts_sim_count += seed_mcts_sim_count
                        if root_visits:
                            best_move_by_visit = max(root_visits.items(), key=lambda x: x[1])[0]
                            best_mcts_wr = mcts_res.get(best_move_by_visit, best_mcts_wr)
                        else:
                            best_mcts_wr = max(best_mcts_wr, seed_best_mcts_wr)
                        self.log(self.format_log_columns(['mcts: merged ponder cache', f'sims={mcts_sim_count}'], [25, 14]))
                    top_visits = sorted(root_visits.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_visit_str = ' | '.join([f'{self.format_move_label(m)}={v}' for m, v in top_visits])
                    self.log(self.format_log_columns(['mcts: done', f'sims={mcts_sim_count}', f'nn_batches={nn_batch_count}', f'best={best_mcts_wr:.1f}%'], [10, 14, 16, 11]))
                    if top_visit_str:
                        self.log(f'mcts: root visits | {top_visit_str}')
                elif is_exact:
                    self.log('mcts: skipped during exact solve')
                else:
                    self.log('mcts: disabled')

            def run_ab():
                nonlocal ab_res, ab_val_res, resolved_flag
                if not use_ab:
                    if self.use_mcts_enabled and self.mcts_pruning_enabled:
                        self.log('ab: skipped (MCTS pruning mode)')
                    else:
                        self.log('ab: skipped (MCTS influence 100%)')
                    self._last_ab_val_res = {}
                    self._last_ab_completed_depth = 2
                    return
                # MCTS枝刈りが有効な場合はMCTS事前探索時間分だけ遅延
                if self.use_mcts_enabled and self.mcts_pruning_enabled:
                    ab_delay = self.mcts_pruning_time  # MCTS事前探索時間分だけ遅延
                    self.log(self.format_log_columns(['ab: delay for MCTS', f'wait={ab_delay:.1f}s'], [18, 13]))
                else:
                    ab_delay = float(search_profile['ab_delay'])
                
                if ab_delay > 0.0:
                    delay_deadline = ai_start_time + ab_delay
                    while time.time() < delay_deadline:
                        if self.sf_arr[0] or not self.running or self.game_id != current_game_id:
                            return
                        time.sleep(0.01)
                if is_exact:
                    self.log('[BLUE]Start Endgame Solver...[/BLUE]')
                # MCTS枝刈りが有効な場合は残り時間をAB探索に使用
                if self.use_mcts_enabled and self.mcts_pruning_enabled:
                    ab_time_limit = self.ab_pruning_time
                    ab_deadline = ai_start_time + ab_time_limit
                    self.log(self.format_log_columns(['ab: pruning mode', f'time={ab_time_limit:.1f}s'], [16, 12]))
                else:
                    ab_deadline = None if is_exact or search_profile['ab_budget'] is None else ai_start_time + float(search_profile['ab_budget'])
                self.log(self.format_log_columns(['ab: start', f'depth={max(start_dp, 2)}'], [9, 10]))
                self.mark_modules_active('αβ')
                self.mark_connections_active(('BOARD', 'TT'), ('TT', 'αβ'))
                ab_val_res = {}
                completed_depth = max(2, max(start_dp, 2) - 1)
                attempted_depth = completed_depth
                if self.use_cpp_engine and cpp_engine is not None:
                    curr_ordered = [int(x) for x in rms]
                    depth_start = max(start_dp, 2)
                    for dp in range(depth_start, 61):
                        attempted_depth = dp
                        depth_exact = bool(is_exact and dp >= empty)
                        if not self.running or self.game_id != current_game_id:
                            break
                        if self.sf_arr[0]:
                            break
                        remain_sec = actual_time_limit - (time.time() - ai_start_time)
                        if ab_deadline is not None:
                            remain_sec = min(remain_sec, ab_deadline - time.time())
                        remain_ms = int(remain_sec * 1000)
                        if remain_ms <= 1:
                            break
                        try:
                            if policy_scores:
                                status = cpp_engine.search_root_parallel_cached_status_policy(int(aB), int(oB), int(mvs), int(dp), bool(depth_exact), curr_ordered, root_policy_vector, remain_ms)
                            else:
                                status = cpp_engine.search_root_parallel_cached_status(int(aB), int(oB), int(mvs), int(dp), bool(depth_exact), curr_ordered, remain_ms)
                            vals = status['vals']
                            nodes = status['nodes']
                            if bool(status.get('timed_out', False)):
                                break
                        except TypeError:
                            try:
                                vals, nodes = cpp_engine.search_root_parallel(int(aB), int(oB), int(mvs), int(dp), bool(depth_exact), curr_ordered, weights_list, order_map_list, remain_ms)
                            except TypeError:
                                vals, nodes = cpp_engine.search_root_parallel(int(aB), int(oB), int(mvs), int(dp), bool(depth_exact), curr_ordered, weights_list, order_map_list)
                        if self.sf_arr[0]:
                            break
                        if len(curr_ordered) == 0:
                            break
                        combined = []
                        for i, m in enumerate(curr_ordered):
                            ab_wr = calculate_win_rate(float(vals[i]), depth_exact)
                            combined.append((int(m), float(vals[i]), ab_wr))
                        combined.sort(key=lambda x: x[2], reverse=True)
                        curr_ordered = [x[0] for x in combined]
                        for m, v, wr in combined:
                            ab_res[m] = wr
                            ab_val_res[m] = v
                        mx = combined[0][2]
                        es = [math.exp(max(-20.0, (c - mx) / 10.0)) for _, _, c in combined]
                        se = sum(es) if es else 1.0
                        probs_map = {m: e / se * 100.0 for (m, _, _), e in zip(combined, es)}
                        pr_str = '[*]' if is_resumed else '[]'
                        el = time.time() - ai_start_time
                        ds = combined[:5]
                        move_summary = self.format_top_moves([(r[0], r[2]) for r in ds], limit=5)
                        self.log(self.format_log_columns([f'{pr_str}ab[c++] depth={dp:2d}', f'best={mx:5.1f}%', f'time={el:4.1f}s', f'nodes={int(sum(nodes))}'], [21, 12, 11, 14]))
                        self.log(f'  moves:  {move_summary}')
                        if is_exact:
                            self.call_on_ui_thread(self.update_gui_from_ai, mx, probs_map, current_game_id)
                        completed_depth = dp
                        if depth_exact or abs(combined[0][1]) > 5000:
                            resolved_flag = bool(depth_exact or abs(combined[0][1]) > 5000)
                            if not use_mcts_enabled:
                                self.call_on_ui_thread(self.update_gui_from_ai, mx, probs_map, current_game_id)
                                time.sleep(0.1)
                                self.sf_arr[0] = 1
                            break
                    self._last_ab_val_res = ab_val_res
                    self._last_ab_completed_depth = completed_depth
                    if attempted_depth > completed_depth:
                        self.log(self.format_log_columns(['ab: done', f'completed={completed_depth}', f'attempted={attempted_depth}'], [8, 15, 15]))
                    else:
                        self.log(self.format_log_columns(['ab: done', f'completed={completed_depth}'], [8, 15]))
                    return
                curr_ordered = np.array(rms, dtype=np.int64)
                ab_val_res = {}
                completed_depth = max(2, max(start_dp, 2) - 1)
                attempted_depth = completed_depth
                for dp in range(start_dp, 61):
                    attempted_depth = dp
                    depth_exact = bool(is_exact and dp >= empty)
                    if not self.running or self.game_id != current_game_id:
                        break
                    if self.sf_arr[0]:
                        break
                    if not is_exact and ab_deadline is not None and (time.time() >= ab_deadline):
                        break
                    vals, nodes = search_root_parallel(aB, oB, np.int64(mvs), dp, depth_exact, curr_ordered, self.W_ary, self.order_map, self.tk_arr, self.ti_arr, self.sf_arr)
                    if self.sf_arr[0]:
                        break
                    if len(curr_ordered) == 0:
                        break
                    combined = []
                    for i, m in enumerate(curr_ordered):
                        ab_wr = calculate_win_rate(vals[i], depth_exact)
                        combined.append((m, vals[i], ab_wr))
                    combined.sort(key=lambda x: x[2], reverse=True)
                    curr_ordered = np.array([x[0] for x in combined], dtype=np.int64)
                    for m, v, wr in combined:
                        ab_res[m] = wr
                        ab_val_res[m] = v
                    mx = combined[0][2]
                    es = [math.exp(max(-20.0, (c - mx) / 10.0)) for _, _, c in combined]
                    se = sum(es) if es else 1.0
                    probs_map = {m: e / se * 100.0 for (m, _, _), e in zip(combined, es)}
                    pr_str = '[*]' if is_resumed else '[]'
                    el = time.time() - ai_start_time
                    ds = combined[:5]
                    move_summary = self.format_top_moves([(r[0], r[2]) for r in ds], limit=5)
                    self.log(self.format_log_columns([f'{pr_str}ab[py] depth={dp:2d}', f'best={mx:5.1f}%', f'time={el:4.1f}s', f'nodes={np.sum(nodes)}'], [21, 12, 11, 14]))
                    self.log(f'  moves:  {move_summary}')
                    if is_exact:
                        self.call_on_ui_thread(self.update_gui_from_ai, mx, probs_map, current_game_id)
                    completed_depth = dp
                    if depth_exact or abs(combined[0][1]) > 5000:
                        resolved_flag = bool(depth_exact or abs(combined[0][1]) > 5000)
                        if not use_mcts_enabled:
                            self.call_on_ui_thread(self.update_gui_from_ai, mx, probs_map, current_game_id)
                            time.sleep(0.1)
                            self.sf_arr[0] = 1
                            break
                self._last_ab_val_res = ab_val_res
                self._last_ab_completed_depth = completed_depth
                if attempted_depth > completed_depth:
                    self.log(self.format_log_columns(['ab: done', f'completed={completed_depth}', f'attempted={attempted_depth}'], [8, 15, 15]))
                else:
                    self.log(self.format_log_columns(['ab: done', f'completed={completed_depth}'], [8, 15]))
            cpp_session_res = None
            if self.use_cpp_engine and cpp_engine is not None and hasattr(cpp_engine, 'SearchSession') and self.use_nn and (self.nn_model is not None):
                if use_mcts_enabled and ponder_mcts_res:
                    self.mark_modules_active('PONDER', 'MCTS')
                    self.mark_connections_active(('PONDER', 'MCTS'), ('POLICY', 'MCTS'), ('VALUE', 'MCTS'))
                    self.log(self.format_log_columns(['mcts: ponder cache hit', f'sims={ponder_mcts_sim_count}'], [22, 14]))
                cpp_session_res = self._run_cpp_search_session(aB, oB, mvs, start_dp, is_exact, rms, root_policy_vector, use_ab, use_mcts_enabled, actual_time_limit, search_profile, current_game_id, is_resumed)
                if cpp_session_res is not None:
                    mcts_res = cpp_session_res['mcts_res']
                    best_mcts_wr = cpp_session_res['best_mcts_wr']
                    root_visits = cpp_session_res['root_visits']
                    mcts_sim_count = cpp_session_res['mcts_sim_count']
                    ab_res = cpp_session_res['ab_res']
                    ab_val_res = cpp_session_res.get('ab_val_res', ab_val_res)
                    resolved_flag = cpp_session_res['resolved_flag']
                    if ponder_mcts_res and mcts_res:
                        merged_scores = {}
                        merged_visits = {}
                        all_moves = set(ponder_mcts_res.keys()) | set(mcts_res.keys()) | set(ponder_root_visits.keys()) | set(root_visits.keys())
                        for move in all_moves:
                            live_visit = int(root_visits.get(move, 0))
                            seed_visit = int(ponder_root_visits.get(move, 0))
                            total_visit = live_visit + seed_visit
                            merged_visits[move] = total_visit
                            if total_visit > 0:
                                merged_scores[move] = (mcts_res.get(move, 50.0) * live_visit + ponder_mcts_res.get(move, 50.0) * seed_visit) / total_visit
                            elif move in mcts_res:
                                merged_scores[move] = mcts_res.get(move, 50.0)
                            else:
                                merged_scores[move] = ponder_mcts_res.get(move, 50.0)
                        mcts_res = merged_scores
                        root_visits = merged_visits
                        mcts_sim_count += ponder_mcts_sim_count
                        if root_visits:
                            best_move_by_visit = max(root_visits.items(), key=lambda x: x[1])[0]
                            best_mcts_wr = mcts_res.get(best_move_by_visit, best_mcts_wr)
                        else:
                            best_mcts_wr = max(best_mcts_wr, ponder_best_mcts_wr)
                        self.log(self.format_log_columns(['mcts: merged ponder cache', f'sims={mcts_sim_count}'], [25, 14]))
                        top_visits = sorted(root_visits.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_visit_str = ' | '.join([f'{self.format_move_label(m)}={v}' for m, v in top_visits])
                        if top_visit_str:
                            self.log(f'mcts: root visits | {top_visit_str}')
            if cpp_session_res is None:
                t_mcts = threading.Thread(target=run_mcts)
                t_ab = threading.Thread(target=run_ab)
                t_mcts.start()
                t_ab.start()
                t_mcts.join()
                t_ab.join()
            if not self.running or self.game_id != current_game_id:
                return
            mcts_reliable = True
            if use_mcts_enabled and mcts_res:
                max_root_visit = max(root_visits.values()) if root_visits else 0
                min_required_sims = max(160, len(rms) * 40)
                min_required_visit = max(24, len(rms) * 6)
                if mcts_sim_count < min_required_sims or max_root_visit < min_required_visit:
                    mcts_reliable = False
                    self.log(self.format_log_columns(['blend: ignore MCTS', f'sims={mcts_sim_count}', f'best_root_visit={max_root_visit}'], [18, 14, 20]))
                    mcts_res = {}
                    best_mcts_wr = 50.0
            max_depth_reached = getattr(self, '_last_ab_completed_depth', 2)
            if resolved_flag and is_exact:
                cached_move = self.cache_exact_result(aB, oB, self.tn, ab_val_res, ab_res)
                if cached_move:
                    self.log(f'exact result cached: {self.format_move_label(cached_move)}')
            mcts_influence_pct = self.adjust_mcts_influence_auto(mcts_sim_count, empty)
            mcts_influence = max(0.0, min(1.0, mcts_influence_pct / 100.0))
            if self.auto_time and mcts_influence_pct != self.mcts_influence_pct:
                self.log(f'mcts: dynamic influence adjusted from {self.mcts_influence_pct}% to {mcts_influence_pct}% based on {mcts_sim_count} sims')
            final_scores = []
            blend_details = {}
            for m in rms:
                if use_ab:
                    score_ab = ab_res.get(m, 50.0)
                    if resolved_flag:
                        # 読み切り完了時：100%確定手はそのまま使用、MCTSの影響を排除
                        final_score = score_ab
                        blend_details[m] = {'move': int(m), 'ab_weight': 1.0, 'mcts_weight': 0.0}
                    else:
                        score_mcts = mcts_res.get(m, 50.0) if use_mcts_enabled else score_ab
                        visits = root_visits.get(m, 0) if use_mcts_enabled else 0
                        w_ab, w_mcts = core.compute_blend_weights(mvs, empty, score_ab, score_mcts, resolved_flag, use_mcts_enabled, visits, max_depth_reached, mcts_sim_count, time_limit, DEVICE_STR)
                        w_mcts *= mcts_influence
                        w_ab = 1.0 - w_mcts
                        final_score = score_ab * w_ab + score_mcts * w_mcts
                        blend_details[m] = {'move': int(m), 'ab_weight': w_ab, 'mcts_weight': w_mcts}
                else:
                    score_ab = mcts_res.get(m, 50.0)
                    score_mcts = score_ab
                    final_score = score_mcts
                    blend_details[m] = {'move': int(m), 'ab_weight': 0.0, 'mcts_weight': 1.0}
                final_scores.append((m, final_score))
            final_scores.sort(key=lambda x: x[1], reverse=True)
            if use_ab:
                self.mark_connections_active(('αβ', 'BLEND'))
            if use_mcts_enabled:
                self.mark_connections_active(('MCTS', 'BLEND'))
            if final_scores:
                self.log('final: ' + self.format_top_moves(final_scores, limit=5))
                if use_ab and use_mcts_enabled and (not resolved_flag):
                    midpoints = []
                    for m in [move for move, _ in final_scores[:5]]:
                        if m in ab_res and m in mcts_res:
                            midpoint = (ab_res[m] + mcts_res[m]) / 2.0
                            midpoints.append((m, midpoint))
                    if midpoints:
                        green_text = '[GREEN]midpoint: ' + self.format_top_moves(midpoints, limit=5) + '[/GREEN]'
                        self.log(green_text)
            if best_move == -1 and final_scores:
                best_move = final_scores[0][0]
            if book_move is not None and primary_move == book_move:
                book_use_details = self.should_use_opening_book(mvs, empty, ab_result={'moves': list(ab_res.keys()), 'win_rates': list(ab_res.values())} if ab_res else None, mcts_result={'move_win_rates': mcts_res} if mcts_res else None, return_details=True)
                book_rate = book_use_details.get('rate', 1.0)
                book_reason = book_use_details.get('reason', 'standard')
                self.log(f'book: AB/MCTS evaluation - book_rate={int(book_rate * 100)}% ({book_reason})')
                if book_rate < 0.999:
                    self.log(f'book: Strong agreement between AB/MCTS - not using opening book')
                    best_move = final_scores[0][0] if final_scores else book_move
            if best_move != -1:
                if book_move is not None and best_move == book_move:
                    self.log(f'selected by book: {self.format_move_label(best_move)}')
                elif mvs == 0 and best_move in rms:
                    self.log(f'selected by standard: {self.format_move_label(best_move)}')
            if best_move == -1:
                return
            self.log(f'move: {self.format_move_label(best_move)}')
            blend_info = blend_details.get(best_move, {'move': int(best_move), 'ab_weight': 1.0, 'mcts_weight': 0.0})
            blend_info = dict(blend_info)
            blend_info['ply'] = mvs + 1
            self.blend_history.append(blend_info)
            self.blend_history = self.blend_history[-3:]
            chosen_final_score = next((float(sc) for move, sc in final_scores if move == best_move), 50.0)
            self.pending_blend_samples.append({'ply': int(mvs + 1), 'move': int(best_move), 'empty': int(empty), 'time_limit': float(time_limit), 'is_exact': bool(is_exact or resolved_flag), 'ab_weight': float(blend_info.get('ab_weight', 1.0)), 'mcts_weight': float(blend_info.get('mcts_weight', 0.0)), 'ab_score': float(ab_res.get(best_move, 50.0)) if use_ab else 50.0, 'mcts_score': float(mcts_res.get(best_move, 50.0)) if use_mcts_enabled else 50.0, 'final_score': chosen_final_score, 'mcts_visits': int(root_visits.get(best_move, 0)), 'mcts_sim_count': int(mcts_sim_count), 'ab_depth': int(max_depth_reached)})
            best_eval = getattr(self, '_last_ab_val_res', {}).get(best_move)
            graph_ab_wr = None
            if use_ab and best_eval is not None:
                self.last_win_rate = calculate_win_rate(best_eval, is_exact or resolved_flag)
                graph_ab_wr = self.last_win_rate
            elif use_ab:
                graph_ab_wr = self.last_win_rate if self.last_win_rate is not None else 50.0
            if resolved_flag or is_exact:
                self.last_mcts_win_rate = None
            else:
                self.last_mcts_win_rate = best_mcts_wr if use_mcts_enabled and mcts_res and mcts_reliable else None
            graph_mcts_wr = self.last_mcts_win_rate if self.last_mcts_win_rate is not None else None
            if self.last_win_rate is not None:
                if self.last_win_rate >= 99.999:
                    self.last_win_rate = 100.0
                elif self.last_win_rate <= 0.001:
                    self.last_win_rate = 0.0
            if self.last_mcts_win_rate is not None:
                if self.last_mcts_win_rate >= 99.999:
                    self.last_mcts_win_rate = 100.0
                elif self.last_mcts_win_rate <= 0.001:
                    self.last_mcts_win_rate = 0.0
                graph_mcts_wr = self.last_mcts_win_rate
            if graph_ab_wr is not None and graph_ab_wr >= 99.999:
                graph_ab_wr = 100.0
            elif graph_ab_wr is not None and graph_ab_wr <= 0.001:
                graph_ab_wr = 0.0
            if graph_mcts_wr is not None and graph_mcts_wr >= 99.999:
                graph_mcts_wr = 100.0
            elif graph_mcts_wr is not None and graph_mcts_wr <= 0.001:
                graph_mcts_wr = 0.0
            self.call_on_ui_thread(self.push_graph_point, min(64, mvs + 1), graph_ab_wr, graph_mcts_wr)
            self.call_on_ui_thread(self.update_title)
            next_player, next_opp = self.apply_move_pair(aB, oB, best_move)
            if acting_side == 1:
                self.B, self.W = (next_player, next_opp)
            else:
                self.W, self.B = (next_player, next_opp)
            self.tn = -self.tn
            self.call_on_ui_thread(self.drw)
            if resolved_flag:
                time.sleep(0.05)
        except Exception as exc:
            self.log(f'AI thread error: {type(exc).__name__}: {exc}')
        finally:
            if self.game_id == current_game_id:
                self.is_thinking = False
                if not follow_up_handled:
                    self.call_on_ui_thread(lambda: self.rt.after(50, self.chk))

    def chk(self):
        if not self.running:
            return
        if self.is_viewing_historical_position():
            if self.btn_pass:
                self.btn_pass.destroy()
                self.btn_pass = None
            return
        bM = self.legal_moves_mask(self.B, self.W)
        wM = self.legal_moves_mask(self.W, self.B)
        board_full = (self.B | self.W).bit_count() == 64
        if board_full or (not bM and (not wM)):
            b, w = (int(self.B).bit_count(), int(self.W).bit_count())
            if self.auto_battle:
                if b > w:
                    msg = 'black wins'
                elif w > b:
                    msg = 'white wins'
                else:
                    msg = 'draw'
            else:
                msg = 'win' if self.hc == 1 and b > w or (self.hc == -1 and w > b) else 'lose' if self.hc == 1 and b < w or (self.hc == -1 and w < b) else 'draw'
                ai_discs = b if self.ac == 1 else w
                human_discs = w if self.ac == 1 else b
                result_value = 1 if ai_discs > human_discs else -1 if ai_discs < human_discs else 0
                self.flush_blend_calibration_samples(result_value, ai_discs - human_discs)
            if not self.game_over_shown:
                self.game_over_shown = True
                messagebox.showinfo('finish', f'Black: {b}  White: {w}\n{msg}')
            return
        current_moves = self.legal_moves_mask(self.B if self.tn == 1 else self.W, self.W if self.tn == 1 else self.B)
        if not current_moves:
            if self.auto_battle:
                self.resolve_forced_pass()
            elif self.tn == self.hc and not self.auto_battle:
                # 人間の手番で、かつAI対戦モードでない場合のみパスボタンを表示
                self.show_pass_btn()
            else:
                self.resolve_forced_pass()
            return
        elif self.btn_pass:
            self.btn_pass.destroy()
            self.btn_pass = None
        if self.auto_battle or self.tn == self.ac:
            if not self.is_thinking:
                self.is_thinking = True
                threading.Thread(target=self.ai_r, args=(self.game_id,), daemon=True).start()
        else:
            self.drw()
            if not self.auto_battle:
                self.start_pondering()
if __name__ == '__main__':
    r = tk.Tk()
    app = UltimateOthello(r)
    r.mainloop()

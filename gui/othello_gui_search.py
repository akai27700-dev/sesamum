import math
import os
import sys
import threading
import time

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.othello_core as core

DEVICE_STR = core.DEVICE_STR
TT_SIZE = core.TT_SIZE
_FULL_MASK = core._FULL_MASK
calculate_win_rate = core.calculate_win_rate
cpp_engine = core.cpp_engine
ensure_numba_warmup = core.ensure_numba_warmup
get_flip = core.get_flip
get_mcts_win_rates_time_batched = core.get_mcts_win_rates_time_batched
get_rotated_bitboard = core.get_rotated_bitboard
make_input_tensor = core.make_input_tensor
nn_infer_batch = core.nn_infer_batch
search_root_parallel = core.search_root_parallel
unrotate_move = core.unrotate_move
zobrist_hash = core.zobrist_hash


class OthelloSearchMixin:
    def make_state_key(self, p_board, o_board, turn):
        return f"{int(p_board)}_{int(o_board)}_{int(turn)}"

    def apply_move_pair(self, player_board, opp_board, move_idx):
        if self.use_cpp_engine and cpp_engine is not None:
            try:
                next_player, next_opp = cpp_engine.apply_move(int(player_board), int(opp_board), int(move_idx))
                return np.uint64(next_player), np.uint64(next_opp)
            except Exception:
                pass
        flip = np.uint64(get_flip(player_board, opp_board, np.int64(move_idx)))
        next_player = (player_board | (np.uint64(1) << np.uint64(move_idx)) | flip) & _FULL_MASK
        next_opp = (opp_board ^ flip) & _FULL_MASK
        return next_player, next_opp

    def should_stop_pondering(self, current_game_id, ponder_token, ponder_sf, force_stop=False):
        if force_stop:
            return True
        stop_flag = int(ponder_sf[0]) != 0 if isinstance(ponder_sf, np.ndarray) else bool(ponder_sf[0])
        return (
            (not self.running)
            or self.game_id != current_game_id
            or self.ponder_token != ponder_token
            or self.tn == self.hc  # AIの手番の時だけ停止（相手ターン中は続ける）
            or stop_flag
        )

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
        if (not self.running) or self.game_id != current_game_id or self.ponder_token != ponder_token or self.tn != self.hc:
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
            root_moves = list(cpp_move_analysis.get("moves", []))
            next_p_list = list(cpp_move_analysis.get("next_p", []))
            next_o_list = list(cpp_move_analysis.get("next_o", []))
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
            candidates.append(
                {
                    "move": move,
                    "human_after": human_after,
                    "ai_after": ai_after,
                    "score": total_score,
                    "αβ": aB,
                    "oB": oB,
                    "legal_indices": legal_indices,
                    "mvs": mvs,
                    "empty": empty,
                    "is_exact": is_exact,
                    "root_policy_vector": root_policy_vector,
                    "cache_key": self.make_state_key(aB, oB, self.ac),
                }
            )
        candidates.sort(key=lambda x: -x["score"])
        return candidates[:3]

    def merge_ponder_cache_entry(self, cache_key, ordered_moves=None, completed_depth=None, mcts_res=None, best_mcts_wr=None, root_visits=None, mcts_sim_count=None):
        with self.ponder_lock:
            entry = self.ponder_cache.get(cache_key)
            if entry is None:
                entry = {
                    "ordered_moves": [],
                    "completed_depth": 2,
                    "mcts_res": {},
                    "best_mcts_wr": 50.0,
                    "root_visits": {},
                    "mcts_sim_count": 0,
                }
            if ordered_moves:
                entry["ordered_moves"] = list(ordered_moves)
            if completed_depth is not None:
                entry["completed_depth"] = max(int(entry.get("completed_depth", 2)), int(completed_depth))
            if mcts_res:
                previous_sim_count = int(entry.get("mcts_sim_count", 0))
                if mcts_sim_count is None or int(mcts_sim_count) >= previous_sim_count:
                    entry["mcts_res"] = dict(mcts_res)
                    entry["best_mcts_wr"] = float(best_mcts_wr if best_mcts_wr is not None else entry.get("best_mcts_wr", 50.0))
                    entry["root_visits"] = dict(root_visits or {})
                    entry["mcts_sim_count"] = int(mcts_sim_count if mcts_sim_count is not None else previous_sim_count)
            self.ponder_cache[cache_key] = entry

    def stop_pondering_explicitly(self):
        """明示的にponderingを停止（exact solve時などに使用）"""
        with self.ponder_lock:
            if self.ponder_sf is not None and isinstance(self.ponder_sf, np.ndarray):
                self.ponder_sf[0] = 1
        self.ponder_token += 1
        if self.ponder_mcts_thread and self.ponder_mcts_thread.is_alive():
            self.ponder_mcts_thread.join(timeout=0.1)
        if self.ponder_ab_thread and self.ponder_ab_thread.is_alive():
            self.ponder_ab_thread.join(timeout=0.1)
        self.ponder_mcts_thread = None
        self.ponder_ab_thread = None
        self.ponder_thread = None
        with self.ponder_lock:
            self.ponder_active_token = -1

    def finish_ponder_worker(self, worker_name, ponder_token):
        with self.ponder_lock:
            if worker_name == "mcts":
                if self.ponder_mcts_thread and self.ponder_mcts_thread.is_alive():
                    # Don't join if this is the current thread
                    if self.ponder_mcts_thread != threading.current_thread():
                        self.ponder_mcts_thread.join(timeout=0.5)
                self.ponder_mcts_thread = None
                self.ponder_thread = None
            elif worker_name == "αβ":
                if self.ponder_ab_thread and self.ponder_ab_thread.is_alive():
                    # Don't join if this is the current thread
                    if self.ponder_ab_thread != threading.current_thread():
                        self.ponder_ab_thread.join(timeout=0.5)
                self.ponder_ab_thread = None
            if self.ponder_mcts_thread is None and self.ponder_ab_thread is None and self.ponder_active_token == ponder_token:
                self.ponder_active_token = -1

    def start_pondering(self):
        if not self.running or self.is_thinking or self.tn != self.hc or not getattr(self, 'use_pondering', True):
            return
        token = self.ponder_token
        run_mcts_worker = self.use_mcts_enabled and self.use_nn and self.nn_model is not None
        if not run_mcts_worker and not (self.use_cpp_engine and cpp_engine is not None):
            return
        with self.ponder_lock:
            mcts_alive = self.ponder_mcts_thread is not None and self.ponder_mcts_thread.is_alive()
            ab_alive = self.ponder_ab_thread is not None and self.ponder_ab_thread.is_alive()
            if self.ponder_active_token == token and (mcts_alive or ab_alive):
                return
        self.ponder_sf = np.zeros(1, dtype=np.uint8)
        candidates = self.build_ponder_candidates(self.game_id, token)
        if not candidates:
            return
        # 終盤（全候補がis_exact）ではponderingをスキップ
        all_exact = all(candidate["is_exact"] for candidate in candidates)
        if all_exact:
            self.log("ponder: skipped (all candidates are exact)")
            return
        run_ab_worker = self.use_cpp_engine and cpp_engine is not None and ((not self.use_mcts_only) or any(candidate["is_exact"] for candidate in candidates))
        self.ponder_active_token = token
        self.log("Pondering start")
        self.mark_modules_active("PONDER")
        self.mark_connections_active(("PONDER", "αβ"), ("PONDER", "MCTS"), ("PONDER", "TT"))
        if run_mcts_worker:
            self.ponder_mcts_thread = threading.Thread(target=self.ponder_mcts_r, args=(self.game_id, token, self.ponder_sf, candidates), daemon=True)
            self.ponder_thread = self.ponder_mcts_thread
            self.ponder_mcts_thread.start()
        else:
            self.ponder_mcts_thread = None
            self.ponder_thread = None
        if run_ab_worker:
            self.ponder_ab_thread = threading.Thread(target=self.ponder_ab_r, args=(self.game_id, token, self.ponder_sf, candidates), daemon=True)
            self.ponder_ab_thread.start()
        else:
            self.ponder_ab_thread = None

    def ponder_mcts_r(self, current_game_id, ponder_token, ponder_sf, candidates):
        try:
            self.mark_modules_active("PONDER", "MCTS", "NN-IN", "TRUNK", "POLICY", "VALUE")
            self.mark_connections_active(("PONDER", "MCTS"), ("BOARD", "NN-IN"), ("NN-IN", "TRUNK"), ("TRUNK", "POLICY"), ("TRUNK", "VALUE"), ("POLICY", "MCTS"), ("VALUE", "MCTS"))
            if DEVICE_STR == "cuda":
                cycle_budget = max(2.2, self.time_limit_sec * (0.70 if self.light_mode else 1.10))
                cycle_budget = min(10.0, cycle_budget)
                warm_slice = 0.35 if self.light_mode else min(0.90, max(0.45, self.time_limit_sec * 0.07))
            else:
                cycle_budget = max(0.9, self.time_limit_sec * (0.45 if self.light_mode else 0.70))
                cycle_budget = min(4.0, cycle_budget)
                warm_slice = 0.18 if self.light_mode else min(0.45, max(0.20, self.time_limit_sec * 0.05))
            candidate_weights = (0.58, 0.27, 0.15)
            warm_rounds = 2 if self.time_limit_sec >= 5.0 else 1
            round_idx = 0
            while not self.should_stop_pondering(current_game_id, ponder_token, ponder_sf):
                for rank, candidate in enumerate(candidates):
                    if self.should_stop_pondering(current_game_id, ponder_token, ponder_sf):
                        return
                    if candidate["is_exact"]:
                        continue
                    legal_indices = list(candidate["legal_indices"])
                    if not legal_indices:
                        continue
                    curr_ordered = legal_indices
                    if round_idx < warm_rounds:
                        candidate_time = warm_slice
                    else:
                        candidate_time = cycle_budget * candidate_weights[min(rank, len(candidate_weights) - 1)]
                    if candidate_time <= 0.05:
                        continue
                    try:
                        if self.light_mode:
                            ponder_batch_size = 4096 if DEVICE_STR == "cuda" else 192
                        else:
                            if DEVICE_STR == "cuda":
                                if self.time_limit_sec >= 15.0:
                                    ponder_batch_size = 16384
                                elif self.time_limit_sec >= 10.0:
                                    ponder_batch_size = 12288
                                else:
                                    ponder_batch_size = 8192
                            else:
                                ponder_batch_size = 768
                        cached_mcts_res, cached_best_mcts_wr, cached_mcts_sim_count, _, _, cached_root_visits = get_mcts_win_rates_time_batched(
                            self.nn_model,
                            candidate["αβ"],
                            candidate["oB"],
                            self.ac,
                            candidate_time,
                            ponder_batch_size,
                            ponder_sf,
                        )
                    except Exception:
                        cached_mcts_res = {}
                        cached_best_mcts_wr = 50.0
                        cached_root_visits = {}
                        cached_mcts_sim_count = 0
                    if cached_mcts_res:
                        curr_ordered = sorted(curr_ordered, key=lambda move: cached_mcts_res.get(move, 50.0), reverse=True)
                        self.merge_ponder_cache_entry(
                            candidate["cache_key"],
                            ordered_moves=curr_ordered,
                            mcts_res=cached_mcts_res,
                            best_mcts_wr=cached_best_mcts_wr,
                            root_visits=cached_root_visits,
                            mcts_sim_count=cached_mcts_sim_count,
                        )
                round_idx += 1
        except Exception:
            pass
        finally:
            self.finish_ponder_worker("mcts", ponder_token)

    def ponder_ab_r(self, current_game_id, ponder_token, ponder_sf, candidates):
        try:
            self.mark_modules_active("PONDER", "αβ")
            self.mark_connections_active(("PONDER", "αβ"), ("PONDER", "TT"), ("TT", "αβ"))
            if self.light_mode:
                cycle_budget = max(0.8, self.time_limit_sec * 0.50)
                cycle_budget = min(2.4, cycle_budget)
                warm_slice = 0.12
            else:
                cycle_budget = max(1.4, self.time_limit_sec * 0.85)
                cycle_budget = min(6.0, cycle_budget)
                warm_slice = min(0.40, max(0.15, self.time_limit_sec * 0.03))
            candidate_weights = (0.56, 0.28, 0.16)
            warm_rounds = 2 if self.time_limit_sec >= 5.0 else 1
            round_idx = 0
            while not self.should_stop_pondering(current_game_id, ponder_token, ponder_sf):
                for rank, candidate in enumerate(candidates):
                    if self.should_stop_pondering(current_game_id, ponder_token, ponder_sf):
                        return
                    if self.use_mcts_only and not candidate["is_exact"]:
                        continue
                    legal_indices = list(candidate["legal_indices"])
                    if not legal_indices:
                        continue
                    if round_idx < warm_rounds:
                        candidate_time = warm_slice
                    else:
                        candidate_time = cycle_budget * candidate_weights[min(rank, len(candidate_weights) - 1)]
                    if candidate_time <= 0.05:
                        continue
                    with self.ponder_lock:
                        cached_entry = self.ponder_cache.get(candidate["cache_key"], {})
                    curr_ordered = [move for move in cached_entry.get("ordered_moves", []) if move in legal_indices]
                    if curr_ordered:
                        ordered_set = set(curr_ordered)
                        curr_ordered.extend([move for move in legal_indices if move not in ordered_set])
                    else:
                        curr_ordered = legal_indices
                    completed_depth = max(2, int(cached_entry.get("completed_depth", 2)))
                    depth_start = 2 if completed_depth <= 2 else (completed_depth + 1)
                    start_t = time.time()
                    for dp in range(depth_start, 61):
                        if self.should_stop_pondering(current_game_id, ponder_token, ponder_sf):
                            return
                        remain_ms = max(1, int((candidate_time - (time.time() - start_t)) * 1000))
                        if remain_ms <= 1:
                            break
                        try:
                            if any(candidate["root_policy_vector"]):
                                status = cpp_engine.search_root_parallel_cached_status_policy(
                                    int(candidate["αβ"]),
                                    int(candidate["oB"]),
                                    int(candidate["mvs"]),
                                    int(dp),
                                    bool(candidate["is_exact"]),
                                    curr_ordered,
                                    candidate["root_policy_vector"],
                                    remain_ms,
                                )
                            else:
                                status = cpp_engine.search_root_parallel_cached_status(
                                    int(candidate["αβ"]),
                                    int(candidate["oB"]),
                                    int(candidate["mvs"]),
                                    int(dp),
                                    bool(candidate["is_exact"]),
                                    curr_ordered,
                                    remain_ms,
                                )
                            vals = status["vals"]
                            if bool(status.get("timed_out", False)):
                                break
                            
                            # Pondering中の深度ログを出力（間引き）
                            elapsed = time.time() - start_t
                            nodes_raw = status.get("nodes", 0)
                            if isinstance(nodes_raw, list):
                                nodes_sum = sum(int(n) for n in nodes_raw)
                            else:
                                nodes_sum = int(nodes_raw)
                            best_move = curr_ordered[0] if curr_ordered else -1
                            best_val = vals[0] if vals else 0.0
                            best_wr = calculate_win_rate(best_val, candidate["is_exact"])
                            
                            # 深度2-5は毎回、6以降は3回に1回、10以降は5回に1回
                            if dp <= 5 or (dp <= 9 and dp % 3 == 0) or (dp >= 10 and dp % 5 == 0):
                                move_label = self.format_move_label(candidate["move"])
                                self.log(f"ponder: αβ[c++] depth={dp:2d}   | best={best_wr:5.1f}%  | time={elapsed:4.1f}s  | nodes={nodes_sum:,} | move=({move_label})")
                                if len(curr_ordered) > 1:
                                    moves_str = " | ".join([f"({self.format_move_label(m)}) {calculate_win_rate(vals[i], candidate['is_exact']):5.1f}%" for i, m in enumerate(curr_ordered[:5])])
                                    self.log(f"  moves: {moves_str}")
                                
                        except Exception as e:
                            # エラーはログせずにbreak
                            break
                        combined = []
                        for i, move in enumerate(curr_ordered):
                            combined.append((int(move), float(vals[i]), calculate_win_rate(float(vals[i]), candidate["is_exact"])))
                        combined.sort(key=lambda x: x[2], reverse=True)
                        curr_ordered = [x[0] for x in combined]
                        completed_depth = dp
                        self.merge_ponder_cache_entry(candidate["cache_key"], ordered_moves=curr_ordered, completed_depth=completed_depth)
                        if combined and (candidate["is_exact"] or abs(combined[0][1]) > 5000):
                            break
                round_idx += 1
        except Exception:
            # 外部例外もログせず
            pass
        finally:
            self.finish_ponder_worker("αβ", ponder_token)

    def update_gui_from_ai(self, win_rate, probs_map, current_game_id):
        if self.game_id != current_game_id:
            return
        self.last_win_rate = win_rate
        self.update_title()
        if self.tn == self.ac:
            self.drw(probs_map)

    def _get_live_mcts_batch_size(self, time_limit):
        if self.light_mode:
            return 8192 if DEVICE_STR == "cuda" else 384
        if DEVICE_STR == "cuda":
            if time_limit >= 15.0:
                return 65536
            if time_limit >= 10.0:
                return 49152
            if time_limit >= 5.0:
                return 32768
            return 16384
        return 1536 if time_limit >= 10.0 else 1024

    def _run_cpp_search_session(self, aB, oB, mvs, start_dp, is_exact, rms, root_policy_vector, use_ab, use_mcts_enabled, time_limit, search_profile, current_game_id, is_resumed):
        if cpp_engine is None or not hasattr(cpp_engine, "SearchSession"):
            return None

        def infer_leaves(leaves):
            tensor_batch_local = make_input_tensor(leaves)
            p_leaf_local, v_leaf_local = nn_infer_batch(self.nn_model, tensor_batch_local)
            return np.asarray(p_leaf_local, dtype=np.float32), np.asarray(v_leaf_local[:, 0], dtype=np.float32)

        def ab_progress(info):
            if not self.running or self.game_id != current_game_id:
                return
            moves = [int(x) for x in info.get("moves", [])]
            values = [float(x) for x in info.get("values", [])]
            win_rates = [float(x) for x in info.get("win_rates", [])]
            if not moves:
                return
            combined = list(zip(moves, values, win_rates))
            mx = combined[0][2]
            es = [math.exp(max(-20.0, (c - mx) / 10.0)) for _, _, c in combined]
            se = sum(es) if es else 1.0
            probs_map = {m: (e / se) * 100.0 for (m, _, _), e in zip(combined, es)}
            pr_str = "[*]" if is_resumed else "[]"
            depth = int(info.get("depth", 0))
            elapsed = float(info.get("elapsed_sec", 0.0))
            nodes = int(info.get("nodes", 0))
            move_summary = self.format_top_moves([(r[0], r[2]) for r in combined[:5]], limit=5)
            self.log(self.format_log_columns([
                f"{pr_str}αβ[c++] depth={depth:2d}",
                f"best={mx:5.1f}%",
                f"time={elapsed:4.1f}s",
                f"nodes={nodes}",
            ], [21, 12, 11, 14]))
            self.log(f"  moves: {move_summary}")
            self.call_on_ui_thread(self.update_gui_from_ai, mx, probs_map, current_game_id)

        if use_mcts_enabled:
            self.log(self.format_log_columns([
                "mcts: start",
                f"batch={self._get_live_mcts_batch_size(time_limit)}",
                f"limit={time_limit:.1f}s",
            ], [11, 15, 12]))
            self.mark_modules_active("MCTS", "NN-IN", "TRUNK", "POLICY", "VALUE")
            self.mark_connections_active(("BOARD", "NN-IN"), ("NN-IN", "TRUNK"), ("TRUNK", "POLICY"), ("TRUNK", "VALUE"), ("POLICY", "MCTS"), ("VALUE", "MCTS"))
            ai_color = "black" if self.ac == -1 else "white"
            self.call_on_ui_thread(self.rt.title, f"GeNeLy - {ai_color}")
        else:
            self.log("mcts: skipped during exact solve" if is_exact else "mcts: disabled")

        if use_ab:
            self.log(self.format_log_columns(["ab: start", f"depth={max(start_dp, 2)}"], [9, 10]))
            self.mark_modules_active("αβ")
            self.mark_connections_active(("BOARD", "TT"), ("TT", "αβ"))
        else:
            self.log("ab: skipped (MCTS influence 100%)")

        session = cpp_engine.SearchSession()
        ab_budget_ms = 0 if is_exact or search_profile["ab_budget"] is None else int(float(search_profile["ab_budget"]) * 1000)
        result = session.run(
            int(aB),
            int(oB),
            int(self.tn),
            int(mvs),
            int(max(start_dp, 2)),
            bool(is_exact),
            [int(x) for x in rms],
            [float(x) for x in root_policy_vector],
            bool(use_ab),
            bool(use_mcts_enabled),
            float(time_limit),
            int(self._get_live_mcts_batch_size(time_limit)),
            0.0 if is_exact else float(search_profile["ab_delay"]),
            int(ab_budget_ms),
            60,
            False,
            self.sf_arr,
            infer_leaves,
            ab_progress if use_ab else None,
        )
        ab_out = dict(result.get("ab", {}))
        mcts_out = dict(result.get("mcts", {}))
        ab_res = {}
        ab_val_res = {}
        for move, value, win_rate in zip(ab_out.get("moves", []), ab_out.get("values", []), ab_out.get("win_rates", [])):
            ab_res[int(move)] = float(win_rate)
            ab_val_res[int(move)] = float(value)
        completed_depth = int(ab_out.get("completed_depth", max(2, start_dp - 1)))
        attempted_depth = int(ab_out.get("attempted_depth", completed_depth))
        self._last_ab_val_res = ab_val_res if use_ab else {}
        self._last_ab_completed_depth = completed_depth if use_ab else 2
        if use_ab:
            if attempted_depth > completed_depth:
                self.log(self.format_log_columns(["ab: done", f"completed={completed_depth}", f"attempted={attempted_depth}"], [8, 15, 15]))
            else:
                self.log(self.format_log_columns(["ab: done", f"completed={completed_depth}"], [8, 15]))
        mcts_res = {int(k): float(v) for k, v in dict(mcts_out.get("move_win_rates", {})).items()}
        root_visits = {int(k): int(v) for k, v in dict(mcts_out.get("root_visits", {})).items()}
        best_mcts_wr = float(mcts_out.get("best_wr", 50.0))
        mcts_sim_count = int(mcts_out.get("simulation_count", 0))
        if use_mcts_enabled:
            nn_batch_count = int(mcts_out.get("nn_batch_count", 0))
            top_visits = sorted(root_visits.items(), key=lambda x: x[1], reverse=True)[:5]
            top_visit_str = " | ".join([f"{self.format_move_label(m)}={v}" for m, v in top_visits])
            self.log(self.format_log_columns([
                "mcts: done",
                f"sims={mcts_sim_count}",
                f"nn_batches={nn_batch_count}",
                f"best={best_mcts_wr:.1f}%",
            ], [10, 14, 16, 11]))
            if top_visit_str:
                self.log(f"mcts: root visits | {top_visit_str}")
        return {
            "mcts_res": mcts_res,
            "best_mcts_wr": best_mcts_wr,
            "root_visits": root_visits,
            "mcts_sim_count": mcts_sim_count,
            "ab_res": ab_res,
            "resolved_flag": bool(ab_out.get("resolved", False)),
        }

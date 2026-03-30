from typing import Sequence

class BatchMCTS:
    def __init__(self, c_puct: float = 2.0, virtual_loss: float = 1.0) -> None: ...
    def initialize_root(self, p: int, o: int, turn: int, policy_logits, add_noise: bool = False) -> None: ...
    def collect_leaves(self, p: int, o: int, turn: int, batch_size: int, stop_flag) -> dict: ...
    def expand_leaves(self, tickets: Sequence[int], policy_batch, value_batch) -> None: ...
    def collect_and_expand(self, p: int, o: int, turn: int, batch_size: int, stop_flag, infer_batch) -> dict: ...
    def root_stats(self, p: int, o: int, turn: int) -> dict: ...

class SearchSession:
    def __init__(self, c_puct: float = 2.0, virtual_loss: float = 1.0) -> None: ...
    def run(
        self,
        p: int,
        o: int,
        turn: int,
        mvs: int,
        start_depth: int,
        is_exact: bool,
        ordered_indices: Sequence[int],
        root_policy: Sequence[float],
        use_ab: bool,
        use_mcts: bool,
        time_limit_sec: float,
        mcts_batch_size: int,
        ab_delay_sec: float,
        ab_time_limit_ms: int,
        max_depth: int = 60,
        add_root_noise: bool = False,
        stop_flag = ...,
        infer_batch = ...,
        ab_progress = ...,
    ) -> dict: ...

def clear_tt() -> None: ...
def set_eval_data(weights: Sequence[float], order_map: Sequence[float]) -> None: ...
def probe_tt(p: int, o: int) -> dict: ...
def get_legal_moves(p: int, o: int) -> int: ...
def legal_move_indices(p: int, o: int) -> list[int]: ...
def apply_move(p: int, o: int, idx: int) -> tuple[int, int]: ...
def get_flip(p: int, o: int, idx: int) -> int: ...
def evaluate_board_full(p: int, o: int, mvs: int, weights: Sequence[float]) -> float: ...
def evaluate_board_cached(p: int, o: int, mvs: int) -> float: ...
def evaluate_moves(p: int, o: int, mvs: int, indices: Sequence[int], weights: Sequence[float]) -> list[float]: ...
def evaluate_moves_cached(p: int, o: int, mvs: int, indices: Sequence[int]) -> list[float]: ...
def analyze_legal_moves_cached(p: int, o: int, mvs: int) -> dict: ...
def search_root_parallel(
    p: int,
    o: int,
    mvs: int,
    depth: int,
    is_exact: bool,
    ordered_indices: Sequence[int],
    weights: Sequence[float],
    order_map: Sequence[float],
    time_limit_ms: int = 0,
) -> tuple[list[float], list[int]]: ...
def search_root_parallel_cached(
    p: int,
    o: int,
    mvs: int,
    depth: int,
    is_exact: bool,
    ordered_indices: Sequence[int],
    time_limit_ms: int = 0,
) -> tuple[list[float], list[int]]: ...
def search_root_parallel_cached_status(
    p: int,
    o: int,
    mvs: int,
    depth: int,
    is_exact: bool,
    ordered_indices: Sequence[int],
    time_limit_ms: int = 0,
) -> dict: ...
def search_root_parallel_cached_status_policy(
    p: int,
    o: int,
    mvs: int,
    depth: int,
    is_exact: bool,
    ordered_indices: Sequence[int],
    root_policy: Sequence[float],
    time_limit_ms: int = 0,
) -> dict: ...
def should_use_early_exact(p: int, o: int, empties: int, base_threshold: int) -> bool: ...
def get_best_move_ab(
    p: int,
    o: int,
    mvs: int,
    max_depth: int,
    is_exact: bool,
    ordered_indices: Sequence[int],
    weights: Sequence[float],
    order_map: Sequence[float],
    time_limit_ms: int = 0,
) -> dict: ...

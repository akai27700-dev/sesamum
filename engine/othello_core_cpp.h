#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

double evaluate_board_full_cpp(
    uint64_t P,
    uint64_t O,
    int32_t mvs,
    const double* W
);

void evaluate_board_batch_cpp(
    const uint64_t* P_arr,
    const uint64_t* O_arr,
    const int32_t* mvs_arr,
    const double* W,
    double* results,
    int32_t count
);

uint64_t get_legal_moves_cpp(uint64_t P, uint64_t O);
uint64_t get_flip_cpp(uint64_t P, uint64_t O, int32_t idx);
uint64_t compute_strict_stable_cpp(uint64_t P, uint64_t O);
int32_t engine_has_openmp();
int32_t engine_get_openmp_max_threads();
int32_t engine_get_openmp_thread_count();
void engine_set_openmp_threads(int32_t thread_count);

// Endgame solver functions
int solve_endgame_exact(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms
);

int get_endgame_best_move(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms
);

int solve_endgame_exact_status(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms,
    int32_t* out_best_move,
    int32_t* out_score,
    int32_t* out_completed_depth
);

#ifdef __cplusplus
}
#endif

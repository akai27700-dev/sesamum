#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Main C++ evaluation function
double evaluate_board_full_cpp(
    uint64_t P,
    uint64_t O,
    int32_t mvs,
    const double* W
);

// Batch evaluation for parallel processing
void evaluate_board_batch_cpp(
    const uint64_t* P_arr,
    const uint64_t* O_arr,
    const int32_t* mvs_arr,
    const double* W,
    double* results,
    int32_t count
);

// Helper functions exposed for Python bindings
uint64_t get_legal_moves_cpp(uint64_t P, uint64_t O);
uint64_t get_flip_cpp(uint64_t P, uint64_t O, int32_t idx);
uint64_t compute_strict_stable_cpp(uint64_t P, uint64_t O);

#ifdef __cplusplus
}
#endif

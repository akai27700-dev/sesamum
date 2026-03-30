#include "othello_core_cpp.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <intrin.h>

// Bitboard masks (from Python version)
constexpr uint64_t FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint64_t NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL;
constexpr uint64_t NOT_H_FILE = 0x7F7F7F7F7F7F7FULL;

constexpr uint64_t MASK_CORNER = 0x8100000000000081ULL;
constexpr uint64_t MASK_X = 0x0042000000004200ULL;
constexpr uint64_t MASK_C = 0x4281000000008142ULL;
constexpr uint64_t MASK_B1 = 0x0F0F0F0F00000000ULL;
constexpr uint64_t MASK_B2 = 0xF0F0F0F000000000ULL;
constexpr uint64_t MASK_B3 = 0x000000000F0F0F0FULL;
constexpr uint64_t MASK_B4 = 0x00000000F0F0F0F0ULL;

// Hardware POPCNT - extremely fast bit counting using CPU instruction
__forceinline int32_t count_bits(uint64_t x) {
    return static_cast<int32_t>(_mm_popcnt_u64(x));
}

// Least significant bit
__forceinline uint64_t lsb(uint64_t x) {
    return x & (~x + 1);
}

// Get bit index from LSB using hardware CTZ (count trailing zeros)
__forceinline int32_t bit_index(uint64_t x) {
    return static_cast<int32_t>(_tzcnt_u64(x));
}

// Neighbor union - highly optimized with forceinline
__forceinline uint64_t neighbor_union(uint64_t bb) {
    uint64_t n = 0;
    n |= (bb << 1) & NOT_A_FILE;
    n |= (bb >> 1) & NOT_H_FILE;
    n |= (bb << 8) & FULL_MASK;
    n |= (bb >> 8) & FULL_MASK;
    n |= (bb << 7) & NOT_A_FILE;
    n |= (bb >> 7) & NOT_H_FILE;
    n |= (bb << 9) & NOT_H_FILE;
    n |= (bb >> 9) & NOT_A_FILE;
    return n & FULL_MASK;
}

// Get legal moves (simplified - calls Python fallback for now)
extern "C" uint64_t get_legal_moves_cpp(uint64_t P, uint64_t O) {
    uint64_t occ = (P | O) & FULL_MASK;
    uint64_t legal = 0;
    
    for (int32_t idx = 0; idx < 64; idx++) {
        uint64_t bit = 1ULL << idx;
        if (occ & bit) continue;
        
        int32_t r = idx / 8;
        int32_t c = idx % 8;
        bool found = false;
        
        const int32_t dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
        const int32_t dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
        
        for (int32_t d = 0; d < 8 && !found; d++) {
            int32_t rr = r + dr[d];
            int32_t cc = c + dc[d];
            bool seen_opponent = false;
            
            while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
                int32_t nidx = rr * 8 + cc;
                uint64_t nbit = 1ULL << nidx;
                
                if (O & nbit) {
                    seen_opponent = true;
                    rr += dr[d];
                    cc += dc[d];
                    continue;
                }
                if (seen_opponent && (P & nbit)) {
                    found = true;
                }
                break;
            }
        }
        if (found) legal |= bit;
    }
    return legal & FULL_MASK;
}

// Get flip bits for a move
extern "C" uint64_t get_flip_cpp(uint64_t P, uint64_t O, int32_t idx) {
    int32_t r = idx / 8;
    int32_t c = idx % 8;
    uint64_t flips = 0;
    
    const int32_t dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
    const int32_t dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    
    for (int32_t d = 0; d < 8; d++) {
        int32_t rr = r + dr[d];
        int32_t cc = c + dc[d];
        uint64_t captured = 0;
        bool seen_opponent = false;
        
        while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
            int32_t nidx = rr * 8 + cc;
            uint64_t nbit = 1ULL << nidx;
            
            if (O & nbit) {
                seen_opponent = true;
                captured |= nbit;
                rr += dr[d];
                cc += dc[d];
                continue;
            }
            if (seen_opponent && (P & nbit)) {
                flips |= captured;
            }
            break;
        }
    }
    return flips & FULL_MASK;
}

// Compute stable pieces
extern "C" uint64_t compute_strict_stable_cpp(uint64_t P, uint64_t O) {
    uint64_t s = P & MASK_CORNER;
    if (!s) return 0;
    
    // First pass: expand from corners
    for (int32_t iter = 0; iter < 7; iter++) {
        uint64_t ns = 0;
        ns |= (s << 1) & NOT_A_FILE;
        ns |= (s >> 1) & NOT_H_FILE;
        ns |= (s << 8) & FULL_MASK;
        ns |= (s >> 8) & FULL_MASK;
        ns |= (s << 7) & NOT_A_FILE;
        ns |= (s >> 7) & NOT_H_FILE;
        ns |= (s << 9) & NOT_H_FILE;
        ns |= (s >> 9) & NOT_A_FILE;
        
        uint64_t a = (ns & P) & ~s;
        if (!a) break;
        s |= a;
    }
    
    // Second pass: check stability conditions
    bool changed = true;
    int32_t it = 0;
    uint64_t occ = P | O;
    
    while (changed && it < 8) {
        changed = false;
        uint64_t b = 1;
        
        for (int32_t pos = 0; pos < 64; pos++) {
            if ((P & b) && !(s & b)) {
                int32_t locked_dirs = 0;
                
                const int32_t dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
                const int32_t dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
                
                for (int32_t d = 0; d < 8; d++) {
                    uint64_t c = b;
                    bool dead = false;
                    bool all_covered = true;
                    
                    for (int32_t step = 0; step < 8 && all_covered; step++) {
                        // Move in direction d
                        if (d == 0) c = (c << 1) & NOT_A_FILE;
                        else if (d == 1) c = (c >> 1) & NOT_H_FILE;
                        else if (d == 2) c = (c << 8) & FULL_MASK;
                        else if (d == 3) c = (c >> 8) & FULL_MASK;
                        else if (d == 4) c = (c << 7) & NOT_A_FILE;
                        else if (d == 5) c = (c >> 7) & NOT_H_FILE;
                        else if (d == 6) c = (c << 9) & NOT_H_FILE;
                        else if (d == 7) c = (c >> 9) & NOT_A_FILE;
                        
                        if (!c || !(c & occ)) {
                            dead = true;
                            break;
                        }
                        if (c & P) {
                            if (c & s) {
                                dead = true;
                                break;
                            } else {
                                all_covered = false;
                                break;
                            }
                        }
                        if (c & O) {
                            all_covered = false;
                            break;
                        }
                    }
                    if (dead) locked_dirs++;
                }
                
                if (locked_dirs >= 4) {
                    s |= b;
                    changed = true;
                }
            }
            b <<= 1;
        }
        it++;
    }
    
    return s & FULL_MASK;
}

// Evaluate corner-X-C regions - inlined for speed
__forceinline double eval_xc(uint64_t p, uint64_t o, uint64_t cor, uint64_t cx, uint64_t cc) {
    double v = 0.0;
    if (!(p & cor) && !(o & cor)) {
        if (p & cx) v -= 50.0;
        if (o & cx) v += 50.0;
        v -= static_cast<double>(count_bits(p & cc)) * 20.0;
        v += static_cast<double>(count_bits(o & cc)) * 20.0;
    } else if (p & cor) {
        if (p & cx) v += 15.0;
        v += static_cast<double>(count_bits(p & cc)) * 10.0;
    } else {
        if (o & cx) v -= 15.0;
        v -= static_cast<double>(count_bits(o & cc)) * 10.0;
    }
    return v;
}

// Fast piece weight accumulation using hardware popcnt
__forceinline double accumulate_piece_weights(uint64_t bits, const double* weights) {
    double sum = 0.0;
    while (bits) {
        uint64_t t = lsb(bits);
        int idx = bit_index(t);
        sum += weights[idx];
        bits &= bits - 1;
    }
    return sum;
}

// Main evaluation function - HEAVILY OPTIMIZED
extern "C" double evaluate_board_full_cpp(
    uint64_t P,
    uint64_t O,
    int32_t mvs,
    const double* W  // Weight array of size 243
) {
    // Determine stage once with ternary
    const int32_t st = (mvs <= 15) ? 0 : ((mvs <= 45) ? 80 : 160);
    const double* wp = &W[st];
    const double* we = &W[st + 64];
    
    // Material score - single pass with optimized bit iteration
    double sc = accumulate_piece_weights(P, wp) - accumulate_piece_weights(O, wp);
    
    // Precompute frequently used values
    uint64_t emp = ~(P | O) & FULL_MASK;
    uint64_t occ = P | O;
    uint64_t np_ = neighbor_union(P);
    uint64_t no = neighbor_union(O);
    
    // Mobility - computed once each
    uint64_t player_moves = get_legal_moves_cpp(P, O);
    uint64_t opp_moves = get_legal_moves_cpp(O, P);
    int32_t lm = count_bits(player_moves);
    int32_t lo = count_bits(opp_moves);
    
    // Mobility multiplier and score
    double m_mult = (mvs >= 20 && mvs <= 45) ? 2.5 : 1.0;
    sc += static_cast<double>(lm - lo) * m_mult * 4.0;
    
    // Parity score using bitwise AND
    sc += (((64 - mvs) & 1) == 0) ? 10.0 : -10.0;
    
    // Corner-X-C evaluation (4 corners) - inlined
    sc += eval_xc(P, O, 0x1ULL, 0x200ULL, 0x102ULL);
    sc += eval_xc(P, O, 0x80ULL, 0x4000ULL, 0x8040ULL);
    sc += eval_xc(P, O, 0x100000000000000ULL, 0x20000000000000ULL, 0x201000000000000ULL);
    sc += eval_xc(P, O, 0x8000000000000000ULL, 0x400000000000000ULL, 0x4080000000000000ULL);
    
    // Stable pieces (only compute in endgame)
    int32_t sp_count = 0, so_count = 0;
    if (mvs >= 30) {
        uint64_t sp = compute_strict_stable_cpp(P, O);
        uint64_t so = compute_strict_stable_cpp(O, P);
        sp_count = count_bits(sp & P);
        so_count = count_bits(so & O);
        sc += static_cast<double>(sp_count - so_count) * 25.0;
    }
    
    // Feature-based evaluation - batch compute all counts with hardware POPCNT
    int32_t emp_np = count_bits(emp & np_);
    int32_t emp_no = count_bits(emp & no);
    int32_t p_np = count_bits(P & np_);
    int32_t o_no = count_bits(O & no);
    int32_t p_corner = count_bits(P & MASK_CORNER);
    int32_t o_corner = count_bits(O & MASK_CORNER);
    int32_t occ_bits = count_bits(occ);
    int32_t p_x = count_bits(P & MASK_X);
    int32_t o_x = count_bits(O & MASK_X);
    int32_t p_c = count_bits(P & MASK_C);
    int32_t o_c = count_bits(O & MASK_C);
    int32_t p_b1 = count_bits(P & MASK_B1);
    int32_t o_b1 = count_bits(O & MASK_B1);
    int32_t p_b2 = count_bits(P & MASK_B2);
    int32_t o_b2 = count_bits(O & MASK_B2);
    int32_t p_b3 = count_bits(P & MASK_B3);
    int32_t o_b3 = count_bits(O & MASK_B3);
    int32_t p_b4 = count_bits(P & MASK_B4);
    int32_t o_b4 = count_bits(O & MASK_B4);
    int32_t occ_b1 = count_bits(occ & MASK_B1);
    int32_t occ_b2 = count_bits(occ & MASK_B2);
    int32_t occ_b3 = count_bits(occ & MASK_B3);
    int32_t occ_b4 = count_bits(occ & MASK_B4);
    
    // Apply weights - use static_cast for type safety
    sc += (static_cast<double>(lm - lo) / 20.0) * we[0];
    sc += (static_cast<double>(emp_np - emp_no) / 64.0) * we[1];
    sc += static_cast<double>(sp_count - so_count) * we[2];
    sc += (static_cast<double>(p_np - o_no) / 64.0) * we[3];
    sc += static_cast<double>(p_corner - o_corner) * we[4];
    sc += static_cast<double>(occ_bits & 1) * we[5];
    sc += static_cast<double>(p_x - o_x) * we[6];
    sc += static_cast<double>(p_c - o_c) * we[7];
    sc += (static_cast<double>(p_b1 - o_b1) / 16.0) * we[8];
    sc += (static_cast<double>(p_b2 - o_b2) / 16.0) * we[9];
    sc += (static_cast<double>(p_b3 - o_b3) / 16.0) * we[10];
    sc += (static_cast<double>(p_b4 - o_b4) / 16.0) * we[11];
    sc += static_cast<double>(occ_b1 & 1) * we[12];
    sc += static_cast<double>(occ_b2 & 1) * we[13];
    sc += static_cast<double>(occ_b3 & 1) * we[14];
    sc += static_cast<double>(occ_b4 & 1) * we[15];

    return sc;
}

// Batch evaluation with OpenMP parallelization
extern "C" void evaluate_board_batch_cpp(
    const uint64_t* P_arr,
    const uint64_t* O_arr,
    const int32_t* mvs_arr,
    const double* W,
    double* results,
    int32_t count
) {
    #pragma omp parallel for schedule(dynamic, 4)
    for (int32_t i = 0; i < count; i++) {
        results[i] = evaluate_board_full_cpp(P_arr[i], O_arr[i], mvs_arr[i], W);
    }
}

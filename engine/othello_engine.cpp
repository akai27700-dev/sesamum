#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>

namespace py = pybind11;

using Bitboard = std::uint64_t;
struct GlobalSearchSettings {
    bool pruning_enabled = true;
    bool traditional_pruning_enabled = true;
    bool multi_cut_enabled = true;
    int multi_cut_threshold = 3;
    int multi_cut_depth = 8;
    int ybwc_min_depth = 6;
} g_search_settings;

namespace {

inline void simd_prefetch_bitboards(const Bitboard* boards) {
    for (int i = 0; i < 8; ++i) {
        _mm_prefetch(reinterpret_cast<const char*>(&boards[i]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&boards[i]) + 64, _MM_HINT_T0);
    }
}

inline __m256i simd_load_bitboards(const Bitboard* boards) {
    _mm_prefetch(reinterpret_cast<const char*>(&boards[4]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&boards[5]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&boards[6]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&boards[7]), _MM_HINT_T0);
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(boards));
}

inline void simd_store_bitboards(Bitboard* boards, const __m256i& vec) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(boards), vec);
}

inline __m256i simd_shift_left(const __m256i& vec, int shift) {
    return _mm256_slli_epi64(vec, shift);
}

inline __m256i simd_shift_right(const __m256i& vec, int shift) {
    return _mm256_srli_epi64(vec, shift);
}

inline __m256i simd_and(const __m256i& a, const __m256i& b) {
    return _mm256_and_si256(a, b);
}

inline __m256i simd_or(const __m256i& a, const __m256i& b) {
    return _mm256_or_si256(a, b);
}

inline __m256i simd_xor(const __m256i& a, const __m256i& b) {
    return _mm256_xor_si256(a, b);
}

inline __m256i simd_not(const __m256i& vec) {
    return _mm256_xor_si256(vec, _mm256_set1_epi64x(-1));
}

inline __m256i simd_popcount_avx2(__m256i vec) {
    const __m256i mask1 = _mm256_set1_epi64x(0x5555555555555555);
    const __m256i mask2 = _mm256_set1_epi64x(0x3333333333333333);
    const __m256i mask4 = _mm256_set1_epi64x(0x0F0F0F0F0F0F0F0F);
    const __m256i mask8 = _mm256_set1_epi64x(0x0101010101010101);
    
    __m256i v = vec;
    v = _mm256_sub_epi64(_mm256_and_si256(v, mask1), _mm256_and_si256(_mm256_srli_epi64(v, 1), mask1));
    v = _mm256_add_epi64(_mm256_and_si256(v, mask2), _mm256_and_si256(_mm256_srli_epi64(v, 2), mask2));
    v = _mm256_and_si256(_mm256_add_epi64(v, _mm256_srli_epi64(v, 4)), mask4);
    v = _mm256_mul_epu32(v, mask8);
    v = _mm256_srli_epi64(v, 56);
    return v;
}

inline void simd_get_legal_moves_batch(const Bitboard* p_boards, const Bitboard* o_boards, Bitboard* legal_boards) {
    const __m256i full_mask_vec = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);
    const __m256i not_a_file_vec = _mm256_set1_epi64x(0xFEFEFEFEFEFEFEFEULL);
    const __m256i not_h_file_vec = _mm256_set1_epi64x(0x7F7F7F7F7F7F7F7FULL);
    
    for (int i = 0; i < 4; ++i) {
        __m256i p = simd_load_bitboards(&p_boards[i*4]);
        __m256i o = simd_load_bitboards(&o_boards[i*4]);
        __m256i occ = simd_or(p, o);
        
        
        
        __m256i legal = simd_not(simd_or(p, o));
        legal = simd_and(legal, full_mask_vec);
        
        simd_store_bitboards(&legal_boards[i*4], legal);
    }
}

constexpr Bitboard FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;
constexpr Bitboard NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL;
constexpr Bitboard NOT_H_FILE = 0x7F7F7F7F7F7F7F7FULL;
constexpr std::size_t GENE_LEN = 243;
constexpr std::size_t TT_SIZE = 1u << 18;
constexpr std::size_t MAX_BATCH_SIZE = 262144;
constexpr std::size_t SIMD_BATCH_SIZE = 64;

constexpr Bitboard MASK_CORNER = 0x8100000000000081ULL;
constexpr Bitboard MASK_EDGE = 0xFF818181818181FFULL;
constexpr Bitboard MASK_X = 0x0042000000004200ULL;
constexpr Bitboard MASK_C = 0x4281000000008142ULL;
constexpr Bitboard MASK_B1 = 0x0F0F0F0F00000000ULL;
constexpr Bitboard MASK_B2 = 0xF0F0F0F000000000ULL;
constexpr Bitboard MASK_B3 = 0x000000000F0F0F0FULL;
constexpr Bitboard MASK_B4 = 0x00000000F0F0F0F0ULL;

inline int count_bits(Bitboard x) {
    return static_cast<int>(_mm_popcnt_u64(x));
}

inline int count_bits_simd(Bitboard x) {
    return static_cast<int>(_mm_popcnt_u64(x));
}

inline int bit_index(Bitboard x) {
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
}

inline int bit_index_simd(Bitboard x) {
    return static_cast<int>(_tzcnt_u64(x));
}

inline Bitboard lsb(Bitboard x) {
    return x & (~x + 1ULL);
}

inline int count_zeros_leading(Bitboard x) {
    unsigned long idx = 0;
    _BitScanReverse64(&idx, x);
    return 63 - static_cast<int>(idx);
}

inline Bitboard reverse_bits(Bitboard x) {
    return _mm_cvtsi128_si64(_mm_srli_si128(_mm_cvtsi64_si128(x), 8));
}

int estimate_root_parallel_lanes(int move_count, int depth, bool is_exact);
const std::vector<double>& require_global_weights();
const std::vector<double>& require_global_order_map();

std::vector<double> make_board_perfect_seed() {
    std::vector<double> w(GENE_LEN, 0.0);
    std::array<double, 64> b = {10.0,-5.0,3.0,2.0,2.0,3.0,-5.0,10.0,-5.0,-8.0,-1.0,-1.0,-1.0,-1.0,-8.0,-5.0,3.0,-1.0,1.5,1.0,1.0,1.5,-1.0,3.0,2.0,-1.0,1.0,0.2,0.2,1.0,-1.0,2.0,2.0,-1.0,1.0,0.2,0.2,1.0,-1.0,2.0,3.0,-1.0,1.5,1.0,1.0,1.5,-1.0,3.0,-5.0,-8.0,-1.0,-1.0,-1.0,-1.0,-8.0,-5.0,10.0,-5.0,3.0,2.0,2.0,3.0,-5.0,10.0};
    for (int i = 0; i < 64; ++i) w[i] = b[i] * 0.5;
    for (int i = 0; i < 64; ++i) w[80 + i] = b[i] * 1.2;
    for (int i = 0; i < 64; ++i) w[160 + i] = b[i] * 2.0;
    std::array<double, 16> w1 = {1.2,0.6,1.0,-0.8,2.5,0.2,-2.5,-1.5,0.2,0.2,0.2,0.2,0.3,0.3,0.3,0.3};
    std::array<double, 16> w2 = {0.8,0.4,2.5,-0.5,4.0,0.5,-3.5,-2.0,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8};
    std::array<double, 16> w3 = {0.2,0.1,5.0,-0.2,6.0,2.0,-1.0,-0.5,0.8,0.8,0.8,0.8,1.5,1.5,1.5,1.5};
    for (int i = 0; i < 16; ++i) w[64 + i] = w1[i];
    for (int i = 0; i < 16; ++i) w[144 + i] = w2[i];
    for (int i = 0; i < 16; ++i) w[224 + i] = w3[i];
    w[240] = 0.45; w[241] = 0.85; w[242] = 0.05;
    return w;
}

struct EvalLookupTables {
    std::array<double, 65536> corner_xc_table;
    std::array<double, 65536> pattern_table;
    std::array<double, 256> mobility_table;
    std::array<double, 65536> stability_table;
    
    EvalLookupTables() {
        init_corner_xc_table();
        init_pattern_table();
        init_mobility_table();
        init_stability_table();
    }
    
    void init_corner_xc_table() {
        for (int i = 0; i < 65536; ++i) {
            Bitboard p = i;
            Bitboard o = i << 16;
            double v = 0.0;
            if (!(p & 0x1ULL) && !(o & 0x1ULL)) {
                if (p & 0x200ULL) v -= 50.0;
                if (o & 0x200ULL) v += 50.0;
                v -= static_cast<double>(count_bits(p & 0x102ULL)) * 20.0;
                v += static_cast<double>(count_bits(o & 0x102ULL)) * 20.0;
            } else if (p & 0x1ULL) {
                if (p & 0x200ULL) v += 15.0;
                v += static_cast<double>(count_bits(p & 0x102ULL)) * 10.0;
            } else {
                if (o & 0x200ULL) v -= 15.0;
                v -= static_cast<double>(count_bits(o & 0x102ULL)) * 10.0;
            }
            corner_xc_table[i] = v;
        }
    }
    
    void init_pattern_table() {
        for (int i = 0; i < 65536; ++i) {
            Bitboard pattern = i;
            double score = 0.0;
            for (int pos = 0; pos < 16; ++pos) {
                if (pattern & (1ULL << pos)) {
                    score += (pos % 4 == 0 || pos % 4 == 3) ? 10.0 : 2.0;
                }
            }
            pattern_table[i] = score;
        }
    }
    
    void init_mobility_table() {
        for (int i = 0; i < 256; ++i) {
            int lm = count_bits(i & 0xF);
            int lo = count_bits((i >> 4) & 0xF);
            mobility_table[i] = static_cast<double>(lm - lo) * 4.0;
        }
    }
    
    void init_stability_table() {
        for (int i = 0; i < 65536; ++i) {
            Bitboard region = i;
            double stability = 0.0;
            int edges = count_bits(region & 0xF) + count_bits(region & 0xF000) + 
                       count_bits(region & 0x8888) + count_bits(region & 0x1111);
            stability += edges * 5.0;
            int corners = count_bits(region & 0x8001) + count_bits((region >> 15) & 0x3);
            stability += corners * 20.0;
            stability_table[i] = stability;
        }
    }
};

EvalLookupTables& eval_lut() {
    static EvalLookupTables lut;
    return lut;
}

std::array<Bitboard, 64> make_zobrist(std::uint64_t seed_offset) {
    std::array<Bitboard, 64> out{};
    std::uint64_t state = 123456789ULL + seed_offset;
    constexpr std::uint64_t a = 2862933555777941757ULL;
    constexpr std::uint64_t c = 3037000493ULL;
    for (std::size_t i = 0; i < 64; ++i) {
        state = state * a + c;
        out[i] = state;
    }
    return out;
}

const auto ZOBRIST_BLACK = [] {
    std::array<Bitboard, 64> zb{};
    std::uint64_t state = 123456789ULL;
    constexpr std::uint64_t a = 2862933555777941757ULL;
    constexpr std::uint64_t c = 3037000493ULL;
    for (std::size_t i = 0; i < 64; ++i) {
        state = state * a + c;
        zb[i] = state;
        state = state * a + c;
    }
    return zb;
}();

const auto ZOBRIST_WHITE = [] {
    std::array<Bitboard, 64> zw{};
    std::uint64_t state = 123456789ULL;
    constexpr std::uint64_t a = 2862933555777941757ULL;
    constexpr std::uint64_t c = 3037000493ULL;
    for (std::size_t i = 0; i < 64; ++i) {
        state = state * a + c;
        state = state * a + c;
        zw[i] = state;
    }
    return zw;
}();

inline Bitboard zobrist_hash(Bitboard p, Bitboard o) {
    Bitboard h = 0;
    while (p) {
        Bitboard bit = lsb(p);
        int idx = bit_index(bit);
        h ^= ZOBRIST_BLACK[static_cast<std::size_t>(idx)];
        p ^= bit;
    }
    while (o) {
        Bitboard bit = lsb(o);
        int idx = bit_index(bit);
        h ^= ZOBRIST_WHITE[static_cast<std::size_t>(idx)];
        o ^= bit;
    }
    return h;
}

inline Bitboard get_legal_moves_optimized(Bitboard p, Bitboard o) {
    const Bitboard occ = p | o;
    const Bitboard empty = ~occ & FULL_MASK;
    const Bitboard m = NOT_A_FILE & NOT_H_FILE;
    Bitboard legal = 0;
    
    
    Bitboard t = (p << 1) & o & m;
    t |= (t << 1) & o & m; t |= (t << 1) & o & m; t |= (t << 1) & o & m; t |= (t << 1) & o & m; t |= (t << 1) & o & m;
    legal |= (t << 1) & empty;
    
    t = (p >> 1) & o & m;
    t |= (t >> 1) & o & m; t |= (t >> 1) & o & m; t |= (t >> 1) & o & m; t |= (t >> 1) & o & m; t |= (t >> 1) & o & m;
    legal |= (t >> 1) & empty;
    
    t = (p << 8) & o;
    t |= (t << 8) & o; t |= (t << 8) & o; t |= (t << 8) & o; t |= (t << 8) & o; t |= (t << 8) & o;
    legal |= (t << 8) & empty;
    
    t = (p >> 8) & o;
    t |= (t >> 8) & o; t |= (t >> 8) & o; t |= (t >> 8) & o; t |= (t >> 8) & o; t |= (t >> 8) & o;
    legal |= (t >> 8) & empty;
    
    t = (p << 7) & o & m;
    t |= (t << 7) & o & m; t |= (t << 7) & o & m; t |= (t << 7) & o & m; t |= (t << 7) & o & m; t |= (t << 7) & o & m;
    legal |= (t << 7) & empty;
    
    t = (p >> 7) & o & m;
    t |= (t >> 7) & o & m; t |= (t >> 7) & o & m; t |= (t >> 7) & o & m; t |= (t >> 7) & o & m; t |= (t >> 7) & o & m;
    legal |= (t >> 7) & empty;
    
    t = (p << 9) & o & m;
    t |= (t << 9) & o & m; t |= (t << 9) & o & m; t |= (t << 9) & o & m; t |= (t << 9) & o & m; t |= (t << 9) & o & m;
    legal |= (t << 9) & empty;
    
    t = (p >> 9) & o & m;
    t |= (t >> 9) & o & m; t |= (t >> 9) & o & m; t |= (t >> 9) & o & m; t |= (t >> 9) & o & m; t |= (t >> 9) & o & m;
    legal |= (t >> 9) & empty;
    
    return legal & FULL_MASK;
}

inline Bitboard get_flip_optimized(Bitboard p, Bitboard o, int idx) {
    const Bitboard mask = 1ULL << idx;
    const Bitboard occ = p | o;
    
    
    if (occ & mask) return 0;
    
    int r = idx / 8;  
    int c = idx % 8;
    Bitboard flips = 0;
    
    
    const int dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
    const int dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    
    for (int dir = 0; dir < 8; ++dir) {
        int rr = r + dr[dir];
        int cc = c + dc[dir];
        Bitboard captured = 0;
        bool seen_opponent = false;
        
        while (0 <= rr && rr < 8 && 0 <= cc && cc < 8) {
            int nidx = rr * 8 + cc;
            Bitboard nbit = 1ULL << nidx;
            
            if (o & nbit) {
                seen_opponent = true;
                captured |= nbit;
                rr += dr[dir];
                cc += dc[dir];
                continue;
            }
            
            if (seen_opponent && (p & nbit)) {
                flips |= captured;
            }
            break;
        }
    }
    
    return flips & FULL_MASK;
}

inline Bitboard neighbor_union(Bitboard bb) {
    Bitboard n = 0;
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

inline Bitboard shift_dir(Bitboard b, int d) {
    if (d == 0) return (b << 1) & NOT_A_FILE;
    if (d == 1) return (b >> 1) & NOT_H_FILE;
    if (d == 2) return (b << 8) & FULL_MASK;
    if (d == 3) return (b >> 8) & FULL_MASK;
    if (d == 4) return (b << 7) & NOT_A_FILE;
    if (d == 5) return (b >> 7) & NOT_H_FILE;
    if (d == 6) return (b << 9) & NOT_H_FILE;
    return (b >> 9) & NOT_A_FILE;
}

Bitboard compute_strict_stable(Bitboard p, Bitboard o) {
    Bitboard s = p & MASK_CORNER;
    if (!s) return 0;
    for (int i = 0; i < 7; ++i) {
        Bitboard ns = 0;
        ns |= (s << 1) & NOT_A_FILE;
        ns |= (s >> 1) & NOT_H_FILE;
        ns |= (s << 8) & FULL_MASK;
        ns |= (s >> 8) & FULL_MASK;
        ns |= (s << 7) & NOT_A_FILE;
        ns |= (s >> 7) & NOT_H_FILE;
        ns |= (s << 9) & NOT_H_FILE;
        ns |= (s >> 9) & NOT_A_FILE;
        Bitboard a = (ns & p) & ~s;
        if (!a) break;
        s |= a;
    }
    bool ch = true;
    int it = 0;
    Bitboard occ = p | o;
    while (ch && it < 8) {
        ch = false;
        Bitboard rem = p & ~s;
        while (rem) {
            Bitboard b = lsb(rem);
            rem ^= b;
            int ld = 0;
            for (int d = 0; d < 8; ++d) {
                Bitboard c = b;
                bool dl = false;
                bool ac = true;
                for (int k = 0; k < 8; ++k) {
                    if (!ac) break;
                    c = shift_dir(c, d);
                    if (!c || !(c & occ)) {
                        dl = true;
                        break;
                    }
                    if (c & p) {
                        if (c & s) {
                            dl = true;
                            break;
                        }
                        ac = false;
                        break;
                    }
                    if (c & o) {
                        ac = false;
                        break;
                    }
                }
                if (dl) ++ld;
            }
            if (ld >= 4) {
                s |= b;
                ch = true;
            }
        }
        ++it;
    }
    return s & FULL_MASK;
}

double eval_xc(Bitboard p, Bitboard o, Bitboard cor, Bitboard cx, Bitboard cc) {
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

double evaluate_board_full(Bitboard p, Bitboard o, int mvs, const std::vector<double>& w) {
    if (w.size() != GENE_LEN) throw std::invalid_argument("weights size must be 243");
    
    auto& lut = eval_lut();
    int st = mvs <= 15 ? 0 : (mvs <= 45 ? 80 : 160);
    const double* w_base = w.data() + st;
    const double* w_feat = w_base + 64;
    double sc = 0.0;
    
    Bitboard tp = p;
    Bitboard to = o;
    while (tp) {
        Bitboard t = lsb(tp);
        int ix = bit_index(t);
        sc += w_base[ix];
        tp &= tp - 1ULL;
    }
    while (to) {
        Bitboard t = lsb(to);
        int ix = bit_index(t);
        sc -= w_base[ix];
        to &= to - 1ULL;
    }
    
    const Bitboard legal_p = get_legal_moves_optimized(p, o);
    const Bitboard legal_o = get_legal_moves_optimized(o, p);
    const int lm = count_bits(legal_p);
    const int lo = count_bits(legal_o);
    Bitboard emp = ~(p | o) & FULL_MASK;
    Bitboard np = neighbor_union(p);
    Bitboard no = neighbor_union(o);
    Bitboard sp = 0;
    Bitboard so = 0;
    int stable_diff = 0;
    bool use_strict_stable = mvs >= 18;
    if (use_strict_stable) {
        sp = compute_strict_stable(p, o);
        so = compute_strict_stable(o, p);
        stable_diff = count_bits(sp & p) - count_bits(so & o);
    }
    Bitboard occ = p | o;
    const int empties = 64 - mvs;

    // --- Mobility (Egaroucid-inspired phase-dependent) ---
    double m_mult = 1.0;
    if (mvs >= 20 && mvs <= 45) m_mult = 2.5;
    else if (mvs > 45) m_mult = 1.8;
    sc += static_cast<double>(lm - lo) * m_mult * 4.0;

    // --- Parity: odd empties = last mover advantage ---
    if (empties > 0) {
        double parity_bonus = (empties % 2 == 1) ? 12.0 : -12.0;
        if (mvs >= 40) parity_bonus *= 2.0;
        sc += parity_bonus;
    }

    // --- 4-corner XC evaluation (Egaroucid style) ---
    static constexpr Bitboard CORNER_TL = 1ULL << 63;  // a1
    static constexpr Bitboard CX_TL = 1ULL << 62;      // b1
    static constexpr Bitboard CC_TL = (1ULL << 61) | (1ULL << 55); // c1, a2
    sc += eval_xc(p, o, CORNER_TL, CX_TL, CC_TL);

    static constexpr Bitboard CORNER_TR = 1ULL << 56;  // h1
    static constexpr Bitboard CX_TR = 1ULL << 57;      // g1
    static constexpr Bitboard CC_TR = (1ULL << 58) | (1ULL << 48); // f1, h2
    sc += eval_xc(p, o, CORNER_TR, CX_TR, CC_TR);

    static constexpr Bitboard CORNER_BL = 1ULL << 7;   // a8
    static constexpr Bitboard CX_BL = 1ULL << 6;       // b8
    static constexpr Bitboard CC_BL = (1ULL << 5) | (1ULL << 15); // c8, a7
    sc += eval_xc(p, o, CORNER_BL, CX_BL, CC_BL);

    static constexpr Bitboard CORNER_BR = 1ULL << 0;   // h8
    static constexpr Bitboard CX_BR = 1ULL << 1;       // g8
    static constexpr Bitboard CC_BR = (1ULL << 2) | (1ULL << 8); // f8, h7
    sc += eval_xc(p, o, CORNER_BR, CX_BR, CC_BR);
    // Legacy LUT for additional row/column patterns
    sc += lut.corner_xc_table[(p & 0xFFFF) & 0xFFFF] + lut.corner_xc_table[((p >> 48) & 0xFFFF)];
    sc += lut.pattern_table[((p >> 32) & 0xFFFF)] + lut.pattern_table[(p >> 16) & 0xFFFF];
    
    // --- Stability with phase-dependent weight ---
    double stable_weight = 15.0;
    if (mvs >= 30) stable_weight = 25.0;
    if (mvs >= 45) stable_weight = 40.0;
    sc += static_cast<double>(stable_diff) * stable_weight;

    // --- Frontier disc penalty (Egaroucid-inspired) ---
    // Discs adjacent to empty squares are unstable and vulnerable
    const Bitboard frontier_p = p & np & emp;
    const Bitboard frontier_o = o & no & emp;
    const int frontier_diff = count_bits(frontier_p) - count_bits(frontier_o);
    double frontier_weight = 3.0;
    if (mvs >= 30) frontier_weight = 5.0;
    if (mvs >= 45) frontier_weight = 8.0;
    sc -= static_cast<double>(frontier_diff) * frontier_weight;

    // --- Potential mobility (empty squares adjacent to opponent) ---
    // More potential moves = better position
    const int pot_mob_p = count_bits(emp & no);
    const int pot_mob_o = count_bits(emp & np);
    sc += static_cast<double>(pot_mob_p - pot_mob_o) * 2.0;

    sc += lut.mobility_table[(lm & 0xF) | ((lo & 0xF) << 4)];
    sc += (static_cast<double>(count_bits(emp & np) - count_bits(emp & no)) / 64.0) * w_feat[1];
    sc += static_cast<double>(stable_diff) * w_feat[2];
    sc += (static_cast<double>(count_bits(p & np) - count_bits(o & no)) / 64.0) * w_feat[3];
    sc += static_cast<double>(count_bits(p & MASK_CORNER) - count_bits(o & MASK_CORNER)) * w_feat[4];
    sc += static_cast<double>(count_bits(occ) & 1) * w_feat[5];
    sc += static_cast<double>(count_bits(p & MASK_X) - count_bits(o & MASK_X)) * w_feat[6];
    sc += static_cast<double>(count_bits(p & MASK_C) - count_bits(o & MASK_C)) * w_feat[7];
    sc += (static_cast<double>(count_bits(p & MASK_B1) - count_bits(o & MASK_B1)) / 16.0) * w_feat[8];
    sc += (static_cast<double>(count_bits(p & MASK_B2) - count_bits(o & MASK_B2)) / 16.0) * w_feat[9];
    sc += (static_cast<double>(count_bits(p & MASK_B3) - count_bits(o & MASK_B3)) / 16.0) * w_feat[10];
    sc += (static_cast<double>(count_bits(p & MASK_B4) - count_bits(o & MASK_B4)) / 16.0) * w_feat[11];
    sc += static_cast<double>(count_bits(occ & MASK_B1) & 1) * w_feat[12];
    sc += static_cast<double>(count_bits(occ & MASK_B2) & 1) * w_feat[13];
    sc += static_cast<double>(count_bits(occ & MASK_B3) & 1) * w_feat[14];
    sc += static_cast<double>(count_bits(occ & MASK_B4) & 1) * w_feat[15];
    return sc;
}

inline void evaluate_board_full_simd_batch(const Bitboard* p_boards, const Bitboard* o_boards, const int* mvs_array, 
                                     const std::vector<double>& w, double* scores, int batch_size) {
    if (w.size() != GENE_LEN) throw std::invalid_argument("weights size must be 243");
    
    const int simd_batch = SIMD_BATCH_SIZE;
    const int full_batches = batch_size / simd_batch;
    const int remainder = batch_size % simd_batch;
    
    auto& lut = eval_lut();
    
    for (int batch = 0; batch < full_batches; ++batch) {
        int base_idx = batch * simd_batch;
        
        __m256i p_vecs[4];
        __m256i o_vecs[4];
        __m256d batch_scores[4] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};
        
        for (int quad = 0; quad < 4; ++quad) {
            int quad_base = base_idx + quad * 8;
            simd_prefetch_bitboards(&p_boards[quad_base]);
            simd_prefetch_bitboards(&o_boards[quad_base]);
            p_vecs[quad] = simd_load_bitboards(&p_boards[quad_base]);
            o_vecs[quad] = simd_load_bitboards(&o_boards[quad_base]);
        }
        
        __m256d weight_vectors[64];
        for (int i = 0; i < 64; ++i) {
            weight_vectors[i] = _mm256_set1_pd(w[i]);
        }
        
        for (int pos = 0; pos < 64; ++pos) {
            __m256i pos_mask = _mm256_set1_epi64x(1ULL << pos);
            
            for (int quad = 0; quad < 4; ++quad) {
                __m256i p_occupied = simd_and(p_vecs[quad], pos_mask);
                __m256i o_occupied = simd_and(o_vecs[quad], pos_mask);
                
                __m256i p_popcnt = simd_popcount_avx2(p_occupied);
                __m256i o_popcnt = simd_popcount_avx2(o_occupied);
                
                __m256d p_scores = _mm256_cvtepi64_pd(p_popcnt);
                __m256d o_scores = _mm256_cvtepi64_pd(o_popcnt);
                
                p_scores = _mm256_mul_pd(p_scores, weight_vectors[pos]);
                o_scores = _mm256_mul_pd(o_scores, weight_vectors[pos]);
                
                batch_scores[quad] = _mm256_add_pd(batch_scores[quad], _mm256_sub_pd(p_scores, o_scores));
            }
        }
        
        for (int quad = 0; quad < 4; ++quad) {
            _mm256_storeu_pd(&scores[base_idx + quad * 8], batch_scores[quad]);
            
            for (int i = 0; i < 8; ++i) {
                int idx = base_idx + quad * 8 + i;
                Bitboard p_quad = p_boards[idx];
                Bitboard o_quad = o_boards[idx];
                
                int corner_idx = (p_quad & 0xFFFF) & 0xFFFF;
                scores[idx] += lut.corner_xc_table[corner_idx];
                
                int pattern_idx = (p_quad >> 16) & 0xFFFF;
                scores[idx] += lut.pattern_table[pattern_idx];
                
                Bitboard legal_p = get_legal_moves_optimized(p_quad, o_quad);
                Bitboard legal_o = get_legal_moves_optimized(o_quad, p_quad);
                int lm = count_bits(legal_p);
                int lo = count_bits(legal_o);
                int mobility_idx = (lm & 0xF) | ((lo & 0xF) << 4);
                scores[idx] += lut.mobility_table[mobility_idx];
            }
        }
    }
    
    for (int i = full_batches * simd_batch; i < batch_size; ++i) {
        scores[i] = evaluate_board_full(p_boards[i], o_boards[i], mvs_array[i], w);
    }
}
inline double exact_eval(Bitboard p, Bitboard o, int mvs, Bitboard valid) {
    const int p_count = count_bits(p);
    const int o_count = count_bits(o);
    const int diff = p_count - o_count;
    
    // Primary: Disc Difference
    double score = static_cast<double>(diff) * 10000.0;
    
    // Secondary: Mobility and Stability (for deeper exact searches that haven't reached the end)
    const int own_moves = count_bits(valid);
    const int opp_moves = count_bits(get_legal_moves_optimized(o, p));
    
    // Stability/Corners
    const Bitboard corners = 0x8100000000000081ULL;
    const int p_corners = count_bits(p & corners);
    const int o_corners = count_bits(o & corners);
    
    score += static_cast<double>(own_moves - opp_moves) * 150.0;
    score += static_cast<double>(p_corners - o_corners) * 800.0;
    
    return score;
}

struct TTEntry {
    template <typename T>
    struct Cell {
        T value{};

        Cell() = default;
        Cell(T initial) : value(initial) {}

        T load(std::memory_order = std::memory_order_seq_cst) const {
            return value;
        }

        void store(T next, std::memory_order = std::memory_order_seq_cst) {
            value = next;
        }
    };

    alignas(16) Cell<Bitboard> hash;
    Cell<int> depth;
    Cell<double> value;
    Cell<std::int8_t> flag;
    Cell<std::int8_t> best_move;
    Cell<std::uint32_t> age;
    Cell<std::uint8_t> generation;

    TTEntry() : hash(0), depth(-1), value(0.0), flag(0), best_move(-1), age(0), generation(0) {}

    TTEntry(const TTEntry& other) : 
        hash(other.hash.load()), 
        depth(other.depth.load()), 
        value(other.value.load()), 
        flag(other.flag.load()), 
        best_move(other.best_move.load()), 
        age(other.age.load()),
        generation(other.generation.load()) {}

    TTEntry& operator=(const TTEntry& other) {
        if (this != &other) {
            hash.store(other.hash.load());
            depth.store(other.depth.load());
            value.store(other.value.load());
            flag.store(other.flag.load());
            best_move.store(other.best_move.load());
            age.store(other.age.load());
            generation.store(other.generation.load());
        }
        return *this;
    }
};

std::vector<TTEntry>& tt_table() {
    static std::vector<TTEntry>* table = []() {
        auto* ptr = new std::vector<TTEntry>();
        ptr->resize(TT_SIZE * 32);
        return ptr;
    }();
    return *table;
}

std::atomic<std::uint32_t>& tt_generation() {
    static std::atomic<std::uint32_t> gen{1};
    return gen;
}

auto& tt_mutexes() {
    static std::array<std::shared_mutex, 4096> mutexes;
    return mutexes;
}

std::vector<double>& global_weights() {
    static std::vector<double> w;
    return w;
}

std::vector<double>& global_order_map() {
    static std::vector<double> w;
    return w;
}

const TTEntry* probe_tt_entry(Bitboard p, Bitboard o) {
    Bitboard hv = zobrist_hash(p, o);
    std::size_t tx0 = static_cast<std::size_t>(hv % TT_SIZE) * 32;
    auto& table = tt_table();
    
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 8]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 16]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 24]), _MM_HINT_T0);
    
    const TTEntry* deep_entry = nullptr;
    const TTEntry* shallow_entry = nullptr;
    
    for (int i = 0; i < 32; ++i) {
        Bitboard h = table[tx0 + i].hash.load();
        if (h == hv) {
            if (i < 16) {
                return &table[tx0 + i];
            } else if (!deep_entry) {
                deep_entry = &table[tx0 + i];
            }
        }
    }
    
    return deep_entry;
}

struct SearchContext {
    const std::vector<double>* weights = nullptr;
    const std::vector<double>* order_map = nullptr;
    const std::vector<double>* root_policy = nullptr;
    std::chrono::steady_clock::time_point deadline{};
    bool use_deadline = false;
    bool timed_out = false;
    bool allow_tt = true;
    bool thread_safe_tt = false;
    bool nn_enabled = false;  
    bool multi_cut_enabled = false;
    int multi_cut_threshold = 3;
    int multi_cut_depth = 8;
    std::atomic<int>* ybwc_active_workers = nullptr;
    int ybwc_max_workers = 1;
    int ybwc_min_depth = 99;
    std::array<std::array<int, 2>, 64> killer_moves{};
    std::array<std::array<double, 64>, 2> history_scores{};
    std::array<std::array<double, 64>, 2> history_avg_depth{};
    std::array<std::array<int, 64>, 2> history_success_counts{};
};

std::pair<double, std::int64_t> alphabeta(Bitboard p, Bitboard o, int mvs, int depth, double alpha, double beta, bool passed, bool is_exact, SearchContext& ctx);

struct MPCThreshold {
    int depth;
    double alpha_threshold;
    double beta_threshold;
    int shallow_depth;
    double alpha_coeff;
    double beta_coeff;
};

constexpr std::array<MPCThreshold, 4> MPC_THRESHOLDS = {{
    {10, 2.2, -2.2, 4, 0.94, 0.94},  
    {12, 2.8, -2.8, 5, 0.96, 0.96},  
    {14, 3.3, -3.3, 6, 0.97, 0.97},
    {16, 3.8, -3.8, 7, 0.98, 0.98}
}};

inline bool try_mpc_pruning(Bitboard p, Bitboard o, int mvs, int depth, double alpha, double beta, bool is_exact, SearchContext& ctx, double& value, std::int8_t& flag) {
    if (is_exact || depth < 10) return false;  
    
    for (const auto& mpc : MPC_THRESHOLDS) {
        if (depth >= mpc.depth) {
            double mpc_alpha = alpha * mpc.alpha_coeff;
            double mpc_beta = beta * mpc.beta_coeff;
            
            auto [alpha_value, alpha_nodes] = alphabeta(p, o, mvs, mpc.shallow_depth, mpc_alpha, mpc_alpha + 1.0, false, is_exact, ctx);
            if (ctx.timed_out) return false;
            
            if (alpha_value <= mpc_alpha) {
                value = alpha_value;
                flag = 3;
                return true;
            }
            
            auto [beta_value, beta_nodes] = alphabeta(p, o, mvs, mpc.shallow_depth, mpc_beta - 1.0, mpc_beta, false, is_exact, ctx);
            if (ctx.timed_out) return false;
            
            if (beta_value >= mpc_beta) {
                value = beta_value;
                flag = 2;
                return true;
            }
        }
    }
    return false;
}

inline bool try_iid(Bitboard p, Bitboard o, int mvs, int depth, double alpha, double beta, bool is_exact, SearchContext& ctx, int& best_move) {
    if (depth >= 6 && best_move == -1) {
        int iid_depth = depth / 2;
        auto [iid_value, iid_nodes] = alphabeta(p, o, mvs, iid_depth, alpha, beta, false, is_exact, ctx);
        if (!ctx.timed_out) {
            const TTEntry* entry = probe_tt_entry(p, o);
            if (entry && entry->depth.load() >= iid_depth) {
                best_move = entry->best_move.load();
            }
        }
    }
    return best_move != -1;
}

inline bool try_futility_pruning(Bitboard p, Bitboard o, int mvs, int depth, double alpha, bool is_exact, SearchContext& ctx) {
    if (is_exact || depth > 3) return false;  

    const Bitboard valid = get_legal_moves_optimized(p, o);
    if ((valid & MASK_CORNER) != 0) return false;
    const int move_count = count_bits(valid);
    if (move_count == 0) return false;

    double futility_margin = 0.0;
    if (depth == 1) futility_margin = 42.0;  
    else if (depth == 2) futility_margin = 98.0 + static_cast<double>(move_count) * 3.5;  
    else if (depth == 3) futility_margin = 156.0 + static_cast<double>(move_count) * 3.0;  

    double static_eval = evaluate_board_full(p, o, mvs, *ctx.weights);
    return static_eval + futility_margin < alpha;
}

inline bool try_reverse_futility_pruning(Bitboard p, Bitboard o, int mvs, int depth, double beta, bool is_exact, SearchContext& ctx) {
    if (is_exact || depth > 3) return false;  

    const Bitboard valid = get_legal_moves_optimized(p, o);
    if ((valid & MASK_CORNER) != 0) return false;
    const int move_count = count_bits(valid);
    if (move_count == 0) return false;

    double reverse_margin = 0.0;
    if (depth == 1) reverse_margin = 48.0;  
    else if (depth == 2) reverse_margin = 112.0 + static_cast<double>(move_count) * 3.5;  
    else if (depth == 3) reverse_margin = 168.0 + static_cast<double>(move_count) * 3.0;  

    double static_eval = evaluate_board_full(p, o, mvs, *ctx.weights);
    return static_eval - reverse_margin > beta;
}

inline std::size_t tt_mutex_index(Bitboard hv) {
    return static_cast<std::size_t>((hv ^ (hv >> 32)) & 4095ULL);
}

inline int tt_replace_priority(const TTEntry& entry) {
    Bitboard h = entry.hash.load();
    if (h == 0) return -1;
    int d = entry.depth.load();
    std::int8_t f = entry.flag.load();
    std::uint8_t g = entry.generation.load();
    std::uint32_t current_gen = tt_generation().load();
    int flag_bonus = f == 1 ? 32 : (f == 2 ? 12 : 8);
    int generation_bonus = (g == static_cast<std::uint8_t>(current_gen)) ? 16 : 0;
    int age_penalty = static_cast<int>((current_gen - g) & 0xFF);
    return d * 64 + flag_bonus + generation_bonus - age_penalty;
}

inline int tt_replace_priority(int depth, std::int8_t flag) {
    int flag_bonus = flag == 1 ? 32 : (flag == 2 ? 12 : 8);
    return depth * 64 + flag_bonus + 16;
}

inline void store_tt(Bitboard hv, int depth, double value, std::int8_t flag, int best_move, bool thread_safe_tt) {
    auto& table = tt_table();
    std::size_t tx0 = static_cast<std::size_t>(hv % TT_SIZE) * 32;
    std::size_t deep_end = tx0 + 16;
    std::size_t shallow_end = tx0 + 32;
    std::unique_lock<std::shared_mutex> lock(tt_mutexes()[tt_mutex_index(hv)], std::defer_lock);
    if (thread_safe_tt) lock.lock();
    
    TTEntry new_entry{};
    new_entry.hash.store(hv);
    new_entry.depth.store(depth);
    new_entry.value.store(value);
    new_entry.flag.store(flag);
    new_entry.best_move.store(static_cast<std::int8_t>(best_move));
    new_entry.age.store(0);
    new_entry.generation.store(static_cast<std::uint8_t>(tt_generation().load()));
    
    int new_priority = tt_replace_priority(depth, flag);
    
    for (int i = 0; i < 16; ++i) {
        Bitboard h = table[tx0 + i].hash.load();
        if (h == hv) {
            if (new_priority >= tt_replace_priority(table[tx0 + i])) {
                table[tx0 + i] = new_entry;
            }
            return;
        }
    }
    
    if (depth >= 6) {
        int min_priority = INT_MAX;
        int replace_idx = 0;
        for (int i = 0; i < 16; ++i) {
            Bitboard h = table[tx0 + i].hash.load();
            int p = tt_replace_priority(table[tx0 + i]);
            if (h == 0 || p < min_priority) {
                min_priority = p;
                replace_idx = i;
                if (h == 0) break;
            }
        }
        
        if (new_priority >= min_priority) {
            table[tx0 + replace_idx] = new_entry;
            return;
        }
    }
    
    int min_priority = INT_MAX;
    int replace_idx = 16;
    for (int i = 16; i < 32; ++i) {
        Bitboard h = table[tx0 + i].hash.load();
        int p = tt_replace_priority(table[tx0 + i]);
        if (h == 0 || p < min_priority) {
            min_priority = p;
            replace_idx = i;
            if (h == 0) break;
        }
    }
    
    if (new_priority >= min_priority) {
        table[replace_idx] = new_entry;
    }
}

std::vector<int> reorder_root_moves(const std::vector<int>& ordered_indices, int mvs, const SearchContext& ctx) {
    if (ordered_indices.size() <= 1) return ordered_indices;
    std::vector<int> reordered = ordered_indices;
    std::array<int, 64> prev_rank{};
    prev_rank.fill(static_cast<int>(ordered_indices.size()));
    for (std::size_t i = 0; i < ordered_indices.size(); ++i) {
        prev_rank[static_cast<std::size_t>(ordered_indices[i])] = static_cast<int>(i);
    }
    std::stable_sort(reordered.begin(), reordered.end(), [&](int a, int b) {
        double score_a = (*ctx.order_map)[static_cast<std::size_t>(a)];
        double score_b = (*ctx.order_map)[static_cast<std::size_t>(b)];
        if (ctx.root_policy && ctx.root_policy->size() == 64) {
            double policy_scale = mvs <= 16 ? 700.0 : (mvs <= 28 ? 520.0 : 320.0);
            score_a += (*ctx.root_policy)[static_cast<std::size_t>(a)] * policy_scale;
            score_b += (*ctx.root_policy)[static_cast<std::size_t>(b)] * policy_scale;
        }
        score_a -= static_cast<double>(prev_rank[static_cast<std::size_t>(a)]) * 0.25;
        score_b -= static_cast<double>(prev_rank[static_cast<std::size_t>(b)]) * 0.25;
        return score_a > score_b;
    });
    return reordered;
}

inline bool probe_tt(Bitboard hv, int depth, double alpha, double beta, int& tm, double& value, bool thread_safe_tt, bool is_exact) {
    std::shared_lock<std::shared_mutex> lock(tt_mutexes()[tt_mutex_index(hv)], std::defer_lock);
    if (thread_safe_tt) lock.lock();
    
    auto& table = tt_table();
    std::size_t tx0 = static_cast<std::size_t>(hv % TT_SIZE) * 32;
    
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 8]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 16]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&table[tx0 + 24]), _MM_HINT_T0);
    
    for (int i = 0; i < 32; ++i) {
        Bitboard h = table[tx0 + i].hash.load();
        if (h == hv) {
            int d = table[tx0 + i].depth.load();
            if (d >= depth) {
                std::int8_t f = table[tx0 + i].flag.load();
                double v = table[tx0 + i].value.load();
                tm = table[tx0 + i].best_move.load();
                if (f == 1 || (f == 2 && v >= beta) || (f == 3 && v <= alpha)) {
                    // If we need an exact result but the entry is heuristic, ignore it
                    bool is_exact_entry = (std::abs(v) > 20000.0);
                    if (is_exact && !is_exact_entry) continue;
                    
                    value = v;
                    return true;
                }
            }
        }
    }
    
    return false;
}

inline bool check_timeout(SearchContext& ctx) {
    if (ctx.use_deadline && std::chrono::steady_clock::now() >= ctx.deadline) {
        ctx.timed_out = true;
    }
    return ctx.timed_out;
}

inline int hardware_thread_budget() {
    unsigned int hw = std::thread::hardware_concurrency();
    return static_cast<int>(hw == 0 ? 8 : hw);
}

inline int acquire_ybwc_workers(SearchContext& ctx, int requested) {
    if (ctx.ybwc_active_workers == nullptr || ctx.ybwc_max_workers <= 0 || requested <= 0) {
        return 0;
    }
    int active = ctx.ybwc_active_workers->load(std::memory_order_relaxed);
    while (true) {
        const int available = std::max(0, ctx.ybwc_max_workers - active);
        const int acquired = std::min(requested, available);
        if (acquired <= 0) {
            return 0;
        }
        if (ctx.ybwc_active_workers->compare_exchange_weak(
                active,
                active + acquired,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return acquired;
        }
    }
}

inline void release_ybwc_workers(SearchContext& ctx, int count) {
    if (ctx.ybwc_active_workers != nullptr && count > 0) {
        ctx.ybwc_active_workers->fetch_sub(count, std::memory_order_acq_rel);
    }
}

inline void raise_atomic_double(std::atomic<double>& target, double value) {
    double current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current, value, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
}

inline double estimate_search_complexity(Bitboard p, Bitboard o, Bitboard valid) {
    const int empties = 64 - count_bits(p | o);
    const int own_moves = count_bits(valid);
    const int opp_moves = count_bits(get_legal_moves_optimized(o, p));
    const int mobility_total = own_moves + opp_moves;
    const int mobility_gap = std::abs(own_moves - opp_moves);
    const int corner_pressure = count_bits(valid & MASK_CORNER);
    const int edge_pressure = count_bits(valid & MASK_EDGE);

    double complexity = 0.0;
    complexity += std::min(1.0, static_cast<double>(empties) / 24.0) * 0.30;
    complexity += std::min(1.0, static_cast<double>(mobility_total) / 18.0) * 0.35;
    complexity += std::min(1.0, static_cast<double>(mobility_gap) / 10.0) * 0.10;
    complexity += corner_pressure > 0 ? 0.15 : 0.0;
    complexity += edge_pressure >= 4 ? 0.10 : 0.0;
    return std::max(0.0, std::min(1.0, complexity));
}

inline double compute_dynamic_pvs_window(int depth, double complexity, std::size_t move_index, bool is_exact) {
    if (is_exact) return 1.0;

    double window = 1.0 + complexity * 1.10;
    if (depth >= 8) window += 0.20;
    if (depth >= 12) window += 0.25;
    if (move_index >= 4) window += 0.15;
    return std::max(1.0, std::min(3.5, window));
}

inline int compute_dynamic_lmr_reduction(
    int depth,
    std::size_t move_index,
    std::size_t move_count,
    double complexity,
    double score_gap,
    bool nn_enabled
) {
    
    int reduction = 1 + (move_index >= 3 ? 1 : 0) + (move_index >= 6 ? 1 : 0) + (move_index >= 10 ? 1 : 0);
    reduction += (depth >= 6 ? 1 : 0) + (depth >= 10 ? 1 : 0);  
    if (move_count >= 8) reduction += 1;  

    if (complexity >= 0.65) reduction -= 1;  
    else if (complexity <= 0.35) reduction += 1;

    if (score_gap <= 32.0) reduction -= 1;  
    else if (score_gap >= 128.0) reduction += 1;  

    if (nn_enabled) reduction -= 1;
    return std::max(0, reduction);
}

inline std::size_t compute_late_move_pruning_start(int depth, std::size_t move_count, double complexity) {
    std::size_t start = 0;
    if (depth <= 1) start = 2;
    else if (depth == 2) start = 3;  
    else if (depth == 3) start = 5;  
    else return move_count + 1;

    if (complexity <= 0.35) start += 2;  
    else if (complexity >= 0.60 && start > 1) start -= 1;  

    return std::min(move_count + 1, start);
}

inline bool is_pruning_sensitive_move(Bitboard bit, int flip_count) {
    return (bit & (MASK_CORNER | MASK_EDGE | MASK_X | MASK_C)) != 0 || flip_count >= 8;
}

struct OrderedMove {
    int sq = -1;
    Bitboard bit = 0;
    Bitboard flip = 0;
    int flip_count = 0;
    double score = 0.0;
};

inline bool should_use_nn_move_ordering(bool nn_enabled, int depth, std::size_t move_count, double complexity) {
    return nn_enabled && depth >= 5 && move_count <= 12 && complexity <= 0.88;
}

inline void age_history_scores(SearchContext& ctx, std::size_t side) {
    for (double& score : ctx.history_scores[side]) {
        score *= 0.95;
    }
}

inline void record_history_cutoff(SearchContext& ctx, int mvs, int sq, int depth) {
    const std::size_t side = static_cast<std::size_t>(mvs & 1);
    const std::size_t move = static_cast<std::size_t>(sq);
    age_history_scores(ctx, side);

    int& success_count = ctx.history_success_counts[side][move];
    double& avg_depth = ctx.history_avg_depth[side][move];
    double& history_score = ctx.history_scores[side][move];

    success_count += 1;
    avg_depth += (static_cast<double>(depth) - avg_depth) / static_cast<double>(success_count);
    history_score += static_cast<double>(depth * depth) + avg_depth * 0.5 + static_cast<double>(std::min(success_count, 32)) * 2.0;
}

inline double compute_move_order_bonus(const SearchContext& ctx, int mvs, int sq, Bitboard bit, int tt_move) {
    double score = 0.0;
    if (sq == tt_move) score += 1e12;

    const int ply_idx = std::max(0, std::min(63, mvs));
    if (ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0] == sq) score += 1e11;
    else if (ctx.killer_moves[static_cast<std::size_t>(ply_idx)][1] == sq) score += 5e10;

    const std::size_t side = static_cast<std::size_t>(mvs & 1);
    score += ctx.history_scores[side][static_cast<std::size_t>(sq)] * 128.0;
    score += ctx.history_avg_depth[side][static_cast<std::size_t>(sq)] * 32.0;
    score += static_cast<double>(std::min(32, ctx.history_success_counts[side][static_cast<std::size_t>(sq)])) * 64.0;

    if ((bit & MASK_CORNER) != 0) score += 1e9;
    else if ((bit & MASK_EDGE) != 0) score += 5e7;
    if ((bit & MASK_X) != 0) score -= 3e8;
    else if ((bit & MASK_C) != 0) score -= 1e8;

    return score;
}


inline bool try_nn_pruning(Bitboard p, Bitboard o, int mvs, int depth, double alpha, double beta, bool is_exact, SearchContext& ctx, double& value, std::int8_t& flag) {
    if (is_exact || depth < 6 || !ctx.nn_enabled) return false;  

    const double nn_eval = evaluate_board_full(p, o, mvs, *ctx.weights);
    const double nn_margin = 108.0 + static_cast<double>(depth) * 16.0;  

    if (beta < 1e17 && nn_eval >= beta + nn_margin) {
        value = nn_eval;
        flag = 2;
        return true;
    }

    if (alpha > -1e17 && nn_eval <= alpha - nn_margin) {
        value = nn_eval;
        flag = 3;
        return true;
    }

    return false;
}


inline int fill_ordered_moves_with_nn(
    Bitboard p,
    Bitboard o,
    Bitboard valid,
    int mvs,
    SearchContext& ctx,
    int tt_move,
    std::array<OrderedMove, 64>& moves
) {
    int move_count = 0;
    while (valid) {
        Bitboard bit = lsb(valid);
        valid ^= bit;
        int idx = bit_index(bit);
        
        Bitboard f = get_flip_optimized(p, o, idx);
        Bitboard np = (p | bit | f) & FULL_MASK;
        Bitboard no = (o ^ f) & FULL_MASK;
        
        
        double move_score = evaluate_board_full(np, no, mvs + 1, *ctx.weights);
        move_score += compute_move_order_bonus(ctx, mvs, idx, bit, tt_move);

        moves[move_count++] = OrderedMove{idx, bit, f, count_bits(f), move_score};
    }
    
    
    std::sort(moves.begin(), moves.begin() + move_count,
              [](const OrderedMove& a, const OrderedMove& b) { return a.score > b.score; });

    return move_count;
}

std::pair<double, std::int64_t> alphabeta(Bitboard p, Bitboard o, int mvs, int depth, double alpha, double beta, bool passed, bool is_exact, SearchContext& ctx) {
    Bitboard hv = zobrist_hash(p, o);
    int tm = -1;
    double oa = alpha;
    const bool tt_allowed = ctx.allow_tt;
    bool static_eval_cached = false;
    double static_eval = 0.0;
    auto get_static_eval = [&]() -> double {
        if (!static_eval_cached) {
            static_eval = evaluate_board_full(p, o, mvs, *ctx.weights);
            static_eval_cached = true;
        }
        return static_eval;
    };

    if (tt_allowed) {
        double tt_value = 0.0;
        if (probe_tt(hv, depth, alpha, beta, tm, tt_value, ctx.thread_safe_tt, is_exact)) return {tt_value, 1};
    }

    Bitboard valid = get_legal_moves_optimized(p, o);
    if (!valid) {
        if (passed) {
            double v = exact_eval(p, o, mvs, 0ULL);
            if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, v, 1, -1, ctx.thread_safe_tt);
            return {v, 1};
        }
        auto [v, n] = alphabeta(o, p, mvs, depth, -beta, -alpha, true, is_exact, ctx);
        return {-v, n + 1};
    }

    if (depth <= 0) {
        // Only use exact_eval if we are certain to be near the end of the game.
        // If depth reached 0 but the game isn't over, heuristics are much more reliable.
        const int empties = 64 - mvs;
        double v = (is_exact && depth >= empties) ? exact_eval(p, o, mvs, valid) : evaluate_board_full(p, o, mvs, *ctx.weights);
        if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, v, 1, -1, ctx.thread_safe_tt);
        return {v, 1};
    }

    double pruned_value = 0.0;
    std::int8_t pruned_flag = 0;
    if (try_nn_pruning(p, o, mvs, depth, alpha, beta, is_exact, ctx, pruned_value, pruned_flag)) {
        if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, pruned_value, pruned_flag, tm, ctx.thread_safe_tt);
        return {pruned_value, 1};
    }

    if (try_mpc_pruning(p, o, mvs, depth, alpha, beta, is_exact, ctx, pruned_value, pruned_flag)) {
        if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, pruned_value, pruned_flag, tm, ctx.thread_safe_tt);
        return {pruned_value, 1};
    }

    if (try_reverse_futility_pruning(p, o, mvs, depth, beta, is_exact, ctx)) {
        const double eval = get_static_eval();
        if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, eval, 2, tm, ctx.thread_safe_tt);
        return {eval, 1};
    }

    if (try_futility_pruning(p, o, mvs, depth, alpha, is_exact, ctx)) {
        const double eval = get_static_eval();
        if (!ctx.timed_out && tt_allowed) store_tt(hv, depth, eval, 3, tm, ctx.thread_safe_tt);
        return {eval, 1};
    }

    if (try_iid(p, o, mvs, depth, alpha, beta, is_exact, ctx, tm) && tt_allowed) {
        const TTEntry* entry = probe_tt_entry(p, o);
        if (entry) tm = entry->best_move.load();
    }

    // Null-move pruning is intentionally disabled for Othello.
    // Pass is a real game mechanic here, so "skip a move and still hold beta"
    // is much less reliable than in chess-like search and was causing tactical misses.

    tt_generation().fetch_add(1, std::memory_order_relaxed);
    const double position_complexity = estimate_search_complexity(p, o, valid);
    const std::size_t legal_move_count = static_cast<std::size_t>(count_bits(valid));
    const bool use_nn_ordering = should_use_nn_move_ordering(ctx.nn_enabled, depth, legal_move_count, position_complexity);
    if (!is_exact && depth <= 3) {
        (void)get_static_eval();
    }

    std::array<OrderedMove, 64> ordered_moves{};
    int move_count = 0;
    if (use_nn_ordering) {
        move_count = fill_ordered_moves_with_nn(p, o, valid, mvs, ctx, tm, ordered_moves);
    } else {
        
        Bitboard v_temp = valid;
        while (v_temp) {
            Bitboard bit = lsb(v_temp);
            v_temp ^= bit;
            int sq = bit_index(bit);
            Bitboard flip = get_flip_optimized(p, o, sq);
            int flip_count = count_bits(flip);
            double score = (*ctx.order_map)[static_cast<std::size_t>(sq)] + compute_move_order_bonus(ctx, mvs, sq, bit, tm);
            if (!is_exact && depth <= 3) {
                score += get_static_eval() * 0.01;
            }
            score += static_cast<double>(flip_count) * (position_complexity <= 0.45 ? 18.0 : 8.0);

            ordered_moves[move_count++] = OrderedMove{sq, bit, flip, flip_count, score};
        }
        
        
        std::sort(ordered_moves.begin(), ordered_moves.begin() + move_count,
                  [](const OrderedMove& a, const OrderedMove& b) { return a.score > b.score; });
    }
    const std::size_t late_move_pruning_start = !is_exact
        ? compute_late_move_pruning_start(depth, static_cast<std::size_t>(move_count), position_complexity)
        : 0;

    double max_val = -1e18;
    int bm = -1;
    std::int64_t nodes = 1;
    bool found_pv = false;

    // Multi-cut pruning: shallow search of first few moves; if enough exceed beta, cut
    // Disabled in opening (mvs < 20) and endgame (empties <= 20) to avoid tactical misses
    const int empties = 64 - mvs;
    if (ctx.multi_cut_enabled && depth >= ctx.multi_cut_depth && move_count >= ctx.multi_cut_threshold && mvs >= 20 && empties > 20) {
        int cut_count = 0;
        double mc_best = -1e18;
        const int mc_shallow_depth = std::max(2, depth / 3);
        for (int i = 0; i < std::min(move_count, ctx.multi_cut_threshold + 2); ++i) {
            const OrderedMove& move = ordered_moves[static_cast<std::size_t>(i)];
            Bitboard np = (p | move.bit | move.flip) & FULL_MASK;
            Bitboard no = (o ^ move.flip) & FULL_MASK;
            auto [v, n] = alphabeta(no, np, mvs + 1, mc_shallow_depth, -beta, -alpha, false, is_exact, ctx);
            if (ctx.timed_out) break;
            double mc_val = -v;
            if (mc_val >= beta) {
                ++cut_count;
                if (mc_val > mc_best) mc_best = mc_val;
            }
        }
        if (cut_count >= ctx.multi_cut_threshold) {
            if (!ctx.timed_out && tt_allowed) store_tt(hv, mc_shallow_depth, mc_best, 3, bm, ctx.thread_safe_tt);
            return {mc_best, nodes};
        }
    }

    if (depth >= g_search_settings.ybwc_min_depth &&
        move_count >= 2 &&
        ctx.ybwc_active_workers != nullptr &&
        !check_timeout(ctx)) {
        std::vector<double> ybwc_vals(static_cast<std::size_t>(move_count), -1e18);
        std::vector<std::int64_t> ybwc_nodes(static_cast<std::size_t>(move_count), 0);
        auto search_ybwc_child = [&](int i, SearchContext& child_ctx, double alpha_snapshot) {
            const OrderedMove& move = ordered_moves[static_cast<std::size_t>(i)];
            Bitboard np = (p | move.bit | move.flip) & FULL_MASK;
            Bitboard no = (o ^ move.flip) & FULL_MASK;
            auto res = alphabeta(no, np, mvs + 1, depth - 1, -beta, -alpha_snapshot, false, true, child_ctx);
            return std::make_pair(-res.first, res.second);
        };

        auto first_res = search_ybwc_child(0, ctx, alpha);
        ybwc_vals[0] = first_res.first;
        ybwc_nodes[0] = first_res.second;
        max_val = first_res.first;
        bm = ordered_moves[0].sq;
        nodes += first_res.second;
        if (!ctx.timed_out && first_res.first > alpha) {
            alpha = first_res.first;
        }
        if (ctx.timed_out || alpha >= beta) {
            if (alpha >= beta) {
                int ply_idx = std::max(0, std::min(63, mvs));
                if (bm != ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0]) {
                    ctx.killer_moves[static_cast<std::size_t>(ply_idx)][1] = ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0];
                    ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0] = bm;
                }
                record_history_cutoff(ctx, mvs, bm, depth);
            }
            return {max_val, nodes};
        }

        const int requested_workers = std::min(move_count - 1, ctx.ybwc_max_workers);
        const int ybwc_worker_count = acquire_ybwc_workers(ctx, requested_workers);
        if (ybwc_worker_count > 0) {
            std::atomic<int> next_child{1};
            std::atomic<double> shared_alpha{alpha};
            std::atomic<bool> cutoff{false};
            std::atomic<bool> saw_timeout{false};
            std::vector<std::thread> ybwc_workers;
            ybwc_workers.reserve(static_cast<std::size_t>(ybwc_worker_count));
            for (int worker_id = 0; worker_id < ybwc_worker_count; ++worker_id) {
                ybwc_workers.emplace_back([&, worker_id]() {
                    SearchContext local_ctx = ctx;
                    local_ctx.thread_safe_tt = true;
                    while (!cutoff.load(std::memory_order_relaxed)) {
                        const int i = next_child.fetch_add(1, std::memory_order_relaxed);
                        if (i >= move_count) {
                            break;
                        }
                        if (check_timeout(local_ctx)) {
                            saw_timeout.store(true, std::memory_order_relaxed);
                            break;
                        }
                        const double alpha_snapshot = shared_alpha.load(std::memory_order_acquire);
                        if (alpha_snapshot >= beta) {
                            cutoff.store(true, std::memory_order_relaxed);
                            break;
                        }
                        auto child_res = search_ybwc_child(i, local_ctx, alpha_snapshot);
                        ybwc_vals[static_cast<std::size_t>(i)] = child_res.first;
                        ybwc_nodes[static_cast<std::size_t>(i)] = child_res.second;
                        if (local_ctx.timed_out) {
                            saw_timeout.store(true, std::memory_order_relaxed);
                            break;
                        }
                        if (child_res.first > alpha_snapshot) {
                            raise_atomic_double(shared_alpha, child_res.first);
                            if (child_res.first >= beta) {
                                cutoff.store(true, std::memory_order_relaxed);
                                break;
                            }
                        }
                    }
                });
            }
            for (auto& worker : ybwc_workers) {
                worker.join();
            }
            release_ybwc_workers(ctx, ybwc_worker_count);
            if (saw_timeout.load(std::memory_order_relaxed)) {
                ctx.timed_out = true;
            }
        } else {
            for (int i = 1; i < move_count; ++i) {
                if (check_timeout(ctx) || alpha >= beta) {
                    break;
                }
                auto child_res = search_ybwc_child(i, ctx, alpha);
                ybwc_vals[static_cast<std::size_t>(i)] = child_res.first;
                ybwc_nodes[static_cast<std::size_t>(i)] = child_res.second;
                if (ctx.timed_out) {
                    break;
                }
                if (child_res.first > alpha) {
                    alpha = child_res.first;
                }
            }
        }

        nodes = 1;
        max_val = -1e18;
        bm = -1;
        for (int i = 0; i < move_count; ++i) {
            const auto child_nodes = ybwc_nodes[static_cast<std::size_t>(i)];
            if (child_nodes <= 0) {
                continue;
            }
            nodes += child_nodes;
            const double child_val = ybwc_vals[static_cast<std::size_t>(i)];
            if (child_val > max_val) {
                max_val = child_val;
                bm = ordered_moves[static_cast<std::size_t>(i)].sq;
            }
        }
        if (bm >= 0 && max_val >= beta) {
            int ply_idx = std::max(0, std::min(63, mvs));
            if (bm != ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0]) {
                ctx.killer_moves[static_cast<std::size_t>(ply_idx)][1] = ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0];
                ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0] = bm;
            }
            record_history_cutoff(ctx, mvs, bm, depth);
        }
        return {max_val, nodes};
    }

    for (int i = 0; i < move_count; ++i) {
        if (check_timeout(ctx)) break;
        const OrderedMove& move = ordered_moves[static_cast<std::size_t>(i)];
        int sq = move.sq;
        Bitboard bit = move.bit;
        Bitboard f = move.flip;
        Bitboard np = (p | bit | f) & FULL_MASK;
        Bitboard no = (o ^ f) & FULL_MASK;
        double val = 0.0;
        std::int64_t n = 0;
        const double pvs_window = compute_dynamic_pvs_window(depth, position_complexity, static_cast<std::size_t>(i), is_exact);
        const double score_gap = i > 0 ? std::max(0.0, ordered_moves[static_cast<std::size_t>(i - 1)].score - move.score) : 0.0;
        const int flip_count = move.flip_count;

        if (!is_exact &&
            depth <= 3 &&
            i > 0 &&
            static_cast<std::size_t>(i) >= late_move_pruning_start &&
            alpha > -1e17 &&
            !is_pruning_sensitive_move(bit, flip_count)) {
            const double late_move_margin = 44.0 + static_cast<double>(depth) * 20.0 + static_cast<double>(flip_count) * 6.0;
            if (get_static_eval() + late_move_margin <= alpha) {
                continue;
            }
        }

        int reduction = 0;
        if (!is_exact &&
            depth >= 4 &&
            i >= 2 &&
            move_count >= 4 &&
            sq != ctx.killer_moves[static_cast<std::size_t>(std::max(0, std::min(63, mvs)))][0] &&
            sq != ctx.killer_moves[static_cast<std::size_t>(std::max(0, std::min(63, mvs)))][1] &&
            !(sq == 0 || sq == 7 || sq == 56 || sq == 63) &&
            flip_count <= 10 &&
            position_complexity < 0.90) {
            reduction = compute_dynamic_lmr_reduction(
                depth,
                static_cast<std::size_t>(i),
                static_cast<std::size_t>(move_count),
                position_complexity,
                score_gap,
                use_nn_ordering
            );
            if (reduction > depth - 2) reduction = depth - 2;
        }

        if (found_pv) {
            auto res = alphabeta(no, np, mvs + 1, depth - 1, -alpha - pvs_window, -alpha, false, is_exact, ctx);
            val = -res.first;
            n = res.second;
            if (!ctx.timed_out && val > alpha && val < beta) {
                auto full_res = alphabeta(no, np, mvs + 1, depth - 1, -beta, -alpha, false, is_exact, ctx);
                val = -full_res.first;
                n += full_res.second;
            }
        } else if (reduction > 0) {
            auto res = alphabeta(no, np, mvs + 1, depth - 1 - reduction, -alpha - pvs_window, -alpha, false, is_exact, ctx);
            val = -res.first;
            n = res.second;
            if (!ctx.timed_out && val > alpha) {
                auto full_res = alphabeta(no, np, mvs + 1, depth - 1, -beta, -alpha, false, is_exact, ctx);
                val = -full_res.first;
                n += full_res.second;
            }
        } else {
            auto res = alphabeta(no, np, mvs + 1, depth - 1, -beta, -alpha, false, is_exact, ctx);
            val = -res.first;
            n = res.second;
        }

        if (ctx.timed_out) break;


        if (val > max_val) {
            max_val = val;
            bm = sq;
            found_pv = true;
        }

        nodes += n;

        if (val > alpha) {
            alpha = val;
            if (val >= beta) {

                int ply_idx = std::max(0, std::min(63, mvs));
                if (sq != ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0]) {
                    ctx.killer_moves[static_cast<std::size_t>(ply_idx)][1] = ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0];
                    ctx.killer_moves[static_cast<std::size_t>(ply_idx)][0] = sq;
                }
                record_history_cutoff(ctx, mvs, sq, depth);
                break;
            }
        }
    }
    
    
    if (!ctx.timed_out && tt_allowed) {
        int flag = 1;
        if (max_val >= beta) flag = 2;
        else if (max_val <= oa) flag = 3;
        store_tt(hv, depth, max_val, flag, bm, ctx.thread_safe_tt);
    }
    
    return {max_val, nodes};
}

std::pair<std::vector<double>, std::vector<std::int64_t>> search_root_parallel_impl(Bitboard p, Bitboard o, int mvs, int depth, bool is_exact, const std::vector<int>& ordered_indices, SearchContext& ctx) {
    ctx.multi_cut_enabled = g_search_settings.multi_cut_enabled;
    ctx.multi_cut_threshold = g_search_settings.multi_cut_threshold;
    ctx.multi_cut_depth = g_search_settings.multi_cut_depth;
    ctx.ybwc_min_depth = g_search_settings.ybwc_min_depth;

    std::atomic<int> ybwc_active_workers{0};
    struct YbwcScope {
        SearchContext& ctx;
        bool owns;
        std::atomic<int>* previous_active;
        int previous_max;
        ~YbwcScope() {
            if (owns) {
                ctx.ybwc_active_workers = previous_active;
                ctx.ybwc_max_workers = previous_max;
            }
        }
    } ybwc_scope{ctx, ctx.ybwc_active_workers == nullptr, ctx.ybwc_active_workers, ctx.ybwc_max_workers};
    if (ybwc_scope.owns) {
        ctx.ybwc_active_workers = &ybwc_active_workers;
        ctx.ybwc_max_workers = std::max(1, hardware_thread_budget() - 1);
    }
    std::vector<int> root_order = reorder_root_moves(ordered_indices, mvs, ctx);
    auto align_to_requested_order = [&](const std::vector<double>& raw_vals, const std::vector<std::int64_t>& raw_nodes) {
        if (root_order == ordered_indices) {
            return std::make_pair(raw_vals, raw_nodes);
        }
        std::array<int, 64> move_pos{};
        move_pos.fill(-1);
        for (std::size_t i = 0; i < root_order.size(); ++i) {
            move_pos[static_cast<std::size_t>(root_order[i])] = static_cast<int>(i);
        }
        std::vector<double> aligned_vals(ordered_indices.size(), 0.0);
        std::vector<std::int64_t> aligned_nodes(ordered_indices.size(), 0);
        for (std::size_t i = 0; i < ordered_indices.size(); ++i) {
            int pos = move_pos[static_cast<std::size_t>(ordered_indices[i])];
            if (pos >= 0) {
                aligned_vals[i] = raw_vals[static_cast<std::size_t>(pos)];
                aligned_nodes[i] = raw_nodes[static_cast<std::size_t>(pos)];
            }
        }
        return std::make_pair(std::move(aligned_vals), std::move(aligned_nodes));
    };

    std::vector<double> vals(root_order.size(), -1e18);
    std::vector<std::int64_t> nodes(root_order.size(), 0);

    const int total_budget = hardware_thread_budget();
    
    // 1. PV-First: Parallelized if we have enough budget to avoid idle cores.
    std::size_t start_idx = 0;
    if (total_budget <= 4 && root_order.size() > 1 && depth >= 6) {
        const int idx = root_order[0];
        Bitboard bit = 1ULL << idx;
        Bitboard f = get_flip_optimized(p, o, idx);
        Bitboard np = (p | bit | f) & FULL_MASK;
        Bitboard no = (o ^ f) & FULL_MASK;
        auto [v, n] = alphabeta(no, np, mvs + 1, depth - 1, -1e18, 1e18, false, is_exact, ctx);
        vals[0] = -v;
        nodes[0] = n;
        if (ctx.timed_out) return align_to_requested_order(vals, nodes);
        start_idx = 1;
    }

    // 2. Parallel Search with Splitting
    struct Task {
        std::size_t root_idx;
        int sub_idx;
        Bitboard p, o;
    };
    std::vector<Task> tasks;
    for (std::size_t i = start_idx; i < root_order.size(); ++i) {
        const int idx = root_order[i];
        Bitboard bit = 1ULL << idx;
        Bitboard f = get_flip_optimized(p, o, idx);
        Bitboard np = (p | bit | f) & FULL_MASK;
        Bitboard no = (o ^ f) & FULL_MASK;

        if ((root_order.size() - start_idx) < static_cast<std::size_t>(total_budget / 2) && depth >= 6) {
            Bitboard sub_valid = get_legal_moves_optimized(no, np);
            if (sub_valid) {
                while (sub_valid) {
                    Bitboard sub_bit = lsb(sub_valid);
                    sub_valid ^= sub_bit;
                    tasks.push_back({i, bit_index(sub_bit), no, np});
                }
                continue;
            }
        }
        tasks.push_back({i, -1, p, o});
    }

    std::vector<std::mutex> root_mutexes(root_order.size());
    std::atomic<std::size_t> next_task{0};
    std::atomic<bool> saw_timeout{false};
    std::vector<std::thread> workers;
    int worker_count = std::min<int>(total_budget, static_cast<int>(tasks.size()));
    if (worker_count < 1 && !tasks.empty()) worker_count = 1;

    for (int w = 0; w < worker_count; ++w) {
        workers.emplace_back([&]() {
            SearchContext local_ctx = ctx;
            local_ctx.thread_safe_tt = true;
            while (true) {
                size_t t = next_task.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks.size()) break;
                if (check_timeout(local_ctx)) { saw_timeout.store(true); break; }

                const auto& task = tasks[t];
                if (task.sub_idx == -1) {
                    int idx = root_order[task.root_idx];
                    Bitboard f = get_flip_optimized(p, o, idx);
                    Bitboard np = (p | (1ULL << idx) | f) & FULL_MASK;
                    Bitboard no = (o ^ f) & FULL_MASK;
                    auto [v, n] = alphabeta(no, np, mvs + 1, depth - 1, -1e18, 1e18, false, is_exact, local_ctx);
                    std::lock_guard<std::mutex> lock(root_mutexes[task.root_idx]);
                    vals[task.root_idx] = -v;
                    nodes[task.root_idx] += n;
                } else {
                    Bitboard f = get_flip_optimized(task.p, task.o, task.sub_idx);
                    Bitboard np = (task.p | (1ULL << task.sub_idx) | f) & FULL_MASK;
                    Bitboard no = (task.o ^ f) & FULL_MASK;
                    auto [v, n] = alphabeta(no, np, mvs + 2, depth - 2, -1e18, 1e18, false, is_exact, local_ctx);
                    std::lock_guard<std::mutex> lock(root_mutexes[task.root_idx]);
                    if (vals[task.root_idx] < -1e17 || v < vals[task.root_idx]) {
                        vals[task.root_idx] = v;
                    }
                    nodes[task.root_idx] += n;
                }
                if (local_ctx.timed_out) saw_timeout.store(true);
            }
        });
    }

    for (auto& w : workers) w.join();
    if (saw_timeout) ctx.timed_out = true;
    return align_to_requested_order(vals, nodes);
}

double calculate_win_rate(double ev, bool is_exact) {
    if (is_exact) {
        if (ev > 0.0) return 100.0;
        else if (ev < 0.0) return 0.0;
        else return 50.0;
    }
    double x = std::max(-4000.0, std::min(4000.0, ev));
    return 100.0 / (1.0 + std::exp(-x / 400.0));
}

std::vector<int> legal_move_indices(Bitboard p, Bitboard o) {
    Bitboard valid = get_legal_moves_optimized(p, o);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(count_bits(valid)));
    while (valid) {
        Bitboard bit = lsb(valid);
        valid ^= bit;
        out.push_back(bit_index(bit));
    }
    return out;
}

int count_empty_regions(Bitboard empty_mask, int* small_region_count = nullptr) {
    int regions = 0;
    int small_regions = 0;
    Bitboard remaining = empty_mask & FULL_MASK;
    while (remaining) {
        Bitboard region = lsb(remaining);
        Bitboard frontier = region;
        while (frontier) {
            Bitboard expanded = neighbor_union(frontier) & empty_mask & (~region);
            region |= expanded;
            frontier = expanded;
        }
        const int region_size = count_bits(region);
        if (region_size <= 2) {
            ++small_regions;
        }
        remaining &= ~region;
        ++regions;
    }
    if (small_region_count != nullptr) {
        *small_region_count = small_regions;
    }
    return regions;
}

int estimate_root_parallel_lanes(int move_count, int depth, bool is_exact) {
    if (move_count <= 1 || depth < 2) {
        return 1;
    }
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 8;
    }
    int workers = static_cast<int>(hw);
    if (!is_exact) {
        if (depth <= 4) {
            workers = std::max(1, workers / 2);
        } else if (depth <= 8) {
            workers = std::max(1, (workers * 3) / 4);
        }
    }
    return std::max(1, std::min(move_count, workers));
}

struct ExactCacheEntry {
    Bitboard p = 0;
    Bitboard o = 0;
    int turn = 1;
    int best_move = -1;
    double best_value = 0.0;
    double best_win_rate = 0.0;
    std::unordered_map<int, double> move_values;
    std::unordered_map<int, double> move_win_rates;
    bool is_resolved = false;
    int empty_count = 64;
    std::chrono::steady_clock::time_point timestamp;
};

class ExactResultCache {
public:
    std::optional<ExactCacheEntry> get(Bitboard p, Bitboard o, int turn, int empty_count) const {
        Bitboard hv = zobrist_hash(p, o) ^ static_cast<Bitboard>(turn);
        std::size_t idx = static_cast<std::size_t>(hv % CACHE_SIZE);
        std::shared_lock<std::shared_mutex> lock(mutexes_[idx]);
        const auto& entry = cache_[idx];
        if (entry.p == p && entry.o == o && entry.turn == turn && entry.is_resolved) {
            
            if (entry.empty_count == empty_count) {
                return entry;
            }
        }
        return std::nullopt;
    }

    void store(Bitboard p, Bitboard o, int turn, int best_move, double best_value, double best_win_rate,
               const std::unordered_map<int, double>& move_values,
               const std::unordered_map<int, double>& move_win_rates,
               int empty_count) {
        Bitboard hv = zobrist_hash(p, o) ^ static_cast<Bitboard>(turn);
        std::size_t idx = static_cast<std::size_t>(hv % CACHE_SIZE);
        std::unique_lock<std::shared_mutex> lock(mutexes_[idx]);
        auto& entry = cache_[idx];
        entry.p = p;
        entry.o = o;
        entry.turn = turn;
        entry.best_move = best_move;
        entry.best_value = best_value;
        entry.best_win_rate = best_win_rate;
        entry.move_values = move_values;
        entry.move_win_rates = move_win_rates;
        entry.is_resolved = true;
        entry.empty_count = empty_count;
        entry.timestamp = std::chrono::steady_clock::now();
    }

    void clear() {
        for (std::size_t i = 0; i < CACHE_SIZE; ++i) {
            std::unique_lock<std::shared_mutex> lock(mutexes_[i]);
            cache_[i] = ExactCacheEntry{};
        }
    }

private:
    static constexpr std::size_t CACHE_SIZE = 1u << 20; 
    std::array<ExactCacheEntry, CACHE_SIZE> cache_;
    mutable std::array<std::shared_mutex, CACHE_SIZE> mutexes_;
};

ExactResultCache& exact_cache() {
    static ExactResultCache cache;
    return cache;
}



double calculate_mcts_influence_ratio(int simulation_count, int empty_count, bool is_auto_mode) {
    if (simulation_count <= 0) return 0.0;
    
    double base_influence = 0.0;
    
    if (is_auto_mode) {
        if (empty_count >= 50) {
            base_influence = 0.32;
        } else if (empty_count >= 40) {
            base_influence = 0.40;
        } else if (empty_count >= 30) {
            base_influence = 0.48;
        } else if (empty_count >= 20) {
            base_influence = 0.55;
        } else {
            base_influence = 0.65;
        }
        
        if (simulation_count >= 100000) {
            base_influence = std::min(0.85, base_influence + 0.12);
        } else if (simulation_count >= 50000) {
            base_influence = std::min(0.85, base_influence + 0.08);
        } else if (simulation_count >= 20000) {
            base_influence = std::min(0.85, base_influence + 0.04);
        }
    } else {
        if (simulation_count >= 100000) {
            base_influence = 0.80;
        } else if (simulation_count >= 50000) {
            base_influence = 0.70;
        } else if (simulation_count >= 20000) {
            base_influence = 0.60;
        } else if (simulation_count >= 10000) {
            base_influence = 0.50;
        } else if (simulation_count >= 5000) {
            base_influence = 0.40;
        } else if (simulation_count >= 2000) {
            base_influence = 0.30;
        } else if (simulation_count >= 1000) {
            base_influence = 0.20;
        } else if (simulation_count >= 500) {
            base_influence = 0.10;
        } else {
            base_influence = 0.05;
        }
        
        if (empty_count <= 10) {
            base_influence *= 0.5;
        } else if (empty_count <= 16) {
            base_influence *= 0.7;
        }
    }
    
    return std::max(0.0, std::min(1.0, base_influence));
}

#if 0
struct MCTSStateKey {
    Bitboard p = 0;
    Bitboard o = 0;
    int turn = 1;

    bool operator==(const MCTSStateKey& other) const noexcept {
        return p == other.p && o == other.o && turn == other.turn;
    }
};

struct MCTSStateKeyHash {
    std::size_t operator()(const MCTSStateKey& key) const noexcept {
        const std::size_t h1 = std::hash<Bitboard>{}(key.p);
        const std::size_t h2 = std::hash<Bitboard>{}(key.o);
        const std::size_t h3 = std::hash<int>{}(key.turn);
        return h1 ^ (h2 + 0x9E3779B97F4A7C15ULL + (h1 << 6) + (h1 >> 2)) ^ (h3 << 1);
    }
};

struct MCTSActionKey {
    MCTSStateKey state{};
    int action = -1;

    bool operator==(const MCTSActionKey& other) const noexcept {
        return action == other.action && state == other.state;
    }
};

struct MCTSActionKeyHash {
    std::size_t operator()(const MCTSActionKey& key) const noexcept {
        const std::size_t base = MCTSStateKeyHash{}(key.state);
        const std::size_t action_hash = std::hash<int>{}(key.action);
        return base ^ (action_hash + 0x9E3779B97F4A7C15ULL + (base << 6) + (base >> 2));
    }
};

struct PendingMCTSPath {
    MCTSStateKey leaf{};
    std::vector<std::pair<MCTSStateKey, int>> path;
};

std::array<float, 64> build_valid_mask_array(Bitboard valid) {
    std::array<float, 64> mask{};
    while (valid) {
        Bitboard bit = lsb(valid);
        valid ^= bit;
        mask[static_cast<std::size_t>(bit_index(bit))] = 1.0f;
    }
    return mask;
}

class BatchMCTS {
public:
    explicit BatchMCTS(double c_puct = 2.0, double virtual_loss = 1.0)
        : c_puct_(c_puct), virtual_loss_(virtual_loss), rng_(std::random_device{}()) {}

    struct LeafBatch {
        std::vector<std::tuple<Bitboard, Bitboard, int>> leaves;
        std::vector<int> tickets;
        int simulation_count = 0;
    };

    struct BatchStepResult {
        int simulation_count = 0;
        int leaf_count = 0;
    };

    struct RootStatsResult {
        std::unordered_map<int, double> move_win_rates;
        std::unordered_map<int, int> root_visits;
        double best_wr = 50.0;
    };

    void initialize_root(Bitboard p, Bitboard o, int turn, const py::array_t<float, py::array::c_style | py::array::forcecast>& policy_logits, bool add_noise = false) {
        if (policy_logits.ndim() != 1 || policy_logits.shape(0) != 64) {
            throw std::invalid_argument("policy_logits must be a 64-element float array");
        }
        const auto logits = policy_logits.unchecked<1>();
        std::array<float, 64> policy{};
        for (py::ssize_t i = 0; i < 64; ++i) {
            policy[static_cast<std::size_t>(i)] = logits(i);
        }
        initialize_root_array(p, o, turn, policy, add_noise);
    }

    void initialize_root_array(Bitboard p, Bitboard o, int turn, const std::array<float, 64>& policy_logits, bool add_noise = false) {
        const MCTSStateKey state{p, o, turn};
        const auto valid_mask = build_valid_mask_array(get_legal_moves_optimized(p, o));
        visited_.insert(state);
        priors_[state] = normalize_priors(policy_logits, valid_mask, add_noise);
        state_visits_[state] = 0;
    }

    py::dict collect_leaves(
        Bitboard p,
        Bitboard o,
        int turn,
        int batch_size,
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& stop_flag
    ) {
        if (stop_flag.ndim() != 1 || stop_flag.shape(0) < 1) {
            throw std::invalid_argument("stop_flag must be a uint8 array with at least one element");
        }
        LeafBatch batch;
        {
            py::gil_scoped_release release;
            batch = collect_leaves_plain(p, o, turn, batch_size, stop_flag.data());
        }
        py::dict out;
        out["leaves"] = batch.leaves;
        out["tickets"] = batch.tickets;
        out["simulation_count"] = batch.simulation_count;
        return out;
    }

    void expand_leaves(
        const std::vector<int>& tickets,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& policy_batch,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& value_batch
    ) {
        if (policy_batch.ndim() != 2 || policy_batch.shape(1) != 64) {
            throw std::invalid_argument("policy_batch must be a float array of shape (n, 64)");
        }
        if (value_batch.ndim() != 1) {
            throw std::invalid_argument("value_batch must be a float array of shape (n,)");
        }
        if (policy_batch.shape(0) != static_cast<py::ssize_t>(tickets.size()) || value_batch.shape(0) != static_cast<py::ssize_t>(tickets.size())) {
            throw std::invalid_argument("ticket count, policy batch, and value batch must have the same length");
        }
        const auto policies = policy_batch.unchecked<2>();
        const auto values = value_batch.unchecked<1>();
        std::vector<std::array<float, 64>> policy_vec;
        std::vector<float> value_vec;
        policy_vec.reserve(tickets.size());
        value_vec.reserve(tickets.size());
        for (py::ssize_t i = 0; i < policy_batch.shape(0); ++i) {
            std::array<float, 64> policy{};
            for (py::ssize_t j = 0; j < 64; ++j) {
                policy[static_cast<std::size_t>(j)] = policies(i, j);
            }
            policy_vec.push_back(policy);
            value_vec.push_back(values(i));
        }
        expand_leaves_plain(tickets, policy_vec, value_vec);
    }

    RootStatsResult root_stats_plain(Bitboard p, Bitboard o, int turn) const {
        RootStatsResult out;
        const MCTSStateKey state{p, o, turn};
        Bitboard valid = get_legal_moves_optimized(p, o);
        int best_action = -1;
        int max_visits = -1;
        int total_root_visits = 0;
        Bitboard temp = valid;
        while (temp) {
            Bitboard bit = lsb(temp);
            temp ^= bit;
            const int action = bit_index(bit);
            const MCTSActionKey key{state, action};
            const int visits = lookup_visits(key);
            out.root_visits[action] = visits;
            total_root_visits += visits;
            if (visits > max_visits) {
                max_visits = visits;
                best_action = action;
            }
        }

        temp = valid;
        while (temp) {
            Bitboard bit = lsb(temp);
            temp ^= bit;
            const int action = bit_index(bit);
            const MCTSActionKey key{state, action};
            const int visits = lookup_visits(key);
            double move_wr = 50.0;
            if (visits > 0) {
                double avg_q = lookup_value(key) / static_cast<double>(visits);
                avg_q = std::max(-1.0, std::min(1.0, avg_q));
                const double q_wr = (avg_q + 1.0) * 50.0;
                const double visit_wr = total_root_visits > 0 ? (static_cast<double>(visits) / static_cast<double>(total_root_visits)) * 100.0 : 50.0;
                move_wr = 0.70 * q_wr + 0.30 * visit_wr;
            }
            out.move_win_rates[action] = move_wr;
        }
        if (best_action >= 0) {
            out.best_wr = out.move_win_rates[best_action];
        }
        return out;
    }

    py::dict root_stats(Bitboard p, Bitboard o, int turn) const {
        const RootStatsResult plain = root_stats_plain(p, o, turn);
        py::dict move_win_rates;
        py::dict root_visits;
        for (const auto& entry : plain.move_win_rates) {
            move_win_rates[py::int_(entry.first)] = entry.second;
        }
        for (const auto& entry : plain.root_visits) {
            root_visits[py::int_(entry.first)] = entry.second;
        }
        py::dict out;
        out["move_win_rates"] = move_win_rates;
        out["root_visits"] = root_visits;
        out["best_wr"] = plain.best_wr;
        return out;
    }

    BatchStepResult collect_and_expand_plain(
        Bitboard p,
        Bitboard o,
        int turn,
        int batch_size,
        const std::uint8_t* stop_ptr,
        const py::function& infer_batch
    ) {
        BatchStepResult out;
        LeafBatch batch = collect_leaves_plain(p, o, turn, batch_size, stop_ptr);
        out.simulation_count = batch.simulation_count;
        if (batch.tickets.empty() || batch.leaves.empty()) {
            return out;
        }
        std::vector<std::array<float, 64>> policy_vec;
        std::vector<float> value_vec;
        {
            py::gil_scoped_acquire acquire;
            py::list leaves_list;
            for (const auto& leaf : batch.leaves) {
                leaves_list.append(py::make_tuple(std::get<0>(leaf), std::get<1>(leaf), std::get<2>(leaf)));
            }
            py::tuple infer_tuple = infer_batch(leaves_list).cast<py::tuple>();
            if (infer_tuple.size() != 2) {
                throw std::runtime_error("infer_batch must return (policy_batch, value_batch)");
            }
            auto policy_batch = infer_tuple[0].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto value_batch = infer_tuple[1].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            if (policy_batch.ndim() != 2 || policy_batch.shape(1) != 64) {
                throw std::runtime_error("infer_batch policy_batch must have shape (n, 64)");
            }
            if (value_batch.ndim() != 1 || policy_batch.shape(0) != value_batch.shape(0) || policy_batch.shape(0) != static_cast<py::ssize_t>(batch.tickets.size())) {
                throw std::runtime_error("infer_batch value_batch must have shape (n,) matching tickets");
            }
            const auto policies = policy_batch.unchecked<2>();
            const auto values = value_batch.unchecked<1>();
            policy_vec.reserve(batch.tickets.size());
            value_vec.reserve(batch.tickets.size());
            for (py::ssize_t i = 0; i < policy_batch.shape(0); ++i) {
                std::array<float, 64> policy{};
                for (py::ssize_t j = 0; j < 64; ++j) {
                    policy[static_cast<std::size_t>(j)] = policies(i, j);
                }
                policy_vec.push_back(policy);
                value_vec.push_back(values(i));
            }
        }
        expand_leaves_plain(batch.tickets, policy_vec, value_vec);
        out.leaf_count = static_cast<int>(batch.tickets.size());
        return out;
    }

    py::dict collect_and_expand(
        Bitboard p,
        Bitboard o,
        int turn,
        int batch_size,
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& stop_flag,
        py::function infer_batch
    ) {
        py::dict out;
        BatchStepResult result = collect_and_expand_plain(p, o, turn, batch_size, stop_flag.data(), infer_batch);
        out["simulation_count"] = result.simulation_count;
        out["leaf_count"] = result.leaf_count;
        return out;
    }

private:
    struct TraverseResult {
        bool is_terminal = false;
        MCTSStateKey leaf{};
        std::vector<std::pair<MCTSStateKey, int>> path;
        double terminal_value = 0.0;
    };

    LeafBatch collect_leaves_plain(Bitboard p, Bitboard o, int turn, int batch_size, const std::uint8_t* stop_ptr) {
        LeafBatch batch;
        batch.leaves.reserve(static_cast<std::size_t>(std::max(batch_size, 0)));
        batch.tickets.reserve(static_cast<std::size_t>(std::max(batch_size, 0)));
        for (int i = 0; i < batch_size; ++i) {
            if (stop_ptr != nullptr && stop_ptr[0] != 0) {
                break;
            }
            auto result = traverse(p, o, turn);
            ++batch.simulation_count;
            if (result.is_terminal) {
                if (!result.path.empty()) {
                    backpropagate_terminal(result.path, result.terminal_value);
                }
                continue;
            }
            apply_virtual_loss(result.path);
            const int ticket = next_ticket_++;
            pending_paths_[ticket] = PendingMCTSPath{result.leaf, std::move(result.path)};
            batch.leaves.emplace_back(result.leaf.p, result.leaf.o, result.leaf.turn);
            batch.tickets.push_back(ticket);
        }
        return batch;
    }

    void expand_leaves_plain(const std::vector<int>& tickets, const std::vector<std::array<float, 64>>& policy_batch, const std::vector<float>& value_batch) {
        if (tickets.size() != policy_batch.size() || tickets.size() != value_batch.size()) {
            throw std::invalid_argument("ticket count, policy batch, and value batch must have the same length");
        }
        for (std::size_t i = 0; i < tickets.size(); ++i) {
            const int ticket = tickets[i];
            auto pending_it = pending_paths_.find(ticket);
            if (pending_it == pending_paths_.end()) {
                continue;
            }
            expand_leaf(pending_it->second.leaf, pending_it->second.path, policy_batch[i], value_batch[i]);
            pending_paths_.erase(pending_it);
        }
    }

    std::array<float, 64> normalize_priors(const std::array<float, 64>& policy_logits, const std::array<float, 64>& valid_mask, bool add_noise) {
        std::array<float, 64> priors{};
        float sum = 0.0f;
        int valid_count = 0;
        for (std::size_t i = 0; i < 64; ++i) {
            priors[i] = policy_logits[i] * valid_mask[i];
            sum += priors[i];
            if (valid_mask[i] > 0.0f) {
                ++valid_count;
            }
        }
        if (sum > 0.0f) {
            for (float& v : priors) {
                v /= sum;
            }
            if (add_noise && valid_count > 0) {
                std::gamma_distribution<float> gamma(0.3f, 1.0f);
                std::array<float, 64> noise{};
                float noise_sum = 0.0f;
                for (std::size_t i = 0; i < 64; ++i) {
                    if (valid_mask[i] > 0.0f) {
                        noise[i] = gamma(rng_);
                        noise_sum += noise[i];
                    }
                }
                if (noise_sum > 0.0f) {
                    for (std::size_t i = 0; i < 64; ++i) {
                        if (valid_mask[i] > 0.0f) {
                            priors[i] = 0.75f * priors[i] + 0.25f * (noise[i] / noise_sum);
                        }
                    }
                }
            }
            return priors;
        }
        if (valid_count > 0) {
            const float uniform = 1.0f / static_cast<float>(valid_count);
            for (std::size_t i = 0; i < 64; ++i) {
                priors[i] = valid_mask[i] > 0.0f ? uniform : 0.0f;
            }
        }
        return priors;
    }

    TraverseResult traverse(Bitboard p, Bitboard o, int turn) const {
        TraverseResult result;
        Bitboard current_p = p;
        Bitboard current_o = o;
        int current_turn = turn;
        while (true) {
            const MCTSStateKey state{current_p, current_o, current_turn};
            const Bitboard valid = get_legal_moves_optimized(current_p, current_o);
            if (valid == 0) {
                if (get_legal_moves_optimized(current_o, current_p) == 0) {
                    const int own_count = count_bits(current_p);
                    const int opp_count = count_bits(current_o);
                    result.is_terminal = true;
                    result.terminal_value = own_count > opp_count ? 1.0 : (own_count < opp_count ? -1.0 : 0.0);
                    return result;
                }
                result.path.emplace_back(state, -1);
                current_p = current_o;
                current_o = state.p;
                current_turn = -current_turn;
                continue;
            }
            if (visited_.find(state) == visited_.end()) {
                result.leaf = state;
                return result;
            }
            const auto prior_it = priors_.find(state);
            if (prior_it == priors_.end()) {
                result.leaf = state;
                return result;
            }
            const double sqrt_ns = c_puct_ * std::sqrt(static_cast<double>(std::max(1, lookup_state_visits(state))));
            int best_action = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            Bitboard temp = valid;
            while (temp) {
                Bitboard bit = lsb(temp);
                temp ^= bit;
                const int action = bit_index(bit);
                const MCTSActionKey action_key{state, action};
                const int visits = lookup_visits(action_key);
                const double q = visits > 0 ? lookup_value(action_key) / static_cast<double>(visits) : 0.0;
                const double u = q + sqrt_ns * static_cast<double>(prior_it->second[static_cast<std::size_t>(action)]) / static_cast<double>(1 + visits);
                
                _mm_prefetch(reinterpret_cast<const char*>(&edge_visits_), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&edge_values_), _MM_HINT_T0);
                if (u > best_score) {
                    best_score = u;
                    best_action = action;
                }
            }
            result.path.emplace_back(state, best_action);
            const Bitboard flips = get_flip_optimized(current_p, current_o, best_action);
            const Bitboard next_p = (current_p | (1ULL << best_action) | flips) & FULL_MASK;
            const Bitboard next_o = (current_o ^ flips) & FULL_MASK;
            current_p = next_o;
            current_o = next_p;
            current_turn = -current_turn;
        }
    }

    void apply_virtual_loss(const std::vector<std::pair<MCTSStateKey, int>>& path) {
        for (const auto& entry : path) {
            if (entry.second == -1) {
                continue;
            }
            const MCTSActionKey action_key{entry.first, entry.second};
            ++edge_visits_[action_key];
            ++state_visits_[entry.first];
            edge_values_[action_key] += -virtual_loss_;
        }
    }

    void backpropagate_terminal(const std::vector<std::pair<MCTSStateKey, int>>& path, double value) {
        double v = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;
            if (it->second == -1) {
                continue;
            }
            const MCTSActionKey action_key{it->first, it->second};
            ++edge_visits_[action_key];
            ++state_visits_[it->first];
            edge_values_[action_key] += v;
        }
    }

    void expand_leaf(const MCTSStateKey& leaf, const std::vector<std::pair<MCTSStateKey, int>>& path, const std::array<float, 64>& policy_logits, float value) {
        visited_.insert(leaf);
        const auto valid_mask = build_valid_mask_array(get_legal_moves_optimized(leaf.p, leaf.o));
        priors_[leaf] = normalize_priors(policy_logits, valid_mask, false);
        state_visits_.emplace(leaf, 0);
        double v = std::max(-1.0, std::min(1.0, static_cast<double>(value)));
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;
            if (it->second == -1) {
                continue;
            }
            const MCTSActionKey action_key{it->first, it->second};
            edge_values_[action_key] += virtual_loss_ + v;
        }
    }

    int lookup_visits(const MCTSActionKey& key) const {
        const auto it = edge_visits_.find(key);
        return it == edge_visits_.end() ? 0 : it->second;
    }

    int lookup_state_visits(const MCTSStateKey& key) const {
        const auto it = state_visits_.find(key);
        return it == state_visits_.end() ? 0 : it->second;
    }

    double lookup_value(const MCTSActionKey& key) const {
        const auto it = edge_values_.find(key);
        return it == edge_values_.end() ? 0.0 : it->second;
    }

    double c_puct_ = 2.0;
    double virtual_loss_ = 1.0;
    mutable std::mt19937 rng_;
    std::unordered_map<MCTSActionKey, double, MCTSActionKeyHash> edge_values_;
    std::unordered_map<MCTSActionKey, int, MCTSActionKeyHash> edge_visits_;
    std::unordered_map<MCTSStateKey, int, MCTSStateKeyHash> state_visits_;
    std::unordered_map<MCTSStateKey, std::array<float, 64>, MCTSStateKeyHash> priors_;
    std::unordered_set<MCTSStateKey, MCTSStateKeyHash> visited_;
    std::unordered_map<int, PendingMCTSPath> pending_paths_;
    int next_ticket_ = 1;
};

struct SearchSessionABResult {
    int completed_depth = 2;
    int attempted_depth = 2;
    bool resolved = false;
    bool timed_out = false;
    std::int64_t total_nodes = 0;
    std::vector<int> moves;
    std::vector<double> values;
    std::vector<double> win_rates;
};

struct SearchSessionMCTSResult {
    std::unordered_map<int, double> move_win_rates;
    std::unordered_map<int, int> root_visits;
    double best_wr = 50.0;
    int simulation_count = 0;
    int nn_batch_count = 0;
    int nn_leaf_count = 0;
};

struct SearchSessionResult {
    SearchSessionABResult ab;
    SearchSessionMCTSResult mcts;
};

class SearchSession {
public:
    explicit SearchSession(double c_puct = 2.0, double virtual_loss = 1.0)
        : mcts_(c_puct, virtual_loss) {}

    SearchSessionResult run(
        Bitboard p,
        Bitboard o,
        int turn,
        int mvs,
        int start_depth,
        bool is_exact,
        const std::vector<int>& ordered_indices,
        const std::vector<double>& root_policy,
        bool use_ab,
        bool use_mcts,
        double time_limit_sec,
        int mcts_batch_size,
        double ab_delay_sec,
        int ab_time_limit_ms,
        int max_depth,
        bool add_root_noise,
        std::uint8_t* stop_ptr,
        const py::function& infer_batch,
        const py::function& ab_progress,
        bool multi_cut_enabled = false,
        int multi_cut_threshold = 3,
        int multi_cut_depth = 8
    ) {
        SearchSessionResult result;
        const auto start_time = std::chrono::steady_clock::now();
        const auto overall_deadline = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(std::max(0.0, time_limit_sec)));
        const std::vector<int> initial_order = ordered_indices.empty() ? legal_move_indices(p, o) : ordered_indices;
        std::array<float, 64> root_policy_arr = make_root_policy(root_policy, p, o, turn, infer_batch);
        bool have_root_policy = false;
        for (float value : root_policy_arr) {
            if (value != 0.0f) {
                have_root_policy = true;
                break;
            }
        }
        if (use_mcts) {
            mcts_.initialize_root_array(p, o, turn, root_policy_arr, add_root_noise);
        }

        std::thread mcts_thread;
        std::thread ab_thread;

        if (use_mcts) {
            mcts_thread = std::thread([&, this]() {
                result.mcts.nn_batch_count = 1;
                result.mcts.nn_leaf_count = 1;
                while (true) {
                    if (stop_ptr != nullptr && stop_ptr[0] != 0) {
                        break;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= overall_deadline) {
                        break;
                    }
                    const double remain = std::chrono::duration<double>(overall_deadline - now).count();
                    if (remain <= 0.0) {
                        break;
                    }
                    const int curr_batch = current_mcts_batch_size(mcts_batch_size, remain);
                    BatchMCTS::BatchStepResult step = mcts_.collect_and_expand_plain(p, o, turn, curr_batch, stop_ptr, infer_batch);
                    result.mcts.simulation_count += step.simulation_count;
                    if (step.leaf_count <= 0) {
                        continue;
                    }
                    result.mcts.nn_batch_count += 1;
                    result.mcts.nn_leaf_count += step.leaf_count;
                }
                const BatchMCTS::RootStatsResult stats = mcts_.root_stats_plain(p, o, turn);
                result.mcts.move_win_rates = stats.move_win_rates;
                result.mcts.root_visits = stats.root_visits;
                result.mcts.best_wr = stats.best_wr;
            });
        }

        if (use_ab) {
            ab_thread = std::thread([&, this]() {
                SearchSessionABResult ab_result;
                ab_result.completed_depth = std::max(2, start_depth - 1);
                ab_result.attempted_depth = ab_result.completed_depth;
                if (ab_delay_sec > 0.0) {
                    const auto delay_deadline = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(ab_delay_sec));
                    while (std::chrono::steady_clock::now() < delay_deadline) {
                        if (stop_ptr != nullptr && stop_ptr[0] != 0) {
                            result.ab = ab_result;
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                std::vector<int> curr_ordered = initial_order;
                const auto ab_deadline = ab_time_limit_ms > 0 ? (start_time + std::chrono::milliseconds(ab_time_limit_ms)) : overall_deadline;
                for (int depth = std::max(start_depth, 2); depth <= max_depth; ++depth) {
                    if (stop_ptr != nullptr && stop_ptr[0] != 0) {
                        break;
                    }
                    ab_result.attempted_depth = depth;
                    SearchContext ctx;
                    ctx.weights = &require_global_weights();
                    ctx.order_map = &require_global_order_map();
                    ctx.multi_cut_enabled = multi_cut_enabled;
                    ctx.multi_cut_threshold = multi_cut_threshold;
                    ctx.multi_cut_depth = multi_cut_depth;
                    if (have_root_policy) {
                        root_policy_buffer_.assign(root_policy_arr.begin(), root_policy_arr.end());
                        ctx.root_policy = &root_policy_buffer_;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    const auto hard_deadline = ab_deadline < overall_deadline ? ab_deadline : overall_deadline;
                    if (now >= hard_deadline) {
                        break;
                    }
                    ctx.use_deadline = true;
                    ctx.deadline = hard_deadline;
                    std::pair<std::vector<double>, std::vector<std::int64_t>> search_result = search_root_parallel_impl(p, o, mvs, depth, is_exact, curr_ordered, ctx);
                    if (ctx.timed_out) {
                        ab_result.timed_out = true;
                        break;
                    }
                    if (curr_ordered.empty()) {
                        break;
                    }
                    struct LayerResult {
                        int move;
                        double value;
                        double win_rate;
                    };
                    std::vector<LayerResult> combined;
                    combined.reserve(curr_ordered.size());
                    std::int64_t layer_nodes = 0;
                    for (std::size_t i = 0; i < curr_ordered.size(); ++i) {
                        const double win_rate = calculate_win_rate(search_result.first[i], is_exact);
                        combined.push_back(LayerResult{curr_ordered[i], search_result.first[i], win_rate});
                        layer_nodes += search_result.second[i];
                    }
                    std::sort(combined.begin(), combined.end(), [](const LayerResult& a, const LayerResult& b) {
                        return a.win_rate > b.win_rate;
                    });
                    curr_ordered.clear();
                    ab_result.moves.clear();
                    ab_result.values.clear();
                    ab_result.win_rates.clear();
                    for (const auto& entry : combined) {
                        curr_ordered.push_back(entry.move);
                        ab_result.moves.push_back(entry.move);
                        ab_result.values.push_back(entry.value);
                        ab_result.win_rates.push_back(entry.win_rate);
                    }
                    ab_result.total_nodes += layer_nodes;
                    ab_result.completed_depth = depth;
                    if (is_exact) {
                        ab_result.resolved = true;
                        emit_ab_progress(ab_progress, depth, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count(), layer_nodes, ab_result);
                        if (!use_mcts && stop_ptr != nullptr) {
                            stop_ptr[0] = 1;
                        }
                        break;
                    }
                    emit_ab_progress(ab_progress, depth, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count(), layer_nodes, ab_result);
                }
                result.ab = ab_result;
            });
        }

        if (mcts_thread.joinable()) {
            mcts_thread.join();
        }
        if (ab_thread.joinable()) {
            ab_thread.join();
        }
        return result;
    }

private:
    static int current_mcts_batch_size(int batch_size, double remain) {
        if (batch_size <= 0) {
            return 1;
        }
        if (remain >= 10.0) {
            return std::min(batch_size, 1048576);
        }
        if (remain >= 5.0) {
            return std::min(batch_size, 786432);
        }
        if (remain >= 2.0) {
            return std::min(batch_size, 524288);
        }
        if (remain >= 1.0) {
            return std::min(batch_size, 262144);
        }
        if (remain >= 0.4) {
            return std::min(batch_size, 131072);
        }
        return std::min(batch_size, 65536);
    }

    static std::array<float, 64> make_root_policy(const std::vector<double>& root_policy, Bitboard p, Bitboard o, int turn, const py::function& infer_batch) {
        std::array<float, 64> out{};
        bool has_policy = false;
        if (root_policy.size() == 64) {
            for (std::size_t i = 0; i < 64; ++i) {
                out[i] = static_cast<float>(root_policy[i]);
                if (out[i] != 0.0f) {
                    has_policy = true;
                }
            }
        }
        if (has_policy || infer_batch.is_none()) {
            return out;
        }
        py::gil_scoped_acquire acquire;
        py::list leaves;
        leaves.append(py::make_tuple(p, o, turn));
        py::tuple infer_tuple = infer_batch(leaves).cast<py::tuple>();
        if (infer_tuple.size() != 2) {
            throw std::runtime_error("infer_batch must return (policy_batch, value_batch)");
        }
        auto policy_batch = infer_tuple[0].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        if (policy_batch.ndim() != 2 || policy_batch.shape(0) < 1 || policy_batch.shape(1) != 64) {
            throw std::runtime_error("infer_batch policy_batch must have shape (n, 64)");
        }
        const auto policies = policy_batch.unchecked<2>();
        for (py::ssize_t j = 0; j < 64; ++j) {
            out[static_cast<std::size_t>(j)] = policies(0, j);
        }
        return out;
    }

    static void emit_ab_progress(const py::function& ab_progress, int depth, double elapsed_sec, std::int64_t layer_nodes, const SearchSessionABResult& ab_result) {
        if (ab_progress.is_none() || ab_result.moves.empty() || ab_result.win_rates.empty()) {
            return;
        }
        try {
            py::gil_scoped_acquire acquire;
            py::dict payload;
            payload["depth"] = depth;
            payload["elapsed_sec"] = elapsed_sec;
            payload["nodes"] = layer_nodes;
            payload["moves"] = ab_result.moves;
            payload["values"] = ab_result.values;
            payload["win_rates"] = ab_result.win_rates;
            payload["best_wr"] = ab_result.win_rates.front();
            payload["resolved"] = ab_result.resolved;
            ab_progress(payload);
        } catch (const py::error_already_set&) {
        }
    }

    BatchMCTS mcts_;
    std::vector<double> root_policy_buffer_;
};

#endif

std::vector<double> evaluate_moves(Bitboard p, Bitboard o, int mvs, const std::vector<int>& indices, const std::vector<double>& weights) {
    std::vector<double> out;
    out.reserve(indices.size());
    for (int idx : indices) {
        Bitboard bit = 1ULL << idx;
        Bitboard f = get_flip_optimized(p, o, idx);
        Bitboard np = (p | bit | f) & FULL_MASK;
        Bitboard no = (o ^ f) & FULL_MASK;
        out.push_back(evaluate_board_full(np, no, mvs + 1, weights));
    }
    return out;
}

std::pair<Bitboard, Bitboard> apply_move(Bitboard p, Bitboard o, int idx) {
    Bitboard bit = 1ULL << idx;
    Bitboard f = get_flip_optimized(p, o, idx);
    Bitboard np = (p | bit | f) & FULL_MASK;
    Bitboard no = (o ^ f) & FULL_MASK;
    return {np, no};
}

struct MoveAnalysisResult {
    std::vector<int> moves;
    std::vector<double> evals;
    std::vector<Bitboard> next_p;
    std::vector<Bitboard> next_o;
};

MoveAnalysisResult analyze_legal_moves_cached(Bitboard p, Bitboard o, int mvs) {
    std::vector<int> indices = legal_move_indices(p, o);
    std::vector<double> evals = evaluate_moves(p, o, mvs, indices, require_global_weights());
    std::vector<Bitboard> next_p;
    std::vector<Bitboard> next_o;
    next_p.reserve(indices.size());
    next_o.reserve(indices.size());
    for (int idx : indices) {
        auto [np, no] = apply_move(p, o, idx);
        next_p.push_back(np);
        next_o.push_back(no);
    }
    return MoveAnalysisResult{std::move(indices), std::move(evals), std::move(next_p), std::move(next_o)};
}

void set_eval_data(const py::array_t<double, py::array::c_style | py::array::forcecast>& weights,
                   const py::array_t<double, py::array::c_style | py::array::forcecast>& order_map) {
    auto w = weights.unchecked<1>();
    auto om = order_map.unchecked<1>();
    global_weights().resize(static_cast<std::size_t>(w.shape(0)));
    for (py::ssize_t i = 0; i < w.shape(0); ++i) global_weights()[static_cast<std::size_t>(i)] = w(i);
    global_order_map().resize(static_cast<std::size_t>(om.shape(0)));
    for (py::ssize_t i = 0; i < om.shape(0); ++i) global_order_map()[static_cast<std::size_t>(i)] = om(i);
}

const std::vector<double>& require_global_weights() {
    if (global_weights().empty()) throw std::runtime_error("global weights are not set");
    return global_weights();
}

const std::vector<double>& require_global_order_map() {
    if (global_order_map().empty()) throw std::runtime_error("global order_map is not set");
    return global_order_map();
}

struct IterativeSearchResult {
    int completed_depth = 0;
    int best_move = -1;
    std::vector<int> moves;
    std::vector<double> values;
    std::vector<double> win_rates;
    std::int64_t nodes = 0;
    bool resolved = false;
    bool timed_out = false;
};

IterativeSearchResult get_best_move_ab_impl(Bitboard p, Bitboard o, int mvs, int max_depth, bool is_exact, std::vector<int> ordered_indices, const std::vector<double>& weights, const std::vector<double>& order_map, int time_limit_ms) {
    if (order_map.size() != 64) throw std::invalid_argument("order_map size must be 64");
    if (ordered_indices.empty()) {
        ordered_indices = legal_move_indices(p, o);
        std::sort(ordered_indices.begin(), ordered_indices.end(), [&](int a, int b) {
            return order_map[static_cast<std::size_t>(a)] > order_map[static_cast<std::size_t>(b)];
        });
    }
    SearchContext ctx;
    ctx.weights = &weights;
    ctx.order_map = &order_map;
    ctx.multi_cut_enabled = g_search_settings.multi_cut_enabled;
    ctx.multi_cut_threshold = g_search_settings.multi_cut_threshold;
    ctx.multi_cut_depth = g_search_settings.multi_cut_depth;
    ctx.ybwc_min_depth = g_search_settings.ybwc_min_depth;

    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    std::vector<int> curr_ordered = ordered_indices;
    std::vector<int> best_moves;
    std::vector<double> best_vals;
    std::vector<double> best_wrs;
    int completed_depth = 0;
    std::int64_t total_nodes = 0;
    bool resolved = false;
    int empty = 64 - count_bits(p | o);
    for (int depth = 2; depth <= max_depth; ++depth) {
        bool depth_exact = is_exact && depth >= empty;
        auto [vals, nodes] = search_root_parallel_impl(p, o, mvs, depth, depth_exact, curr_ordered, ctx);
        if (ctx.timed_out) break;
        struct Result {
            int move;
            double val;
            double wr;
            std::int64_t nodes;
        };
        std::vector<Result> results;
        results.reserve(curr_ordered.size());
        std::int64_t layer_nodes = 0;
        for (std::size_t i = 0; i < curr_ordered.size(); ++i) {
            double wr = calculate_win_rate(vals[i], depth_exact);
            results.push_back({curr_ordered[i], vals[i], wr, nodes[i]});
            layer_nodes += nodes[i];
        }
        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            return a.wr > b.wr;
        });
        curr_ordered.clear();
        best_moves.clear();
        best_vals.clear();
        best_wrs.clear();
        for (const auto& r : results) {
            curr_ordered.push_back(r.move);
            best_moves.push_back(r.move);
            best_vals.push_back(r.val);
            best_wrs.push_back(r.wr);
        }
        total_nodes += layer_nodes;
        completed_depth = depth;
        if (depth_exact) {
            resolved = true;
            break;
        }
    }
    IterativeSearchResult out;
    out.completed_depth = completed_depth;
    out.best_move = best_moves.empty() ? -1 : best_moves.front();
    out.moves = std::move(best_moves);
    out.values = std::move(best_vals);
    out.win_rates = std::move(best_wrs);
    out.nodes = total_nodes;
    out.resolved = resolved;
    out.timed_out = ctx.timed_out;
    return out;
}

bool should_use_early_exact(Bitboard p, Bitboard o, int empties, int base_threshold) {
    if (empties <= base_threshold) return true;
    if (empties > base_threshold + 7) return false;
    const int own_moves = count_bits(get_legal_moves_optimized(p, o));
    const int opp_moves = count_bits(get_legal_moves_optimized(o, p));
    int total_moves = own_moves + opp_moves;
    const int min_moves = std::min(own_moves, opp_moves);
    const Bitboard occ = p | o;
    const Bitboard empty_mask = (~occ) & FULL_MASK;
    const int frontier = count_bits(neighbor_union(occ) & empty_mask);
    int stable_hint = 0;
    if (empties <= 34) {
        const Bitboard sp = compute_strict_stable(p, o);
        const Bitboard so = compute_strict_stable(o, p);
        stable_hint = std::abs(count_bits(sp & p) - count_bits(so & o));
    }
    int small_regions = 0;
    const int region_count = count_empty_regions(empty_mask, &small_regions);
    const bool pass_pressure = min_moves <= 2;
    const bool low_branch = total_moves <= 8;
    const bool narrow_frontier = frontier <= 14;
    int exact_score = 0;
    if (low_branch) exact_score += 2;
    if (total_moves <= 6) exact_score += 1;
    if (pass_pressure) exact_score += 2;
    if (min_moves <= 1) exact_score += 1;
    if (narrow_frontier) exact_score += 1;
    if (frontier <= 10) exact_score += 1;
    if (stable_hint >= 2) exact_score += 1;
    if (stable_hint >= 4) exact_score += 1;
    if (region_count <= 3) exact_score += 1;
    if (small_regions > 0) exact_score += 1;
    if (empties <= base_threshold + 3) exact_score += 1;
    if (empties <= base_threshold + 5) exact_score += 1;
    return exact_score >= (empties <= base_threshold + 3 ? 4 : empties <= base_threshold + 5 ? 5 : 6);
}

void clear_tt() {
    auto& table = tt_table();
    TTEntry empty_entry{};
    for (auto& entry : table) {
        entry.hash.store(0);
        entry.depth.store(-1);
        entry.value.store(0.0);
        entry.flag.store(0);
        entry.best_move.store(-1);
        entry.age.store(0);
        entry.generation.store(0);
    }
    tt_generation().store(1);
}


std::vector<double> benchmark_optimizations(int iterations = 1000) {
    std::vector<double> results(4);
    auto start = std::chrono::high_resolution_clock::now();
    
    
    Bitboard test_p = 0x0000000810000000ULL;
    Bitboard test_o = 0x0000001008000000ULL;
    for (int i = 0; i < iterations; ++i) {
        store_tt(zobrist_hash(test_p, test_o), 10, 100.0, 1, 19, false);
        double value;
        int move;
        probe_tt(zobrist_hash(test_p, test_o), 8, -1000.0, 1000.0, move, value, false, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    results[0] = std::chrono::duration<double>(end - start).count();
    
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations * 100; ++i) {
        Bitboard legal = get_legal_moves_optimized(test_p, test_o);
        Bitboard flip = get_flip_optimized(test_p, test_o, 19);
        int bits = count_bits(legal);
    }
    end = std::chrono::high_resolution_clock::now();
    results[1] = std::chrono::duration<double>(end - start).count();
    
    
    std::vector<double> weights = make_board_perfect_seed();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations * 10; ++i) {
        double score = evaluate_board_full(test_p, test_o, 4, weights);
    }
    end = std::chrono::high_resolution_clock::now();
    results[2] = std::chrono::duration<double>(end - start).count();
    
    
#if defined(__AVX2__)
    if (iterations >= 4) {
        Bitboard p_boards[4] = {test_p, test_p, test_p, test_p};
        Bitboard o_boards[4] = {test_o, test_o, test_o, test_o};
        int mvs_array[4] = {4, 4, 4, 4};
        double scores[4];
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations / 4; ++i) {
            evaluate_board_full_simd_batch(p_boards, o_boards, mvs_array, weights, scores, 4);
        }
        end = std::chrono::high_resolution_clock::now();
        results[3] = std::chrono::duration<double>(end - start).count();
    } else {
        results[3] = 0.0;
    }
#else
    results[3] = -1.0; 
#endif
    
    return results;
}

}

void bind_engine_free_functions(py::module_& m);
void bind_engine_session_classes(py::module_& m);

Bitboard engine_get_legal_moves(Bitboard p, Bitboard o) {
    return get_legal_moves_optimized(p, o);
}

Bitboard engine_get_flip(Bitboard p, Bitboard o, int idx) {
    return get_flip_optimized(p, o, idx);
}

int engine_count_bits(Bitboard x) {
    return count_bits(x);
}

int engine_bit_index(Bitboard x) {
    return bit_index(x);
}

Bitboard engine_lsb(Bitboard x) {
    return lsb(x);
}

std::vector<int> engine_legal_move_indices(Bitboard p, Bitboard o) {
    return legal_move_indices(p, o);
}

const std::vector<double>& engine_require_global_weights() {
    return require_global_weights();
}

const std::vector<double>& engine_require_global_order_map() {
    return require_global_order_map();
}

void engine_clear_tt() {
    clear_tt();
}

double engine_calculate_mcts_influence_ratio(int simulation_count, int empty_count, bool is_auto_mode) {
    return calculate_mcts_influence_ratio(simulation_count, empty_count, is_auto_mode);
}

void engine_set_eval_data(const py::array_t<double, py::array::c_style | py::array::forcecast>& weights,
                          const py::array_t<double, py::array::c_style | py::array::forcecast>& order_map) {
    set_eval_data(weights, order_map);
}

std::vector<double> engine_benchmark_optimizations(int iterations) {
    return benchmark_optimizations(iterations);
}

std::pair<Bitboard, Bitboard> engine_apply_move(Bitboard p, Bitboard o, int idx) {
    return apply_move(p, o, idx);
}

bool engine_should_use_early_exact(Bitboard p, Bitboard o, int empties, int base_threshold) {
    return should_use_early_exact(p, o, empties, base_threshold);
}

double engine_calculate_win_rate(double ev, bool is_exact) {
    return calculate_win_rate(ev, is_exact);
}

void engine_set_search_params(const py::dict& params) {
    if (params.contains("pruning_enabled")) g_search_settings.pruning_enabled = params["pruning_enabled"].cast<bool>();
    if (params.contains("traditional_pruning_enabled")) g_search_settings.traditional_pruning_enabled = params["traditional_pruning_enabled"].cast<bool>();
    if (params.contains("multi_cut_enabled")) g_search_settings.multi_cut_enabled = params["multi_cut_enabled"].cast<bool>();
    if (params.contains("multi_cut_threshold")) g_search_settings.multi_cut_threshold = params["multi_cut_threshold"].cast<int>();
    if (params.contains("multi_cut_depth")) g_search_settings.multi_cut_depth = params["multi_cut_depth"].cast<int>();
    if (params.contains("ybwc_min_depth")) g_search_settings.ybwc_min_depth = params["ybwc_min_depth"].cast<int>();
}

std::tuple<std::vector<double>, std::vector<std::int64_t>, bool> engine_search_root_parallel_layer(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    const std::vector<double>* root_policy,
    int time_limit_ms
) {
    SearchContext ctx;
    ctx.weights = &require_global_weights();
    ctx.order_map = &require_global_order_map();
    ctx.root_policy = root_policy;
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    auto result = search_root_parallel_impl(p, o, mvs, depth, is_exact, ordered_indices, ctx);
    return std::make_tuple(std::move(result.first), std::move(result.second), ctx.timed_out);
}

py::dict probe_exact_cache_py(Bitboard p, Bitboard o, int turn, int empty_count) {
    py::dict out;
    auto entry_opt = exact_cache().get(p, o, turn, empty_count);
    if (entry_opt.has_value()) {
        const auto& entry = entry_opt.value();
        out["found"] = true;
        out["best_move"] = entry.best_move;
        out["best_value"] = entry.best_value;
        out["best_win_rate"] = entry.best_win_rate;
        out["is_resolved"] = entry.is_resolved;
        py::dict move_values;
        py::dict move_win_rates;
        for (const auto& [move, value] : entry.move_values) {
            move_values[py::int_(move)] = value;
        }
        for (const auto& [move, wr] : entry.move_win_rates) {
            move_win_rates[py::int_(move)] = wr;
        }
        out["move_values"] = move_values;
        out["move_win_rates"] = move_win_rates;
    } else {
        out["found"] = false;
        out["best_move"] = -1;
        out["best_value"] = 0.0;
        out["best_win_rate"] = 50.0;
        out["is_resolved"] = false;
        out["move_values"] = py::dict();
        out["move_win_rates"] = py::dict();
    }
    return out;
}

void clear_exact_cache_py() {
    exact_cache().clear();
}

void store_exact_cache_py(
    Bitboard p,
    Bitboard o,
    int turn,
    int best_move,
    double best_value,
    double best_win_rate,
    const std::unordered_map<int, double>& move_values,
    const std::unordered_map<int, double>& move_win_rates,
    int empty_count
) {
    exact_cache().store(p, o, turn, best_move, best_value, best_win_rate, move_values, move_win_rates, empty_count);
}

py::dict probe_tt_py(Bitboard p, Bitboard o) {
    py::dict out;
    if (const TTEntry* entry = probe_tt_entry(p, o)) {
        out["found"] = true;
        out["depth"] = entry->depth.load();
        out["value"] = entry->value.load();
        out["flag"] = entry->flag.load();
        out["best_move"] = entry->best_move.load();
    } else {
        out["found"] = false;
        out["depth"] = -1;
        out["value"] = 0.0;
        out["flag"] = 0;
        out["best_move"] = -1;
    }
    return out;
}

Bitboard get_legal_moves_py(Bitboard p, Bitboard o) {
    return get_legal_moves_optimized(p, o);
}

Bitboard get_flip_py(Bitboard p, Bitboard o, int idx) {
    return get_flip_optimized(p, o, idx);
}

double evaluate_board_full_py(Bitboard p, Bitboard o, int mvs, const std::vector<double>& weights) {
    py::gil_scoped_release release;
    return evaluate_board_full(p, o, mvs, weights);
}

std::vector<double> evaluate_moves_py(Bitboard p, Bitboard o, int mvs, const std::vector<int>& indices, const std::vector<double>& weights) {
    py::gil_scoped_release release;
    return evaluate_moves(p, o, mvs, indices, weights);
}

double evaluate_board_cached_py(Bitboard p, Bitboard o, int mvs) {
    py::gil_scoped_release release;
    return evaluate_board_full(p, o, mvs, require_global_weights());
}

std::vector<double> evaluate_moves_cached_py(Bitboard p, Bitboard o, int mvs, const std::vector<int>& indices) {
    py::gil_scoped_release release;
    return evaluate_moves(p, o, mvs, indices, require_global_weights());
}

py::dict analyze_legal_moves_cached_py(Bitboard p, Bitboard o, int mvs) {
    MoveAnalysisResult result;
    {
        py::gil_scoped_release release;
        result = analyze_legal_moves_cached(p, o, mvs);
    }
    py::dict out;
    out["moves"] = result.moves;
    out["evals"] = result.evals;
    out["next_p"] = result.next_p;
    out["next_o"] = result.next_o;
    return out;
}

std::pair<std::vector<double>, std::vector<std::int64_t>> search_root_parallel_py(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    const std::vector<double>& weights,
    const std::vector<double>& order_map,
    int time_limit_ms
) {
    SearchContext ctx;
    ctx.weights = &weights;
    ctx.order_map = &order_map;
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    py::gil_scoped_release release;
    return search_root_parallel_impl(p, o, mvs, depth, is_exact, ordered_indices, ctx);
}

std::pair<std::vector<double>, std::vector<std::int64_t>> search_root_parallel_cached_py(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    int time_limit_ms
) {
    SearchContext ctx;
    ctx.weights = &require_global_weights();
    ctx.order_map = &require_global_order_map();
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    py::gil_scoped_release release;
    return search_root_parallel_impl(p, o, mvs, depth, is_exact, ordered_indices, ctx);
}

py::dict search_root_parallel_cached_status_py(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    int time_limit_ms
) {
    SearchContext ctx;
    ctx.weights = &require_global_weights();
    ctx.order_map = &require_global_order_map();
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    std::pair<std::vector<double>, std::vector<std::int64_t>> result;
    {
        py::gil_scoped_release release;
        result = search_root_parallel_impl(p, o, mvs, depth, is_exact, ordered_indices, ctx);
    }
    py::dict out;
    out["vals"] = result.first;
    out["nodes"] = result.second;
    out["timed_out"] = ctx.timed_out;
    return out;
}

py::dict search_root_parallel_cached_status_policy_py(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    const std::vector<double>& root_policy,
    int time_limit_ms
) {
    SearchContext ctx;
    ctx.weights = &require_global_weights();
    ctx.order_map = &require_global_order_map();
    ctx.root_policy = &root_policy;
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_ms);
    }
    std::pair<std::vector<double>, std::vector<std::int64_t>> result;
    {
        py::gil_scoped_release release;
        result = search_root_parallel_impl(p, o, mvs, depth, is_exact, ordered_indices, ctx);
    }
    py::dict out;
    out["vals"] = result.first;
    out["nodes"] = result.second;
    out["timed_out"] = ctx.timed_out;
    return out;
}

py::dict get_best_move_ab_py(
    Bitboard p,
    Bitboard o,
    int mvs,
    int max_depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    const std::vector<double>& weights,
    const std::vector<double>& order_map,
    int time_limit_ms
) {
    IterativeSearchResult result;
    {
        py::gil_scoped_release release;
        result = get_best_move_ab_impl(p, o, mvs, max_depth, is_exact, ordered_indices, weights, order_map, time_limit_ms);
    }
    py::dict out;
    out["completed_depth"] = result.completed_depth;
    out["best_move"] = result.best_move;
    out["moves"] = result.moves;
    out["values"] = result.values;
    out["win_rates"] = result.win_rates;
    out["nodes"] = result.nodes;
    out["resolved"] = result.resolved;
    out["timed_out"] = result.timed_out;
    return out;
}

PYBIND11_MODULE(othello_engine, m) {
    bind_engine_free_functions(m);
    bind_engine_session_classes(m);
    m.def("set_search_params", &engine_set_search_params, "Set search parameters for C++ engine");
}

#include "othello_core_cpp.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

using Bitboard = std::uint64_t;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace {

constexpr Bitboard FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;
constexpr Bitboard NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL;
constexpr Bitboard NOT_H_FILE = 0x7F7F7F7F7F7F7F7FULL;
constexpr int EXACT_INF = 65;
constexpr std::size_t DEFAULT_TT_SIZE = 1u << 22;
constexpr int TIME_CHECK_INTERVAL_MASK = 2047;

struct TTEntry {
    Bitboard key = 0;
    std::int16_t score = 0;
    std::int8_t best_move = -1;
    std::int8_t flag = 0;  // 0=exact, 1=lower, 2=upper
    std::int8_t depth = -1;
};

struct SearchContext {
    TimePoint deadline{};
    bool use_deadline = false;
    bool timed_out = false;
    std::uint64_t nodes = 0;
    std::atomic<int>* ybwc_active_workers = nullptr;
    int ybwc_max_workers = 1;
    int ybwc_min_empties = 10;
};

struct MoveInfo {
    int square = -1;
    Bitboard flip = 0;
    int priority = 0;
};

struct EndgameIterativeResult {
    int score = 0;
    int best_move = -1;
    int completed_depth = 0;
    bool fully_solved = false;
};

TTEntry* tt_table = nullptr;
std::size_t tt_size = 0;
std::array<std::shared_mutex, 4096> tt_mutexes;

inline void init_endgame_tt() {
    if (tt_table == nullptr) {
        tt_size = DEFAULT_TT_SIZE;
        tt_table = new TTEntry[tt_size]();
    }
}

inline void clear_endgame_tt() {
    if (tt_table != nullptr) {
        std::fill(tt_table, tt_table + tt_size, TTEntry{});
    }
}

inline int count_bits(Bitboard b) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

inline int bit_scan_forward(Bitboard b) {
#ifdef _MSC_VER
    unsigned long index = 0;
    _BitScanForward64(&index, b);
    return static_cast<int>(index);
#else
    return __builtin_ctzll(b);
#endif
}

inline Bitboard lsb(Bitboard b) {
    return b & (~b + 1);
}

inline Bitboard position_key(Bitboard p, Bitboard o) {
    Bitboard key = p ^ 0x9E3779B97F4A7C15ULL;
    key ^= o + 0xC2B2AE3D27D4EB4FULL + (key << 6) + (key >> 2);
    return key != 0 ? key : 0xD6E8FEB86659FD93ULL;
}

inline std::size_t hash_key(Bitboard p, Bitboard o) {
    return static_cast<std::size_t>(position_key(p, o) & (tt_size - 1));
}

inline std::size_t tt_mutex_index(Bitboard p, Bitboard o) {
    return static_cast<std::size_t>(position_key(p, o) & (tt_mutexes.size() - 1));
}

inline bool probe_tt(Bitboard p, Bitboard o, int& score, int& best_move, int depth, int& alpha, int& beta) {
    std::shared_lock<std::shared_mutex> lock(tt_mutexes[tt_mutex_index(p, o)]);
    const TTEntry& entry = tt_table[hash_key(p, o)];
    if (entry.key == position_key(p, o)) {
        score = static_cast<int>(entry.score);
        best_move = static_cast<int>(entry.best_move);
        if (entry.depth >= depth) {
            if (entry.flag == 0) { // exact
                return true;
            } else if (entry.flag == 1) { // lower bound
                if (score >= beta) { alpha = beta; return true; }
                if (score > alpha) alpha = score;
            } else if (entry.flag == 2) { // upper bound
                if (score <= alpha) { beta = alpha; return true; }
                if (score < beta) beta = score;
            }
        }
        return false;
    }
    return false;
}

inline void store_tt(Bitboard p, Bitboard o, int score, int best_move, int depth, int orig_alpha, int orig_beta) {
    std::unique_lock<std::shared_mutex> lock(tt_mutexes[tt_mutex_index(p, o)]);
    TTEntry& entry = tt_table[hash_key(p, o)];
    entry.key = position_key(p, o);
    entry.score = static_cast<std::int16_t>(score);
    entry.best_move = static_cast<std::int8_t>(best_move);
    entry.depth = static_cast<std::int8_t>(depth);
    if (score <= orig_alpha) entry.flag = 2; // upper bound
    else if (score >= orig_beta) entry.flag = 1; // lower bound
    else entry.flag = 0; // exact
}

inline bool is_corner(int sq) {
    return sq == 0 || sq == 7 || sq == 56 || sq == 63;
}

inline bool is_x_square(int sq) {
    return sq == 9 || sq == 14 || sq == 49 || sq == 54;
}

inline bool is_c_square(int sq) {
    switch (sq) {
    case 1:
    case 6:
    case 8:
    case 15:
    case 48:
    case 55:
    case 57:
    case 62:
        return true;
    default:
        return false;
    }
}

inline bool is_edge(int sq) {
    return sq < 8 || sq >= 56 || (sq % 8) == 0 || (sq % 8) == 7;
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

constexpr Bitboard MASK_CORNER = (1ULL << 0) | (1ULL << 7) | (1ULL << 56) | (1ULL << 63);

// --- Egaroucid-inspired helpers ---

// Corner-weighted mobility: count corner legal moves twice (Egaroucid get_n_moves_cornerX2)
inline int get_n_moves_cornerX2(Bitboard legal) {
    return count_bits(legal) + count_bits(legal & MASK_CORNER);
}

// Parity quadrant masks (4 quadrants of the board)
constexpr Bitboard QUAD_TL = 0x0000000000FFFF00ULL; // rows 1-3, cols 1-3
constexpr Bitboard QUAD_TR = 0x00000000FF000000ULL; // rows 1-3, cols 4-7  (approx)
constexpr Bitboard QUAD_BL = 0x0000FFFF00000000ULL; // rows 4-6, cols 1-3  (approx)
constexpr Bitboard QUAD_BR = 0xFFFF000000000000ULL; // rows 4-7, cols 4-7  (approx)

// Get parity for a square (which quadrant it belongs to, 0-3)
inline int get_quadrant(int sq) {
    int r = sq / 8;
    int c = sq % 8;
    return ((r >> 2) << 1) | (c >> 2);
}

// Compute parity bitfield: bit i set if quadrant i has odd empties
inline int compute_parity(Bitboard empty) {
    int parity = 0;
    int q0 = count_bits(empty & 0x0000000000FFFFFFULL); // simplified quadrant approx
    int q1 = count_bits(empty & 0x00000000FF000000ULL);
    int q2 = count_bits(empty & 0x000000FF00000000ULL);
    int q3 = count_bits(empty & 0xFFFFFF0000000000ULL);
    if (q0 & 1) parity |= 1;
    if (q1 & 1) parity |= 2;
    if (q2 & 1) parity |= 4;
    if (q3 & 1) parity |= 8;
    return parity;
}

// Egaroucid-style end_evaluate: final score with all empties going to winner
inline int end_evaluate(Bitboard p, int empties) {
    int score = count_bits(p) * 2 - 64;
    int diff = score + empties; // = n_discs_p - (64 - empties - n_discs_p)
    if (diff == 0) return 0;
    if (diff > 0) return diff + empties;
    return diff - empties;
}

inline Bitboard compute_stable_fast(Bitboard p, Bitboard o) {
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
    return s;
}

inline Bitboard get_legal_moves_fast(Bitboard p, Bitboard o) {
    const Bitboard occ = p | o;
    const Bitboard empty = ~occ & FULL_MASK;
    const Bitboard m = NOT_A_FILE & NOT_H_FILE;
    Bitboard legal = 0;

    Bitboard t = (p << 1) & o & m;
    t |= (t << 1) & o & m;
    t |= (t << 1) & o & m;
    t |= (t << 1) & o & m;
    t |= (t << 1) & o & m;
    t |= (t << 1) & o & m;
    legal |= (t << 1) & empty;

    t = (p >> 1) & o & m;
    t |= (t >> 1) & o & m;
    t |= (t >> 1) & o & m;
    t |= (t >> 1) & o & m;
    t |= (t >> 1) & o & m;
    t |= (t >> 1) & o & m;
    legal |= (t >> 1) & empty;

    t = (p << 8) & o;
    t |= (t << 8) & o;
    t |= (t << 8) & o;
    t |= (t << 8) & o;
    t |= (t << 8) & o;
    t |= (t << 8) & o;
    legal |= (t << 8) & empty;

    t = (p >> 8) & o;
    t |= (t >> 8) & o;
    t |= (t >> 8) & o;
    t |= (t >> 8) & o;
    t |= (t >> 8) & o;
    t |= (t >> 8) & o;
    legal |= (t >> 8) & empty;

    t = (p << 7) & o & m;
    t |= (t << 7) & o & m;
    t |= (t << 7) & o & m;
    t |= (t << 7) & o & m;
    t |= (t << 7) & o & m;
    t |= (t << 7) & o & m;
    legal |= (t << 7) & empty;

    t = (p >> 7) & o & m;
    t |= (t >> 7) & o & m;
    t |= (t >> 7) & o & m;
    t |= (t >> 7) & o & m;
    t |= (t >> 7) & o & m;
    t |= (t >> 7) & o & m;
    legal |= (t >> 7) & empty;

    t = (p << 9) & o & m;
    t |= (t << 9) & o & m;
    t |= (t << 9) & o & m;
    t |= (t << 9) & o & m;
    t |= (t << 9) & o & m;
    t |= (t << 9) & o & m;
    legal |= (t << 9) & empty;

    t = (p >> 9) & o & m;
    t |= (t >> 9) & o & m;
    t |= (t >> 9) & o & m;
    t |= (t >> 9) & o & m;
    t |= (t >> 9) & o & m;
    t |= (t >> 9) & o & m;
    legal |= (t >> 9) & empty;

    return legal & FULL_MASK;
}

inline Bitboard get_flip_fast(Bitboard p, Bitboard o, int idx) {
    const Bitboard move_bit = 1ULL << idx;
    if (((p | o) & move_bit) != 0) {
        return 0;
    }
    int r = idx / 8;
    int c = idx % 8;
    Bitboard flips = 0;
    static constexpr int dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
    static constexpr int dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    for (int dir = 0; dir < 8; ++dir) {
        int rr = r + dr[dir];
        int cc = c + dc[dir];
        Bitboard captured = 0;
        bool seen_opponent = false;
        while (0 <= rr && rr < 8 && 0 <= cc && cc < 8) {
            const int nidx = rr * 8 + cc;
            const Bitboard bit = 1ULL << nidx;
            if ((o & bit) != 0) {
                captured |= bit;
                seen_opponent = true;
                rr += dr[dir];
                cc += dc[dir];
                continue;
            }
            if (seen_opponent && ((p & bit) != 0)) {
                flips |= captured;
            }
            break;
        }
    }
    return flips & FULL_MASK;
}

inline bool check_timeout(SearchContext& ctx) {
    if (!ctx.use_deadline) {
        return false;
    }
    ++ctx.nodes;
    if ((ctx.nodes & TIME_CHECK_INTERVAL_MASK) != 0) {
        return false;
    }
    if (Clock::now() >= ctx.deadline) {
        ctx.timed_out = true;
        return true;
    }
    return false;
}

void build_region_size_map(Bitboard empty_mask, std::array<int, 64>& region_sizes) {
    region_sizes.fill(0);
    Bitboard remaining = empty_mask & FULL_MASK;
    while (remaining) {
        Bitboard region = lsb(remaining);
        Bitboard frontier = region;
        while (frontier) {
            Bitboard expanded = neighbor_union(frontier) & empty_mask & ~region;
            region |= expanded;
            frontier = expanded;
        }
        const int size = count_bits(region);
        Bitboard bits = region;
        while (bits) {
            Bitboard bit = lsb(bits);
            bits ^= bit;
            region_sizes[static_cast<std::size_t>(bit_scan_forward(bit))] = size;
        }
        remaining &= ~region;
    }
}

std::vector<MoveInfo> generate_ordered_moves(Bitboard p, Bitboard o, Bitboard legal, int tt_move, int parity = 0) {
    const int canput = count_bits(legal);
    std::vector<MoveInfo> moves;
    moves.reserve(static_cast<std::size_t>(canput));
    Bitboard bits = legal;
    while (bits) {
        Bitboard move_bit = lsb(bits);
        bits ^= move_bit;
        const int sq = bit_scan_forward(move_bit);
        const Bitboard flip = get_flip_fast(p, o, sq);
        if (flip == 0) {
            continue;
        }
        // Wipeout detection (Egaroucid): flip == opponent means all opponent discs flipped
        if (flip == o) {
            moves.push_back(MoveInfo{sq, flip, 10000000});
            continue;
        }
        const Bitboard np = (p | move_bit | flip) & FULL_MASK;
        const Bitboard no = (o & ~flip) & FULL_MASK;
        const Bitboard opp_legal = get_legal_moves_fast(no, np);
        const int opp_mobility_cornerX2 = get_n_moves_cornerX2(opp_legal);
        const int flip_count = count_bits(flip);
        int priority = 0;

        // TT best move first (Egaroucid: W_1ST_MOVE)
        if (sq == tt_move) priority += 1000000;
        // Corner/X/C/Edge (Egaroucid-style)
        if (is_corner(sq)) priority += 50000;
        else if (is_x_square(sq)) priority -= 30000;
        else if (is_c_square(sq)) priority -= 12000;
        else if (is_edge(sq)) priority += 5000;

        // Parity bonus (Egaroucid: W_END_NWS_SIMPLE_PARITY = 17)
        // Moves in odd-empty quadrants should be searched first
        if (parity & (1 << get_quadrant(sq))) {
            priority += 1700;
        }

        // Opponent mobility penalty (Egaroucid: cornerX2 weighted, W_END_NWS_MOBILITY = 40)
        // MO_OFFSET_L_PM = 38, so priority += (38 - nm) * 40
        if (opp_mobility_cornerX2 == 0) priority += 18000;
        priority += (38 - opp_mobility_cornerX2) * 40;
        priority -= flip_count * 24;

        moves.push_back(MoveInfo{sq, flip, priority});
    }

    // Selection sort instead of full sort (Egaroucid uses swap_next_best_move)
    // Only sort the first few elements as needed during search
    std::sort(moves.begin(), moves.end(), [](const MoveInfo& a, const MoveInfo& b) {
        return a.priority > b.priority;
    });
    return moves;
}

int exact_negamax(Bitboard p, Bitboard o, int empties, bool passed, int alpha, int beta, SearchContext& ctx);
inline int acquire_ybwc_workers(SearchContext& ctx, int requested);
inline void release_ybwc_workers(SearchContext& ctx, int count);
inline void raise_atomic_int(std::atomic<int>& target, int value);

// Depth-limited negamax for iterative deepening warmup
int exact_negamax_limited(Bitboard p, Bitboard o, int depth, bool passed, int alpha, int beta, SearchContext& ctx) {
    if (check_timeout(ctx)) {
        return 0;
    }

    if (depth <= 0) {
        // Leaf: return heuristic (disc diff + mobility bonus)
        const int disc_diff = count_bits(p) - count_bits(o);
        const int opp_mob = count_bits(get_legal_moves_fast(o, p));
        const int own_mob = count_bits(get_legal_moves_fast(p, o));
        int eval = disc_diff * 100;
        eval += (own_mob - opp_mob) * 5;
        if (opp_mob == 0 && own_mob > 0) eval += 30;
        return eval;
    }

    const Bitboard legal = get_legal_moves_fast(p, o);
    if (legal == 0) {
        if (passed) {
            return count_bits(p) - count_bits(o);
        }
        return -exact_negamax_limited(o, p, depth, true, -beta, -alpha, ctx);
    }

    int tt_best_move = -1;
    int tt_score_dummy = 0;
    int dummy_a = -EXACT_INF, dummy_b = EXACT_INF;
    probe_tt(p, o, tt_score_dummy, tt_best_move, depth, dummy_a, dummy_b);

    const Bitboard empty_bb_l = ~(p | o) & FULL_MASK;
    const int parity_l = compute_parity(empty_bb_l);
    std::vector<MoveInfo> moves = generate_ordered_moves(p, o, legal, tt_best_move, parity_l);
    int best_score = -EXACT_INF;
    for (const MoveInfo& move : moves) {
        const Bitboard move_bit = 1ULL << move.square;
        const Bitboard np = (p | move.flip | move_bit) & FULL_MASK;
        const Bitboard no = (o & ~move.flip) & FULL_MASK;
        const int child_score = -exact_negamax_limited(no, np, depth - 1, false, -beta, -alpha, ctx);
        if (ctx.timed_out) return 0;
        if (child_score > best_score) best_score = child_score;
        if (child_score > alpha) alpha = child_score;
        if (alpha >= beta) break;
    }
    return best_score;
}

int exact_negamax(Bitboard p, Bitboard o, int empties, bool passed, int alpha, int beta, SearchContext& ctx) {
    if (check_timeout(ctx)) {
        return 0;
    }

    const int orig_alpha = alpha;
    const int orig_beta = beta;

    int tt_score = 0;
    int tt_best_move = -1;
    if (probe_tt(p, o, tt_score, tt_best_move, empties, alpha, beta)) {
        return tt_score;
    }
    if (alpha >= beta) return tt_score;

    // --- Stability cutoff (Egaroucid: 2*stable_p - 64 <= score <= 64 - 2*stable_o) ---
    if (empties >= 4) {
        const Bitboard sp = compute_stable_fast(p, o);
        const Bitboard so = compute_stable_fast(o, p);
        const int stable_p = count_bits(sp);
        const int stable_o = count_bits(so);
        // Upper bound: opponent can't lose more than all remaining empties
        const int n_beta = 64 - 2 * stable_o;
        // Lower bound: player can't gain more than all remaining empties
        const int n_alpha = 2 * stable_p - 64;
        if (n_beta <= alpha) return n_beta;
        if (n_alpha >= beta) return n_alpha;
        if (n_alpha > alpha) alpha = n_alpha;
        if (n_beta < beta) beta = n_beta;
        if (alpha >= beta) return alpha;
    }

    const Bitboard legal = get_legal_moves_fast(p, o);
    if (legal == 0) {
        if (passed) {
            const int final_score = end_evaluate(p, empties);
            store_tt(p, o, final_score, -1, empties, orig_alpha, orig_beta);
            return final_score;
        }
        const int score = -exact_negamax(o, p, empties, true, -beta, -alpha, ctx);
        if (!ctx.timed_out) {
            store_tt(p, o, score, -1, empties, -orig_beta, -orig_alpha);
        }
        return score;
    }

    // Compute parity for move ordering (Egaroucid-style)
    const Bitboard empty_bb = ~(p | o) & FULL_MASK;
    const int parity = compute_parity(empty_bb);

    // Generate moves with Egaroucid-style ordering
    std::vector<MoveInfo> moves = generate_ordered_moves(p, o, legal, tt_best_move, parity);
    int best_score = -EXACT_INF;
    int best_move = -1;

    // Egaroucid: if opponent has 0-1 legal moves after our move, search immediately
    // without full ordering overhead (already handled via priority in generate_ordered_moves)
    if (!moves.empty() &&
        moves.size() >= 3 &&
        empties >= ctx.ybwc_min_empties &&
        ctx.ybwc_active_workers != nullptr &&
        !check_timeout(ctx)) {
        if (moves[0].flip == o) {
            store_tt(p, o, 64, moves[0].square, empties, orig_alpha, orig_beta);
            return 64;
        }

        std::vector<int> scores(moves.size(), -EXACT_INF);
        std::vector<unsigned char> searched(moves.size(), 0);
        auto search_child = [&](std::size_t i, SearchContext& child_ctx, int alpha_snapshot) {
            const MoveInfo& move = moves[i];
            const Bitboard move_bit = 1ULL << move.square;
            const Bitboard np = (p | move.flip | move_bit) & FULL_MASK;
            const Bitboard no = (o & ~move.flip) & FULL_MASK;
            return -exact_negamax(no, np, empties - 1, false, -beta, -alpha_snapshot, child_ctx);
        };

        const int first_score = search_child(0, ctx, alpha);
        if (ctx.timed_out) {
            return 0;
        }
        scores[0] = first_score;
        searched[0] = 1;
        best_score = first_score;
        best_move = moves[0].square;
        if (first_score > alpha) {
            alpha = first_score;
        }
        if (alpha < beta) {
            const int worker_count = acquire_ybwc_workers(ctx, std::min<int>(static_cast<int>(moves.size() - 1), ctx.ybwc_max_workers));
            if (worker_count > 0) {
                std::atomic<std::size_t> next_index{1};
                std::atomic<int> shared_alpha{alpha};
                std::atomic<bool> saw_timeout{false};
                std::atomic<bool> cutoff{false};
                std::vector<std::thread> workers;
                workers.reserve(static_cast<std::size_t>(worker_count));
                for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
                    workers.emplace_back([&, worker_id]() {
                        SearchContext local_ctx;
                        local_ctx.use_deadline = ctx.use_deadline;
                        local_ctx.deadline = ctx.deadline;
                        local_ctx.ybwc_active_workers = ctx.ybwc_active_workers;
                        local_ctx.ybwc_max_workers = ctx.ybwc_max_workers;
                        local_ctx.ybwc_min_empties = ctx.ybwc_min_empties;
                        while (!cutoff.load(std::memory_order_relaxed)) {
                            const std::size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
                            if (i >= moves.size()) {
                                break;
                            }
                            if (moves[i].flip == o) {
                                scores[i] = 64;
                                searched[i] = 1;
                                raise_atomic_int(shared_alpha, 64);
                                cutoff.store(true, std::memory_order_relaxed);
                                break;
                            }
                            if (check_timeout(local_ctx)) {
                                saw_timeout.store(true, std::memory_order_relaxed);
                                break;
                            }
                            const int alpha_snapshot = shared_alpha.load(std::memory_order_acquire);
                            if (alpha_snapshot >= beta) {
                                cutoff.store(true, std::memory_order_relaxed);
                                break;
                            }
                            const int child_score = search_child(i, local_ctx, alpha_snapshot);
                            if (local_ctx.timed_out) {
                                saw_timeout.store(true, std::memory_order_relaxed);
                                break;
                            }
                            scores[i] = child_score;
                            searched[i] = 1;
                            if (child_score > alpha_snapshot) {
                                raise_atomic_int(shared_alpha, child_score);
                                if (child_score >= beta) {
                                    cutoff.store(true, std::memory_order_relaxed);
                                    break;
                                }
                            }
                        }
                    });
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                release_ybwc_workers(ctx, worker_count);
                if (saw_timeout.load(std::memory_order_relaxed)) {
                    ctx.timed_out = true;
                    return 0;
                }
            } else {
                for (std::size_t i = 1; i < moves.size(); ++i) {
                    if (check_timeout(ctx) || alpha >= beta) {
                        break;
                    }
                    if (moves[i].flip == o) {
                        scores[i] = 64;
                        searched[i] = 1;
                        alpha = 64;
                        break;
                    }
                    const int child_score = search_child(i, ctx, alpha);
                    if (ctx.timed_out) {
                        return 0;
                    }
                    scores[i] = child_score;
                    searched[i] = 1;
                    if (child_score > alpha) {
                        alpha = child_score;
                    }
                }
            }
        }

        for (std::size_t i = 0; i < moves.size(); ++i) {
            if (!searched[i]) {
                continue;
            }
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_move = moves[i].square;
            }
        }
        if (!ctx.timed_out) {
            store_tt(p, o, best_score, best_move, empties, orig_alpha, orig_beta);
        }
        return best_score;
    }

    for (const MoveInfo& move : moves) {
        // Wipeout detection (Egaroucid)
        if (move.flip == o) {
            best_score = 64;
            best_move = move.square;
            break;
        }
        const Bitboard move_bit = 1ULL << move.square;
        const Bitboard np = (p | move.flip | move_bit) & FULL_MASK;
        const Bitboard no = (o & ~move.flip) & FULL_MASK;
        const int child_score = -exact_negamax(no, np, empties - 1, false, -beta, -alpha, ctx);
        if (ctx.timed_out) {
            return 0;
        }
        if (child_score > best_score) {
            best_score = child_score;
            best_move = move.square;
        }
        if (child_score > alpha) {
            alpha = child_score;
        }
        if (alpha >= beta) {
            break;
        }
    }

    if (!ctx.timed_out) {
        store_tt(p, o, best_score, best_move, empties, orig_alpha, orig_beta);
    }
    return best_score;
}

inline int hardware_thread_count() {
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

inline int endgame_root_worker_count(int move_count, int depth) {
    if (move_count <= 1 || depth < 6) {
        return 1;
    }
    return std::max(1, std::min(move_count, hardware_thread_count()));
}

inline int search_endgame_root_child(
    const MoveInfo& move,
    Bitboard p,
    Bitboard o,
    int empties,
    int depth,
    int alpha,
    int beta,
    SearchContext& ctx
) {
    const Bitboard move_bit = 1ULL << move.square;
    const Bitboard np = (p | move.flip | move_bit) & FULL_MASK;
    const Bitboard no = (o & ~move.flip) & FULL_MASK;
    if (depth == empties) {
        return -exact_negamax(no, np, empties - 1, false, -beta, -alpha, ctx);
    }
    return -exact_negamax_limited(no, np, depth - 1, false, -beta, -alpha, ctx);
}

inline void raise_atomic_int(std::atomic<int>& target, int value) {
    int current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current, value, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
}

EndgameIterativeResult solve_endgame_exact_internal(Bitboard p, Bitboard o, int32_t player_to_move, int32_t time_limit_ms) {
    init_endgame_tt();

    if (player_to_move == -1) {
        std::swap(p, o);
    }

    const int empties = 64 - count_bits(p | o);
    const Bitboard legal = get_legal_moves_fast(p, o);
    std::atomic<int> ybwc_active_workers{0};
    const int ybwc_max_workers = std::max(1, hardware_thread_count() - 1);
    if (legal == 0) {
        const Bitboard opp_legal = get_legal_moves_fast(o, p);
        if (opp_legal == 0) {
            return EndgameIterativeResult{count_bits(p) - count_bits(o), -1, empties, true};
        }
        SearchContext ctx;
        ctx.ybwc_active_workers = &ybwc_active_workers;
        ctx.ybwc_max_workers = ybwc_max_workers;
        if (time_limit_ms > 0) {
            ctx.use_deadline = true;
            ctx.deadline = Clock::now() + std::chrono::milliseconds(time_limit_ms);
        }
        const int score = -exact_negamax(o, p, empties, true, -EXACT_INF, EXACT_INF, ctx);
        if (ctx.timed_out) {
            return EndgameIterativeResult{0, -1, 0, false};
        }
        int result_score = score;
        if (player_to_move == -1) result_score = -result_score;
        return EndgameIterativeResult{result_score, -1, empties, true};
    }

    // --- Iterative deepening ---
    // Start from a shallow depth and work up to full depth.
    // This warms up the TT and provides partial results on timeout.
    int start_depth = std::max(4, empties - 16);
    int best_move = -1;
    int best_score = 0;
    int completed_depth = 0;

    for (int depth = start_depth; depth <= empties; ++depth) {
        SearchContext ctx;
        ctx.ybwc_active_workers = &ybwc_active_workers;
        ctx.ybwc_max_workers = ybwc_max_workers;
        if (time_limit_ms > 0) {
            // Reserve 15% of remaining time for deeper iterations
            ctx.use_deadline = true;
            ctx.deadline = Clock::now() + std::chrono::milliseconds(time_limit_ms);
        }

        int tt_score = 0;
        int tt_best_move = -1;
        int dummy_alpha = -EXACT_INF, dummy_beta = EXACT_INF;
        probe_tt(p, o, tt_score, tt_best_move, depth, dummy_alpha, dummy_beta);
        const Bitboard empty_bb_id = ~(p | o) & FULL_MASK;
        const int parity_id = compute_parity(empty_bb_id);
        std::vector<MoveInfo> moves = generate_ordered_moves(p, o, legal, tt_best_move, parity_id);
        int cur_best_score = -EXACT_INF;
        int cur_best_move = -1;
        int alpha = -EXACT_INF;
        const int beta = EXACT_INF;

        if (!moves.empty()) {
            const int worker_count = endgame_root_worker_count(static_cast<int>(moves.size()), depth);
            std::vector<int> scores(moves.size(), -EXACT_INF);
            std::vector<unsigned char> searched(moves.size(), 0);

            const int first_score = search_endgame_root_child(moves[0], p, o, empties, depth, alpha, beta, ctx);
            if (ctx.timed_out) {
                if (completed_depth > 0) {
                    int result_score = best_score;
                    if (player_to_move == -1) result_score = -result_score;
                    return EndgameIterativeResult{result_score, best_move, completed_depth, false};
                }
                return EndgameIterativeResult{0, -1, 0, false};
            }
            scores[0] = first_score;
            searched[0] = 1;
            cur_best_score = first_score;
            cur_best_move = moves[0].square;
            if (first_score > alpha) {
                alpha = first_score;
            }

            if (alpha < beta && moves.size() > 1) {
                if (worker_count <= 1) {
                    for (std::size_t i = 1; i < moves.size(); ++i) {
                        const int child_score = search_endgame_root_child(moves[i], p, o, empties, depth, alpha, beta, ctx);
                        if (ctx.timed_out) {
                            break;
                        }
                        scores[i] = child_score;
                        searched[i] = 1;
                        if (child_score > alpha) {
                            alpha = child_score;
                        }
                        if (alpha >= beta) {
                            break;
                        }
                    }
                } else {
                    std::atomic<std::size_t> next_index{1};
                    std::atomic<int> shared_alpha{alpha};
                    std::atomic<bool> saw_timeout{false};
                    std::atomic<bool> cutoff{false};
                    std::vector<std::thread> workers;
                    const int spawned = std::max(1, std::min(worker_count, static_cast<int>(moves.size() - 1)));
                    workers.reserve(static_cast<std::size_t>(spawned));
                    for (int worker_id = 0; worker_id < spawned; ++worker_id) {
                        workers.emplace_back([&, worker_id]() {
                            SearchContext local_ctx;
                            local_ctx.use_deadline = ctx.use_deadline;
                            local_ctx.deadline = ctx.deadline;
                            local_ctx.ybwc_active_workers = ctx.ybwc_active_workers;
                            local_ctx.ybwc_max_workers = ctx.ybwc_max_workers;
                            local_ctx.ybwc_min_empties = ctx.ybwc_min_empties;
                            while (!cutoff.load(std::memory_order_relaxed)) {
                                const std::size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
                                if (i >= moves.size()) {
                                    break;
                                }
                                if (check_timeout(local_ctx)) {
                                    saw_timeout.store(true, std::memory_order_relaxed);
                                    break;
                                }
                                const int alpha_snapshot = shared_alpha.load(std::memory_order_acquire);
                                if (alpha_snapshot >= beta) {
                                    cutoff.store(true, std::memory_order_relaxed);
                                    break;
                                }
                                const int child_score = search_endgame_root_child(moves[i], p, o, empties, depth, alpha_snapshot, beta, local_ctx);
                                if (local_ctx.timed_out) {
                                    saw_timeout.store(true, std::memory_order_relaxed);
                                    break;
                                }
                                scores[i] = child_score;
                                searched[i] = 1;
                                if (child_score > alpha_snapshot) {
                                    raise_atomic_int(shared_alpha, child_score);
                                    if (child_score >= beta) {
                                        cutoff.store(true, std::memory_order_relaxed);
                                        break;
                                    }
                                }
                            }
                        });
                    }
                    for (auto& worker : workers) {
                        worker.join();
                    }
                    if (saw_timeout.load(std::memory_order_relaxed)) {
                        ctx.timed_out = true;
                    }
                    alpha = std::max(alpha, shared_alpha.load(std::memory_order_acquire));
                }
            }

            if (ctx.timed_out) {
                if (completed_depth > 0) {
                    int result_score = best_score;
                    if (player_to_move == -1) result_score = -result_score;
                    return EndgameIterativeResult{result_score, best_move, completed_depth, false};
                }
                return EndgameIterativeResult{0, -1, 0, false};
            }

            for (std::size_t i = 0; i < moves.size(); ++i) {
                if (!searched[i]) {
                    continue;
                }
                if (scores[i] > cur_best_score) {
                    cur_best_score = scores[i];
                    cur_best_move = moves[i].square;
                }
            }
        }

        best_score = cur_best_score;
        best_move = cur_best_move;
        completed_depth = depth;

        // If we solved the full depth, we're done
        if (depth == empties) {
            break;
        }
    }

    if (player_to_move == -1) {
        best_score = -best_score;
    }
    return EndgameIterativeResult{best_score, best_move, completed_depth, completed_depth >= empties};
}

}  // namespace

extern "C" int solve_endgame_exact(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms
) {
    const int empties = 64 - count_bits(P | O);
    if (max_depth < empties) {
        return static_cast<int>(count_bits(P) - count_bits(O)) * (player_to_move == 1 ? 1 : -1);
    }
    EndgameIterativeResult result = solve_endgame_exact_internal(P, O, player_to_move, time_limit_ms);
    if (result.fully_solved) {
        return result.score;
    }
    return static_cast<int>(count_bits(P) - count_bits(O)) * (player_to_move == 1 ? 1 : -1);
}

extern "C" int get_endgame_best_move(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms
) {
    const int empties = 64 - count_bits(P | O);
    if (max_depth < empties) {
        return -1;
    }
    EndgameIterativeResult result = solve_endgame_exact_internal(P, O, player_to_move, time_limit_ms);
    return result.fully_solved ? result.best_move : -1;
}

extern "C" int solve_endgame_exact_status(
    uint64_t P,
    uint64_t O,
    int32_t player_to_move,
    int32_t max_depth,
    int32_t time_limit_ms,
    int32_t* out_best_move,
    int32_t* out_score,
    int32_t* out_completed_depth
) {
    const int empties = 64 - count_bits(P | O);
    if (max_depth < empties) {
        if (out_best_move != nullptr) *out_best_move = -1;
        if (out_score != nullptr) *out_score = 0;
        if (out_completed_depth != nullptr) *out_completed_depth = 0;
        return 0;
    }

    EndgameIterativeResult result = solve_endgame_exact_internal(P, O, player_to_move, time_limit_ms);
    if (out_best_move != nullptr) *out_best_move = result.best_move;
    if (out_score != nullptr) *out_score = result.score;
    if (out_completed_depth != nullptr) *out_completed_depth = result.completed_depth;
    return result.fully_solved ? 1 : 0;
}

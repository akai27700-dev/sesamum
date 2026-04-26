#include "othello_core_cpp.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
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
};

struct SearchContext {
    TimePoint deadline{};
    bool use_deadline = false;
    bool timed_out = false;
    std::uint64_t nodes = 0;
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

inline bool probe_tt(Bitboard p, Bitboard o, int& score, int& best_move) {
    const TTEntry& entry = tt_table[hash_key(p, o)];
    if (entry.key == position_key(p, o)) {
        score = static_cast<int>(entry.score);
        best_move = static_cast<int>(entry.best_move);
        return true;
    }
    return false;
}

inline void store_tt(Bitboard p, Bitboard o, int score, int best_move) {
    TTEntry& entry = tt_table[hash_key(p, o)];
    entry.key = position_key(p, o);
    entry.score = static_cast<std::int16_t>(score);
    entry.best_move = static_cast<std::int8_t>(best_move);
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

std::vector<MoveInfo> generate_ordered_moves(Bitboard p, Bitboard o, Bitboard legal, int tt_move) {
    std::array<int, 64> region_sizes{};
    const Bitboard empty_mask = (~(p | o)) & FULL_MASK;
    build_region_size_map(empty_mask, region_sizes);

    std::vector<MoveInfo> moves;
    moves.reserve(static_cast<std::size_t>(count_bits(legal)));
    Bitboard bits = legal;
    while (bits) {
        Bitboard move_bit = lsb(bits);
        bits ^= move_bit;
        const int sq = bit_scan_forward(move_bit);
        const Bitboard flip = get_flip_fast(p, o, sq);
        if (flip == 0) {
            continue;
        }
        const Bitboard np = (p | move_bit | flip) & FULL_MASK;
        const Bitboard no = (o & ~flip) & FULL_MASK;
        const Bitboard opp_legal = get_legal_moves_fast(no, np);
        const int opp_mobility = count_bits(opp_legal);
        const int flip_count = count_bits(flip);
        int priority = 0;

        if (sq == tt_move) priority += 1000000;
        if (is_corner(sq)) priority += 50000;
        else if (is_x_square(sq)) priority -= 30000;
        else if (is_c_square(sq)) priority -= 12000;
        else if (is_edge(sq)) priority += 5000;

        const int region_size = region_sizes[static_cast<std::size_t>(sq)];
        if ((region_size & 1) != 0) priority += 3000;
        else if (region_size > 0) priority -= 1200;

        if (opp_mobility == 0) priority += 18000;
        priority -= opp_mobility * 512;
        priority -= flip_count * 24;

        moves.push_back(MoveInfo{sq, flip, priority});
    }

    std::sort(moves.begin(), moves.end(), [](const MoveInfo& a, const MoveInfo& b) {
        return a.priority > b.priority;
    });
    return moves;
}

int exact_negamax(Bitboard p, Bitboard o, int empties, bool passed, int alpha, int beta, SearchContext& ctx) {
    if (check_timeout(ctx)) {
        return 0;
    }

    int tt_score = 0;
    int tt_best_move = -1;
    if (probe_tt(p, o, tt_score, tt_best_move)) {
        return tt_score;
    }

    const Bitboard legal = get_legal_moves_fast(p, o);
    if (legal == 0) {
        if (passed) {
            const int final_score = count_bits(p) - count_bits(o);
            store_tt(p, o, final_score, -1);
            return final_score;
        }
        const int score = -exact_negamax(o, p, empties, true, -beta, -alpha, ctx);
        if (!ctx.timed_out) {
            store_tt(p, o, score, -1);
        }
        return score;
    }

    std::vector<MoveInfo> moves = generate_ordered_moves(p, o, legal, tt_best_move);
    int best_score = -EXACT_INF;
    int best_move = -1;
    for (const MoveInfo& move : moves) {
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
        store_tt(p, o, best_score, best_move);
    }
    return best_score;
}

EndgameIterativeResult solve_endgame_exact_internal(Bitboard p, Bitboard o, int32_t player_to_move, int32_t time_limit_ms) {
    init_endgame_tt();

    if (player_to_move == -1) {
        std::swap(p, o);
    }

    const int empties = 64 - count_bits(p | o);
    const Bitboard legal = get_legal_moves_fast(p, o);
    if (legal == 0) {
        const Bitboard opp_legal = get_legal_moves_fast(o, p);
        if (opp_legal == 0) {
            return EndgameIterativeResult{count_bits(p) - count_bits(o), -1, empties, true};
        }
        SearchContext ctx;
        if (time_limit_ms > 0) {
            ctx.use_deadline = true;
            ctx.deadline = Clock::now() + std::chrono::milliseconds(time_limit_ms);
        }
        const int score = -exact_negamax(o, p, empties, true, -EXACT_INF, EXACT_INF, ctx);
        if (ctx.timed_out) {
            return EndgameIterativeResult{0, -1, 0, false};
        }
        return EndgameIterativeResult{score, -1, empties, true};
    }

    SearchContext ctx;
    if (time_limit_ms > 0) {
        ctx.use_deadline = true;
        ctx.deadline = Clock::now() + std::chrono::milliseconds(time_limit_ms);
    }

    int tt_score = 0;
    int tt_best_move = -1;
    std::vector<MoveInfo> moves = generate_ordered_moves(p, o, legal, probe_tt(p, o, tt_score, tt_best_move) ? tt_best_move : -1);
    int best_score = -EXACT_INF;
    int best_move = -1;
    int alpha = -EXACT_INF;
    const int beta = EXACT_INF;

    for (const MoveInfo& move : moves) {
        const Bitboard move_bit = 1ULL << move.square;
        const Bitboard np = (p | move.flip | move_bit) & FULL_MASK;
        const Bitboard no = (o & ~move.flip) & FULL_MASK;
        const int child_score = -exact_negamax(no, np, empties - 1, false, -beta, -alpha, ctx);
        if (ctx.timed_out) {
            return EndgameIterativeResult{0, -1, 0, false};
        }
        if (child_score > best_score) {
            best_score = child_score;
            best_move = move.square;
        }
        if (child_score > alpha) {
            alpha = child_score;
        }
    }

    store_tt(p, o, best_score, best_move);
    if (player_to_move == -1) {
        best_score = -best_score;
    }
    return EndgameIterativeResult{best_score, best_move, empties, true};
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
    if (out_completed_depth != nullptr) *out_completed_depth = result.fully_solved ? empties : 0;
    return result.fully_solved ? 1 : 0;
}

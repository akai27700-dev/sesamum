#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

using Bitboard = std::uint64_t;

constexpr Bitboard FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;

Bitboard engine_get_legal_moves(Bitboard p, Bitboard o);
Bitboard engine_get_flip(Bitboard p, Bitboard o, int idx);
int engine_count_bits(Bitboard x);
int engine_bit_index(Bitboard x);
Bitboard engine_lsb(Bitboard x);
std::vector<int> engine_legal_move_indices(Bitboard p, Bitboard o);
double engine_calculate_win_rate(double ev, bool is_exact);
std::tuple<std::vector<double>, std::vector<std::int64_t>, bool> engine_search_root_parallel_layer(
    Bitboard p,
    Bitboard o,
    int mvs,
    int depth,
    bool is_exact,
    const std::vector<int>& ordered_indices,
    const std::vector<double>* root_policy,
    int time_limit_ms
);

void bind_engine_session_classes(py::module_& m);

namespace {

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
        Bitboard bit = engine_lsb(valid);
        valid ^= bit;
        mask[static_cast<std::size_t>(engine_bit_index(bit))] = 1.0f;
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
        const auto valid_mask = build_valid_mask_array(engine_get_legal_moves(p, o));
        visited_.insert(state);
        priors_[state] = normalize_priors(policy_logits, valid_mask, add_noise);
        state_visits_[state] = 0;
    }

    py::dict collect_leaves(Bitboard p, Bitboard o, int turn, int batch_size, const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& stop_flag) {
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

    void expand_leaves(const std::vector<int>& tickets,
                       const py::array_t<float, py::array::c_style | py::array::forcecast>& policy_batch,
                       const py::array_t<float, py::array::c_style | py::array::forcecast>& value_batch);

    RootStatsResult root_stats_plain(Bitboard p, Bitboard o, int turn) const;
    py::dict root_stats(Bitboard p, Bitboard o, int turn) const;
    BatchStepResult collect_and_expand_plain(Bitboard p, Bitboard o, int turn, int batch_size, const std::uint8_t* stop_ptr, const py::function& infer_batch);
    py::dict collect_and_expand(Bitboard p,
                                Bitboard o,
                                int turn,
                                int batch_size,
                                const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& stop_flag,
                                py::function infer_batch);

private:
    struct TraverseResult {
        bool is_terminal = false;
        MCTSStateKey leaf{};
        std::vector<std::pair<MCTSStateKey, int>> path;
        double terminal_value = 0.0;
    };

    LeafBatch collect_leaves_plain(Bitboard p, Bitboard o, int turn, int batch_size, const std::uint8_t* stop_ptr);
    void expand_leaves_plain(const std::vector<int>& tickets, const std::vector<std::array<float, 64>>& policy_batch, const std::vector<float>& value_batch);
    std::array<float, 64> normalize_priors(const std::array<float, 64>& policy_logits, const std::array<float, 64>& valid_mask, bool add_noise);
    TraverseResult traverse(Bitboard p, Bitboard o, int turn) const;
    void apply_virtual_loss(const std::vector<std::pair<MCTSStateKey, int>>& path);
    void backpropagate_terminal(const std::vector<std::pair<MCTSStateKey, int>>& path, double value);
    void expand_leaf(const MCTSStateKey& leaf, const std::vector<std::pair<MCTSStateKey, int>>& path, const std::array<float, 64>& policy_logits, float value);
    int lookup_visits(const MCTSActionKey& key) const;
    int lookup_state_visits(const MCTSStateKey& key) const;
    double lookup_value(const MCTSActionKey& key) const;

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

void BatchMCTS::expand_leaves(const std::vector<int>& tickets,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& policy_batch,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& value_batch) {
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

BatchMCTS::RootStatsResult BatchMCTS::root_stats_plain(Bitboard p, Bitboard o, int turn) const {
    RootStatsResult out;
    const MCTSStateKey state{p, o, turn};
    Bitboard valid = engine_get_legal_moves(p, o);
    int best_action = -1;
    int max_visits = -1;
    int total_root_visits = 0;
    Bitboard temp = valid;
    while (temp) {
        Bitboard bit = engine_lsb(temp);
        temp ^= bit;
        const int action = engine_bit_index(bit);
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
        Bitboard bit = engine_lsb(temp);
        temp ^= bit;
        const int action = engine_bit_index(bit);
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

py::dict BatchMCTS::root_stats(Bitboard p, Bitboard o, int turn) const {
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

BatchMCTS::BatchStepResult BatchMCTS::collect_and_expand_plain(Bitboard p, Bitboard o, int turn, int batch_size, const std::uint8_t* stop_ptr, const py::function& infer_batch) {
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

py::dict BatchMCTS::collect_and_expand(Bitboard p,
                                       Bitboard o,
                                       int turn,
                                       int batch_size,
                                       const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& stop_flag,
                                       py::function infer_batch) {
    py::dict out;
    BatchStepResult result = collect_and_expand_plain(p, o, turn, batch_size, stop_flag.data(), infer_batch);
    out["simulation_count"] = result.simulation_count;
    out["leaf_count"] = result.leaf_count;
    return out;
}

BatchMCTS::LeafBatch BatchMCTS::collect_leaves_plain(Bitboard p, Bitboard o, int turn, int batch_size, const std::uint8_t* stop_ptr) {
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

void BatchMCTS::expand_leaves_plain(const std::vector<int>& tickets, const std::vector<std::array<float, 64>>& policy_batch, const std::vector<float>& value_batch) {
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

std::array<float, 64> BatchMCTS::normalize_priors(const std::array<float, 64>& policy_logits, const std::array<float, 64>& valid_mask, bool add_noise) {
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

BatchMCTS::TraverseResult BatchMCTS::traverse(Bitboard p, Bitboard o, int turn) const {
    TraverseResult result;
    Bitboard current_p = p;
    Bitboard current_o = o;
    int current_turn = turn;
    while (true) {
        const MCTSStateKey state{current_p, current_o, current_turn};
        const Bitboard valid = engine_get_legal_moves(current_p, current_o);
        if (valid == 0) {
            if (engine_get_legal_moves(current_o, current_p) == 0) {
                const int own_count = engine_count_bits(current_p);
                const int opp_count = engine_count_bits(current_o);
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
            Bitboard bit = engine_lsb(temp);
            temp ^= bit;
            const int action = engine_bit_index(bit);
            const MCTSActionKey action_key{state, action};
            const int visits = lookup_visits(action_key);
            const double q = visits > 0 ? lookup_value(action_key) / static_cast<double>(visits) : 0.0;
            const double u = q + sqrt_ns * static_cast<double>(prior_it->second[static_cast<std::size_t>(action)]) / static_cast<double>(1 + visits);
            if (u > best_score) {
                best_score = u;
                best_action = action;
            }
        }
        result.path.emplace_back(state, best_action);
        const Bitboard flips = engine_get_flip(current_p, current_o, best_action);
        const Bitboard next_p = (current_p | (1ULL << best_action) | flips) & FULL_MASK;
        const Bitboard next_o = (current_o ^ flips) & FULL_MASK;
        current_p = next_o;
        current_o = next_p;
        current_turn = -current_turn;
    }
}

void BatchMCTS::apply_virtual_loss(const std::vector<std::pair<MCTSStateKey, int>>& path) {
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

void BatchMCTS::backpropagate_terminal(const std::vector<std::pair<MCTSStateKey, int>>& path, double value) {
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

void BatchMCTS::expand_leaf(const MCTSStateKey& leaf, const std::vector<std::pair<MCTSStateKey, int>>& path, const std::array<float, 64>& policy_logits, float value) {
    visited_.insert(leaf);
    const auto valid_mask = build_valid_mask_array(engine_get_legal_moves(leaf.p, leaf.o));
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

int BatchMCTS::lookup_visits(const MCTSActionKey& key) const {
    const auto it = edge_visits_.find(key);
    return it == edge_visits_.end() ? 0 : it->second;
}

int BatchMCTS::lookup_state_visits(const MCTSStateKey& key) const {
    const auto it = state_visits_.find(key);
    return it == state_visits_.end() ? 0 : it->second;
}

double BatchMCTS::lookup_value(const MCTSActionKey& key) const {
    const auto it = edge_values_.find(key);
    return it == edge_values_.end() ? 0.0 : it->second;
}

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

    SearchSessionResult run(Bitboard p,
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
                            int multi_cut_depth = 8);

private:
    static int current_mcts_batch_size(int batch_size, double remain);
    static std::array<float, 64> make_root_policy(const std::vector<double>& root_policy, Bitboard p, Bitboard o, int turn, const py::function& infer_batch);
    static void emit_ab_progress(const py::function& ab_progress, int depth, double elapsed_sec, std::int64_t layer_nodes, const SearchSessionABResult& ab_result);

    BatchMCTS mcts_;
};

SearchSessionResult SearchSession::run(Bitboard p,
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
                                       bool multi_cut_enabled,
                                       int multi_cut_threshold,
                                       int multi_cut_depth) {
    SearchSessionResult result;
    const auto start_time = std::chrono::steady_clock::now();
    const auto overall_deadline = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(std::max(0.0, time_limit_sec)));
    const std::vector<int> initial_order = ordered_indices.empty() ? engine_legal_move_indices(p, o) : ordered_indices;
    const int empty_count = 64 - engine_count_bits(p | o);
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
                if (stop_ptr != nullptr && stop_ptr[0] != 0) break;
                const auto now = std::chrono::steady_clock::now();
                if (now >= overall_deadline) break;
                const double remain = std::chrono::duration<double>(overall_deadline - now).count();
                if (remain <= 0.0) break;
                const int curr_batch = current_mcts_batch_size(mcts_batch_size, remain);
                BatchMCTS::BatchStepResult step = mcts_.collect_and_expand_plain(p, o, turn, curr_batch, stop_ptr, infer_batch);
                result.mcts.simulation_count += step.simulation_count;
                if (step.leaf_count <= 0) continue;
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
                if (stop_ptr != nullptr && stop_ptr[0] != 0) break;
                ab_result.attempted_depth = depth;
                const bool depth_exact = is_exact && depth >= empty_count;
                std::vector<double> root_policy_buffer;
                const std::vector<double>* root_policy_ptr = nullptr;
                if (have_root_policy) {
                    root_policy_buffer.assign(root_policy_arr.begin(), root_policy_arr.end());
                    root_policy_ptr = &root_policy_buffer;
                }
                int remain_ms = 0;
                const auto now = std::chrono::steady_clock::now();
                const auto hard_deadline = ab_deadline < overall_deadline ? ab_deadline : overall_deadline;
                if (now >= hard_deadline) break;
                remain_ms = static_cast<int>(std::max(0LL, std::chrono::duration_cast<std::chrono::milliseconds>(hard_deadline - now).count()));
                auto [vals, nodes, timed_out] = engine_search_root_parallel_layer(p, o, mvs, depth, depth_exact, curr_ordered, root_policy_ptr, remain_ms);
                if (timed_out) {
                    ab_result.timed_out = true;
                    break;
                }
                if (curr_ordered.empty()) break;
                struct LayerResult { int move; double value; double win_rate; };
                std::vector<LayerResult> combined;
                combined.reserve(curr_ordered.size());
                std::int64_t layer_nodes = 0;
                for (std::size_t i = 0; i < curr_ordered.size(); ++i) {
                    const double win_rate = engine_calculate_win_rate(vals[i], depth_exact);
                    combined.push_back(LayerResult{curr_ordered[i], vals[i], win_rate});
                    layer_nodes += nodes[i];
                }
                std::sort(combined.begin(), combined.end(), [](const LayerResult& a, const LayerResult& b) { return a.win_rate > b.win_rate; });
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
                if (depth_exact) {
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

    if (mcts_thread.joinable()) mcts_thread.join();
    if (ab_thread.joinable()) ab_thread.join();
    return result;
}

int SearchSession::current_mcts_batch_size(int batch_size, double remain) {
    if (batch_size <= 0) return 1;
    if (remain >= 10.0) return std::min(batch_size, 1048576);
    if (remain >= 5.0) return std::min(batch_size, 786432);
    if (remain >= 2.0) return std::min(batch_size, 524288);
    if (remain >= 1.0) return std::min(batch_size, 262144);
    if (remain >= 0.4) return std::min(batch_size, 131072);
    return std::min(batch_size, 65536);
}

std::array<float, 64> SearchSession::make_root_policy(const std::vector<double>& root_policy, Bitboard p, Bitboard o, int turn, const py::function& infer_batch) {
    std::array<float, 64> out{};
    bool has_policy = false;
    if (root_policy.size() == 64) {
        for (std::size_t i = 0; i < 64; ++i) {
            out[i] = static_cast<float>(root_policy[i]);
            if (out[i] != 0.0f) has_policy = true;
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

void SearchSession::emit_ab_progress(const py::function& ab_progress, int depth, double elapsed_sec, std::int64_t layer_nodes, const SearchSessionABResult& ab_result) {
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

}  

void bind_engine_session_classes(py::module_& m) {
    py::class_<BatchMCTS>(m, "BatchMCTS")
        .def(py::init<double, double>(), py::arg("c_puct") = 2.0, py::arg("virtual_loss") = 1.0)
        .def("initialize_root", &BatchMCTS::initialize_root, py::arg("p"), py::arg("o"), py::arg("turn"), py::arg("policy_logits"), py::arg("add_noise") = false)
        .def("collect_leaves", &BatchMCTS::collect_leaves, py::arg("p"), py::arg("o"), py::arg("turn"), py::arg("batch_size"), py::arg("stop_flag"))
        .def("expand_leaves", &BatchMCTS::expand_leaves, py::arg("tickets"), py::arg("policy_batch"), py::arg("value_batch"))
        .def("collect_and_expand", &BatchMCTS::collect_and_expand, py::arg("p"), py::arg("o"), py::arg("turn"), py::arg("batch_size"), py::arg("stop_flag"), py::arg("infer_batch"))
        .def("root_stats", &BatchMCTS::root_stats, py::arg("p"), py::arg("o"), py::arg("turn"));

    py::class_<SearchSession>(m, "SearchSession")
        .def(py::init<double, double>(), py::arg("c_puct") = 2.0, py::arg("virtual_loss") = 1.0)
        .def("run", [](SearchSession& self,
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
                       py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> stop_flag,
                       py::function infer_batch,
                       py::object ab_progress,
                       bool multi_cut_enabled = false,
                       int multi_cut_threshold = 3,
                       int multi_cut_depth = 8) {
            if (stop_flag.ndim() != 1 || stop_flag.shape(0) < 1) {
                throw std::invalid_argument("stop_flag must be a uint8 array with at least one element");
            }
            py::function progress_cb = ab_progress.is_none() ? py::function() : ab_progress.cast<py::function>();
            SearchSessionResult result;
            {
                py::gil_scoped_release release;
                result = self.run(p, o, turn, mvs, start_depth, is_exact, ordered_indices, root_policy, use_ab, use_mcts, time_limit_sec, mcts_batch_size, ab_delay_sec, ab_time_limit_ms, max_depth, add_root_noise, stop_flag.mutable_data(), infer_batch, progress_cb, multi_cut_enabled, multi_cut_threshold, multi_cut_depth);
            }
            py::dict ab_out;
            ab_out["completed_depth"] = result.ab.completed_depth;
            ab_out["attempted_depth"] = result.ab.attempted_depth;
            ab_out["resolved"] = result.ab.resolved;
            ab_out["timed_out"] = result.ab.timed_out;
            ab_out["nodes"] = result.ab.total_nodes;
            ab_out["moves"] = result.ab.moves;
            ab_out["values"] = result.ab.values;
            ab_out["win_rates"] = result.ab.win_rates;
            py::dict mcts_out;
            py::dict mcts_scores;
            py::dict mcts_visits;
            for (const auto& entry : result.mcts.move_win_rates) {
                mcts_scores[py::int_(entry.first)] = entry.second;
            }
            for (const auto& entry : result.mcts.root_visits) {
                mcts_visits[py::int_(entry.first)] = entry.second;
            }
            mcts_out["move_win_rates"] = mcts_scores;
            mcts_out["root_visits"] = mcts_visits;
            mcts_out["best_wr"] = result.mcts.best_wr;
            mcts_out["simulation_count"] = result.mcts.simulation_count;
            mcts_out["nn_batch_count"] = result.mcts.nn_batch_count;
            mcts_out["nn_leaf_count"] = result.mcts.nn_leaf_count;
            py::dict out;
            out["ab"] = ab_out;
            out["mcts"] = mcts_out;
            return out;
        }, py::arg("p"), py::arg("o"), py::arg("turn"), py::arg("mvs"), py::arg("start_depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("root_policy"), py::arg("use_ab"), py::arg("use_mcts"), py::arg("time_limit_sec"), py::arg("mcts_batch_size"), py::arg("ab_delay_sec"), py::arg("ab_time_limit_ms"), py::arg("max_depth") = 60, py::arg("add_root_noise") = false, py::arg("stop_flag"), py::arg("infer_batch"), py::arg("ab_progress") = py::none(), py::arg("multi_cut_enabled") = false, py::arg("multi_cut_threshold") = 3, py::arg("multi_cut_depth") = 8);
}

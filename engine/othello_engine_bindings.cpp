#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "othello_core_cpp.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

using Bitboard = std::uint64_t;

void engine_clear_tt();
void clear_exact_cache_py();
double engine_calculate_mcts_influence_ratio(int simulation_count, int empty_count, bool is_auto_mode = false);
void engine_set_eval_data(const py::array_t<double, py::array::c_style | py::array::forcecast>& weights,
                          const py::array_t<double, py::array::c_style | py::array::forcecast>& order_map);
std::vector<double> engine_benchmark_optimizations(int iterations = 1000);


int solve_endgame_exact_py(Bitboard p, Bitboard o, int player_to_move, int max_depth, int time_limit_ms) {
    return solve_endgame_exact(p, o, player_to_move, max_depth, time_limit_ms);
}

int get_endgame_best_move_py(Bitboard p, Bitboard o, int player_to_move, int max_depth, int time_limit_ms) {
    return get_endgame_best_move(p, o, player_to_move, max_depth, time_limit_ms);
}

py::dict solve_endgame_status_py(Bitboard p, Bitboard o, int player_to_move, int max_depth, int time_limit_ms) {
    int best_move = -1;
    int score = 0;
    int completed_depth = 0;
    const int fully_solved = solve_endgame_exact_status(p, o, player_to_move, max_depth, time_limit_ms, &best_move, &score, &completed_depth);
    py::dict out;
    out["best_move"] = best_move;
    out["score"] = score;
    out["completed_depth"] = completed_depth;
    out["fully_solved"] = (fully_solved != 0);
    return out;
}
std::vector<int> engine_legal_move_indices(Bitboard p, Bitboard o);
std::pair<Bitboard, Bitboard> engine_apply_move(Bitboard p, Bitboard o, int idx);
bool engine_should_use_early_exact(Bitboard p, Bitboard o, int empties, int base_threshold);

py::dict probe_exact_cache_py(Bitboard p, Bitboard o, int turn, int empty_count);
void store_exact_cache_py(Bitboard p,
                          Bitboard o,
                          int turn,
                          int best_move,
                          double best_value,
                          double best_win_rate,
                          const std::unordered_map<int, double>& move_values,
                          const std::unordered_map<int, double>& move_win_rates,
                          int empty_count);
py::dict probe_tt_py(Bitboard p, Bitboard o);
Bitboard get_legal_moves_py(Bitboard p, Bitboard o);
Bitboard get_flip_py(Bitboard p, Bitboard o, int idx);
double evaluate_board_full_py(Bitboard p, Bitboard o, int mvs, const std::vector<double>& weights);
std::vector<double> evaluate_moves_py(Bitboard p, Bitboard o, int mvs, const std::vector<int>& indices, const std::vector<double>& weights);
double evaluate_board_cached_py(Bitboard p, Bitboard o, int mvs);
std::vector<double> evaluate_moves_cached_py(Bitboard p, Bitboard o, int mvs, const std::vector<int>& indices);
py::dict analyze_legal_moves_cached_py(Bitboard p, Bitboard o, int mvs);
std::pair<std::vector<double>, std::vector<std::int64_t>> search_root_parallel_py(Bitboard p,
                                                                                   Bitboard o,
                                                                                   int mvs,
                                                                                   int depth,
                                                                                   bool is_exact,
                                                                                   const std::vector<int>& ordered_indices,
                                                                                   const std::vector<double>& weights,
                                                                                   const std::vector<double>& order_map,
                                                                                   int time_limit_ms);
std::pair<std::vector<double>, std::vector<std::int64_t>> search_root_parallel_cached_py(Bitboard p,
                                                                                          Bitboard o,
                                                                                          int mvs,
                                                                                          int depth,
                                                                                          bool is_exact,
                                                                                          const std::vector<int>& ordered_indices,
                                                                                          int time_limit_ms);
py::dict search_root_parallel_cached_status_py(Bitboard p,
                                               Bitboard o,
                                               int mvs,
                                               int depth,
                                               bool is_exact,
                                               const std::vector<int>& ordered_indices,
                                               int time_limit_ms);
py::dict search_root_parallel_cached_status_policy_py(Bitboard p,
                                                      Bitboard o,
                                                      int mvs,
                                                      int depth,
                                                      bool is_exact,
                                                      const std::vector<int>& ordered_indices,
                                                      const std::vector<double>& root_policy,
                                                      int time_limit_ms);
py::dict get_best_move_ab_py(Bitboard p,
                             Bitboard o,
                             int mvs,
                             int max_depth,
                             bool is_exact,
                             const std::vector<int>& ordered_indices,
                             const std::vector<double>& weights,
                             const std::vector<double>& order_map,
                             int time_limit_ms);


double evaluate_board_full_cpp_py(Bitboard P, Bitboard O, int mvs, py::array_t<double> weights) {
    auto buf = weights.request();
    if (buf.size != 243) {
        throw std::runtime_error("Weights must have 243 elements");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    return evaluate_board_full_cpp(P, O, mvs, ptr);
}


py::array_t<double> evaluate_board_batch_cpp_py(
    py::array_t<uint64_t> P_arr,
    py::array_t<uint64_t> O_arr,
    py::array_t<int32_t> mvs_arr,
    py::array_t<double> weights
) {
    auto P_buf = P_arr.request();
    auto O_buf = O_arr.request();
    auto mvs_buf = mvs_arr.request();
    auto W_buf = weights.request();
    
    if (W_buf.size != 243) {
        throw std::runtime_error("Weights must have 243 elements");
    }
    
    int32_t count = static_cast<int32_t>(P_buf.size);
    if (O_buf.size != count || mvs_buf.size != count) {
        throw std::runtime_error("P, O, and mvs arrays must have same length");
    }
    
    uint64_t* P_ptr = static_cast<uint64_t*>(P_buf.ptr);
    uint64_t* O_ptr = static_cast<uint64_t*>(O_buf.ptr);
    int32_t* mvs_ptr = static_cast<int32_t*>(mvs_buf.ptr);
    double* W_ptr = static_cast<double*>(W_buf.ptr);
    
    
    py::array_t<double> results(count);
    auto res_buf = results.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    
    
    evaluate_board_batch_cpp(P_ptr, O_ptr, mvs_ptr, W_ptr, res_ptr, count);
    
    return results;
}

void bind_engine_free_functions(py::module_& m) {
    m.def("clear_tt", &engine_clear_tt);
    m.def("clear_exact_cache", &clear_exact_cache_py);
    m.def("probe_exact_cache", &probe_exact_cache_py);
    m.def("store_exact_cache", &store_exact_cache_py);
    m.def("calculate_mcts_influence", &engine_calculate_mcts_influence_ratio, py::arg("simulation_count"), py::arg("empty_count"), py::arg("is_auto_mode") = false);
    m.def("has_openmp", &engine_has_openmp);
    m.def("get_openmp_max_threads", &engine_get_openmp_max_threads);
    m.def("get_openmp_thread_count", &engine_get_openmp_thread_count);
    m.def("set_openmp_threads", &engine_set_openmp_threads, py::arg("thread_count"));
    m.def("set_eval_data", &engine_set_eval_data);
    m.def("benchmark_optimizations", &engine_benchmark_optimizations, py::arg("iterations") = 1000);
    m.def("probe_tt", &probe_tt_py);
    m.def("get_legal_moves", &get_legal_moves_py);
    m.def("legal_move_indices", &engine_legal_move_indices);
    m.def("apply_move", &engine_apply_move);
    m.def("get_flip", &get_flip_py);
    m.def("evaluate_board_full", &evaluate_board_full_py);
    m.def("evaluate_board_full_cpp", &evaluate_board_full_cpp_py, "Fast C++ evaluation function",
          py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("weights"));
    m.def("evaluate_board_batch_cpp", &evaluate_board_batch_cpp_py, "Parallel batch C++ evaluation",
          py::arg("p_arr"), py::arg("o_arr"), py::arg("mvs_arr"), py::arg("weights"));
    m.def("evaluate_moves", &evaluate_moves_py);
    m.def("evaluate_board_cached", &evaluate_board_cached_py);
    m.def("evaluate_moves_cached", &evaluate_moves_cached_py);
    m.def("analyze_legal_moves_cached", &analyze_legal_moves_cached_py);
    m.def("search_root_parallel", &search_root_parallel_py, py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("weights"), py::arg("order_map"), py::arg("time_limit_ms") = 0);
    m.def("search_root_parallel_cached", &search_root_parallel_cached_py, py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("time_limit_ms") = 0);
    m.def("search_root_parallel_cached_status", &search_root_parallel_cached_status_py, py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("time_limit_ms") = 0);
    m.def("search_root_parallel_cached_status_policy", &search_root_parallel_cached_status_policy_py, py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("root_policy"), py::arg("time_limit_ms") = 0);
    m.def("should_use_early_exact", &engine_should_use_early_exact, py::arg("p"), py::arg("o"), py::arg("empties"), py::arg("base_threshold"));
    m.def("get_best_move_ab", &get_best_move_ab_py, py::arg("p"), py::arg("o"), py::arg("mvs"), py::arg("max_depth"), py::arg("is_exact"), py::arg("ordered_indices"), py::arg("weights"), py::arg("order_map"), py::arg("time_limit_ms") = 0);
    
    
    m.def("solve_endgame_exact", &solve_endgame_exact_py, py::arg("p"), py::arg("o"), py::arg("player_to_move"), py::arg("max_depth"), py::arg("time_limit_ms") = 5000);
    m.def("get_endgame_best_move", &get_endgame_best_move_py, py::arg("p"), py::arg("o"), py::arg("player_to_move"), py::arg("max_depth"), py::arg("time_limit_ms") = 5000);
    m.def("solve_endgame_status", &solve_endgame_status_py, py::arg("p"), py::arg("o"), py::arg("player_to_move"), py::arg("max_depth"), py::arg("time_limit_ms") = 5000);
}

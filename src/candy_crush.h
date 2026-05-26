#ifndef CANDY_CRUSH_H
#define CANDY_CRUSH_H

#include <vector>
#include <tuple>
#include <string>
#include <set>
#include <random>
#include <memory>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cstring>

class CandyCrushEnv {
public:
    CandyCrushEnv(int max_steps = 100, int target_score = 1000, int seed = -1);

    // 拷贝构造函数（用于clone）
    CandyCrushEnv(const CandyCrushEnv& other);

    std::vector<std::vector<int>> reset();
    std::tuple<std::vector<std::vector<int>>, int, bool> step(
        const std::tuple<int, int, int, int>& action);
    std::vector<std::tuple<int, int, int, int>> get_valid_actions();
    void render();

    // 基本状态设置
    void set_state(const std::vector<std::vector<int>>& new_board, int new_steps, int new_score);

    // 真正的clone：返回一个新的独立环境副本
    std::unique_ptr<CandyCrushEnv> clone() const;

    std::vector<std::vector<int>> get_board() const { return board; }
    int get_score() const { return score; }
    int get_steps() const { return steps; }
    int get_max_steps() const { return max_steps; }
    int get_target_score() const { return target_score; }
    std::tuple<int, int, int, int> get_last_swap() const { return last_swap; }
    std::vector<std::string> get_last_reward_breakdown() const { 
        return last_reward_breakdown; 
    }

    std::string get_board_string() const;

private:
    static constexpr int BOARD_SIZE = 8;
    static constexpr int NUM_COLORS = 6;
    static constexpr int MAX_OBSTACLES = 7;

    std::vector<std::vector<int>> board;
    int score;
    int steps;
    int max_steps;
    int target_score;
    std::tuple<int, int, int, int> last_swap;
    std::vector<std::string> last_reward_breakdown;

    std::mt19937 rng;

    void initialize_board();
    void ensure_valid_moves();
    bool has_valid_moves() const;
    bool is_valid_swap(int r1, int c1, int r2, int c2) const;
    std::tuple<std::vector<int>, std::vector<int>> find_matches() const;
    std::tuple<int, std::vector<std::string>> execute_cascade(int reward_multiplier = 1);
    void apply_gravity();
    void fill_empty();
    void auto_shuffle();
    void convert_obstacles();
    bool is_safe_color(int row, int col, int color) const;
    void supplement_obstacles();
    bool is_valid_action(int r1, int c1, int r2, int c2) const;

    std::string get_emoji(int value) const;
    void print_board_simple() const;
    void print_board_with_highlights(const std::set<std::pair<int, int>>& highlight_positions) const;
    void print_valid_actions(const std::vector<std::tuple<int, int, int, int>>& valid_actions) const;

    void swap_cells(int r1, int c1, int r2, int c2) {
        std::swap(board[r1][c1], board[r2][c2]);
    }
};

#endif

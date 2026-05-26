#include "candy_crush.h"

CandyCrushEnv::CandyCrushEnv(int max_steps, int target_score, int seed)
    : max_steps(max_steps), target_score(target_score), score(0), steps(0) {

    board = std::vector<std::vector<int>>(BOARD_SIZE, std::vector<int>(BOARD_SIZE));

    if (seed >= 0) {
        rng.seed(seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
}

// 拷贝构造函数：完整拷贝所有成员，包括RNG的内部状态
CandyCrushEnv::CandyCrushEnv(const CandyCrushEnv& other)
    : board(other.board),
      score(other.score),
      steps(other.steps),
      max_steps(other.max_steps),
      target_score(other.target_score),
      last_swap(other.last_swap),
      last_reward_breakdown(other.last_reward_breakdown),
      rng(other.rng)  // mt19937的拷贝构造函数会精确复制624个状态字
{
}

// clone：使用拷贝构造函数创建堆上的副本
std::unique_ptr<CandyCrushEnv> CandyCrushEnv::clone() const {
    return std::make_unique<CandyCrushEnv>(*this);
}

std::vector<std::vector<int>> CandyCrushEnv::reset() {
    score = 0;
    steps = 0;
    last_swap = std::make_tuple(0, 0, 0, 0);
    last_reward_breakdown.clear();
    initialize_board();
    ensure_valid_moves();
    return board;
}

void CandyCrushEnv::set_state(const std::vector<std::vector<int>>& new_board, 
                               int new_steps, int new_score) {
    board = new_board;
    steps = new_steps;
    score = new_score;
    last_swap = std::make_tuple(0, 0, 0, 0);
    last_reward_breakdown.clear();
}

void CandyCrushEnv::initialize_board() {
    std::uniform_int_distribution<int> color_dist(0, NUM_COLORS - 1);

    while (true) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                board[r][c] = color_dist(rng);
            }
        }

        while (true) {
            auto [rows, cols] = find_matches();
            if (rows.empty()) break;

            for (size_t i = 0; i < rows.size(); ++i) {
                board[rows[i]][cols[i]] = -2;
            }
            apply_gravity();
            fill_empty();
        }

        if (has_valid_moves()) break;
    }

    std::uniform_int_distribution<int> obstacle_dist(1, MAX_OBSTACLES);
    int num_obstacles = obstacle_dist(rng);

    std::vector<std::pair<int, int>> all_positions;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            all_positions.push_back({r, c});
        }
    }

    std::shuffle(all_positions.begin(), all_positions.end(), rng);

    for (int i = 0; i < num_obstacles; ++i) {
        auto [row, col] = all_positions[i];
        board[row][col] = -1;
    }

    if (!has_valid_moves()) {
        ensure_valid_moves();
    }
}

void CandyCrushEnv::ensure_valid_moves() {
    if (!has_valid_moves()) {
        auto_shuffle();
    }
}

bool CandyCrushEnv::has_valid_moves() const {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == -1) continue;

            if (col + 1 < BOARD_SIZE && board[row][col + 1] != -1) {
                if (const_cast<CandyCrushEnv*>(this)->is_valid_swap(row, col, row, col + 1)) {
                    return true;
                }
            }

            if (row + 1 < BOARD_SIZE && board[row + 1][col] != -1) {
                if (const_cast<CandyCrushEnv*>(this)->is_valid_swap(row, col, row + 1, col)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool CandyCrushEnv::is_valid_swap(int r1, int c1, int r2, int c2) const {
    const_cast<CandyCrushEnv*>(this)->swap_cells(r1, c1, r2, c2);

    auto [rows, cols] = find_matches();
    bool has_match = !rows.empty();

    const_cast<CandyCrushEnv*>(this)->swap_cells(r1, c1, r2, c2);

    return has_match;
}

std::tuple<std::vector<int>, std::vector<int>> CandyCrushEnv::find_matches() const {
    std::set<std::pair<int, int>> matched_positions;

    for (int row = 0; row < BOARD_SIZE; ++row) {
        int col = 0;
        while (col < BOARD_SIZE) {
            if (board[row][col] <= -1) {
                ++col;
                continue;
            }

            int color = board[row][col];
            int end_col = col;
            while (end_col + 1 < BOARD_SIZE && board[row][end_col + 1] == color) {
                ++end_col;
            }

            if (end_col - col + 1 >= 3) {
                for (int c = col; c <= end_col; ++c) {
                    matched_positions.insert({row, c});
                }
            }

            col = end_col + 1;
        }
    }

    for (int col = 0; col < BOARD_SIZE; ++col) {
        int row = 0;
        while (row < BOARD_SIZE) {
            if (board[row][col] <= -1) {
                ++row;
                continue;
            }

            int color = board[row][col];
            int end_row = row;
            while (end_row + 1 < BOARD_SIZE && board[end_row + 1][col] == color) {
                ++end_row;
            }

            if (end_row - row + 1 >= 3) {
                for (int r = row; r <= end_row; ++r) {
                    matched_positions.insert({r, col});
                }
            }

            row = end_row + 1;
        }
    }

    std::vector<int> rows, cols;
    for (const auto& [r, c] : matched_positions) {
        rows.push_back(r);
        cols.push_back(c);
    }

    return {rows, cols};
}

std::tuple<int, std::vector<std::string>> CandyCrushEnv::execute_cascade(int reward_multiplier) {
    int total_reward = 0;
    int cascade_multiplier = 1;
    std::vector<std::string> breakdown;

    while (true) {
        auto [rows, cols] = find_matches();
        if (rows.empty()) break;

        int matched_count = static_cast<int>(rows.size());

        int reward = matched_count * cascade_multiplier * reward_multiplier;
        total_reward += reward;

        for (size_t i = 0; i < rows.size(); ++i) {
            board[rows[i]][cols[i]] = -2;
        }

        apply_gravity();
        fill_empty();

        ++cascade_multiplier;
    }

    return {total_reward, breakdown};
}

void CandyCrushEnv::apply_gravity() {
    for (int col = 0; col < BOARD_SIZE; ++col) {
        std::vector<int> non_empty;
        for (int row = 0; row < BOARD_SIZE; ++row) {
            if (board[row][col] != -2) {
                non_empty.push_back(board[row][col]);
            }
        }

        int idx = static_cast<int>(non_empty.size()) - 1;
        for (int row = BOARD_SIZE - 1; row >= 0; --row) {
            if (idx >= 0) {
                board[row][col] = non_empty[idx--];
            } else {
                board[row][col] = -2;
            }
        }
    }
}

void CandyCrushEnv::fill_empty() {
    std::uniform_int_distribution<int> color_dist(0, NUM_COLORS - 1);

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == -2) {
                board[row][col] = color_dist(rng);
            }
        }
    }
}

void CandyCrushEnv::auto_shuffle() {
    int shuffle_count = 0;

    while (true) {
        ++shuffle_count;

        std::vector<int> all_pieces;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                all_pieces.push_back(board[r][c]);
            }
        }

        std::shuffle(all_pieces.begin(), all_pieces.end(), rng);

        int idx = 0;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                board[r][c] = all_pieces[idx++];
            }
        }

        auto [rows, cols] = find_matches();
        if (!rows.empty()) {
            execute_cascade(0);
        }

        if (has_valid_moves()) break;
    }
}

void CandyCrushEnv::convert_obstacles() {
    std::vector<std::pair<int, int>> obstacle_positions;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board[r][c] == -1) {
                obstacle_positions.push_back({r, c});
            }
        }
    }

    if (obstacle_positions.empty()) return;

    for (size_t i = 0; i < obstacle_positions.size(); ++i) {
        auto [row, col] = obstacle_positions[i];

        std::vector<int> safe_colors;
        for (int color = 0; color < NUM_COLORS; ++color) {
            if (is_safe_color(row, col, color)) {
                safe_colors.push_back(color);
            }
        }

        if (!safe_colors.empty()) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(safe_colors.size()) - 1);
            int new_color = safe_colors[dist(rng)];
            board[row][col] = new_color;
        }
    }
}

bool CandyCrushEnv::is_safe_color(int row, int col, int color) const {
    int original = board[row][col];
    const_cast<CandyCrushEnv*>(this)->board[row][col] = color;

    bool has_new_match = false;

    if (col >= 2) {
        if (board[row][col] == board[row][col - 1] && 
            board[row][col] == board[row][col - 2] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }
    if (col >= 1 && col + 1 < BOARD_SIZE) {
        if (board[row][col] == board[row][col - 1] && 
            board[row][col] == board[row][col + 1] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }
    if (col + 2 < BOARD_SIZE) {
        if (board[row][col] == board[row][col + 1] && 
            board[row][col] == board[row][col + 2] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }

    if (row >= 2) {
        if (board[row][col] == board[row - 1][col] && 
            board[row][col] == board[row - 2][col] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }
    if (row >= 1 && row + 1 < BOARD_SIZE) {
        if (board[row][col] == board[row - 1][col] && 
            board[row][col] == board[row + 1][col] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }
    if (row + 2 < BOARD_SIZE) {
        if (board[row][col] == board[row + 1][col] && 
            board[row][col] == board[row + 2][col] && 
            board[row][col] != -1) {
            has_new_match = true;
        }
    }

    const_cast<CandyCrushEnv*>(this)->board[row][col] = original;

    return !has_new_match;
}

void CandyCrushEnv::supplement_obstacles() {
    int n_old = 0;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board[r][c] == -1) ++n_old;
        }
    }

    std::uniform_int_distribution<int> dist(0, MAX_OBSTACLES);
    int target_num = dist(rng);

    if (target_num > n_old) {
        int need_to_add = target_num - n_old;

        for (int i = 0; i < need_to_add; ++i) {
            std::vector<std::pair<int, int>> candy_positions;
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    if (board[r][c] != -1) {
                        candy_positions.push_back({r, c});
                    }
                }
            }

            if (!candy_positions.empty()) {
                std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(candy_positions.size()) - 1);
                auto [row, col] = candy_positions[pos_dist(rng)];
                board[row][col] = -1;
            }
        }
    }
}

std::tuple<std::vector<std::vector<int>>, int, bool> CandyCrushEnv::step(
    const std::tuple<int, int, int, int>& action) {

    auto [r1, c1, r2, c2] = action;
    last_swap = action;
    last_reward_breakdown.clear();

    ensure_valid_moves();

    if (!is_valid_action(r1, c1, r2, c2)) {
        throw std::invalid_argument("Invalid action");
    }

    swap_cells(r1, c1, r2, c2);

    auto [reward, breakdown] = execute_cascade(1);
    last_reward_breakdown = breakdown;

    convert_obstacles();

    if (!has_valid_moves()) {
        auto_shuffle();
    } else {
        supplement_obstacles();

        if (!has_valid_moves()) {
            auto_shuffle();
        }
    }

    score += reward;
    ++steps;

    bool done = (steps >= max_steps) || (score >= target_score);

    return {board, reward, done};
}

std::vector<std::tuple<int, int, int, int>> CandyCrushEnv::get_valid_actions() {
    std::vector<std::tuple<int, int, int, int>> valid_actions;

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == -1) continue;

            if (col + 1 < BOARD_SIZE && board[row][col + 1] != -1) {
                if (is_valid_swap(row, col, row, col + 1)) {
                    valid_actions.push_back({row, col, row, col + 1});
                }
            }

            if (row + 1 < BOARD_SIZE && board[row + 1][col] != -1) {
                if (is_valid_swap(row, col, row + 1, col)) {
                    valid_actions.push_back({row, col, row + 1, col});
                }
            }
        }
    }

    return valid_actions;
}

bool CandyCrushEnv::is_valid_action(int r1, int c1, int r2, int c2) const {
    if (r1 < 0 || r1 >= BOARD_SIZE || c1 < 0 || c1 >= BOARD_SIZE ||
        r2 < 0 || r2 >= BOARD_SIZE || c2 < 0 || c2 >= BOARD_SIZE) {
        return false;
    }

    if (std::abs(r1 - r2) + std::abs(c1 - c2) != 1) {
        return false;
    }

    if (board[r1][c1] == -1 || board[r2][c2] == -1) {
        return false;
    }

    return const_cast<CandyCrushEnv*>(this)->is_valid_swap(r1, c1, r2, c2);
}

void CandyCrushEnv::render() {
    print_board_simple();
    auto valid_moves = get_valid_actions();
    std::cout << "Valid moves: " << valid_moves.size() << std::endl;
}

std::string CandyCrushEnv::get_emoji(int value) const {
    switch (value) {
        case -1: return "X ";
        case 0:  return "R ";
        case 1:  return "B ";
        case 2:  return "G ";
        case 3:  return "P ";
        case 4:  return "O ";
        case 5:  return "Y ";
        case -2: return ". ";
        default: return "? ";
    }
}

void CandyCrushEnv::print_board_simple() const {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            std::cout << get_emoji(board[row][col]);
        }
        std::cout << std::endl;
    }
}

void CandyCrushEnv::print_board_with_highlights(
    const std::set<std::pair<int, int>>& highlight_positions) const {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            std::string emoji = get_emoji(board[row][col]);
            if (highlight_positions.count({row, col})) {
                std::cout << "[" << emoji << "]";
            } else {
                std::cout << " " << emoji << " ";
            }
        }
        std::cout << std::endl;
    }
}

void CandyCrushEnv::print_valid_actions(
    const std::vector<std::tuple<int, int, int, int>>& valid_actions) const {
    std::cout << "  Valid moves: " << valid_actions.size() << std::endl;
}

std::string CandyCrushEnv::get_board_string() const {
    std::ostringstream oss;
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            oss << std::setw(2) << board[row][col] << " ";
        }
        oss << "\n";
    }
    return oss.str();
}

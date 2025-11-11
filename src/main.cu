#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>

#include "cuda_runtime.h"

#define N_ACTIONS 2
#define ACT_PROB 1.0 / N_ACTIONS

#define N_CARDS 3
#define N_POSS N_CARDS * (N_CARDS - 1)
#define POSS_PROB 1.0 / N_POSS

#define N_ITERATIONS 10000
#define N_INFOSETS 12

struct Actions {
    double call = 0.;
    double bet = 0.;
};

__global__
void fill_strategies(Actions* data) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_INFOSETS) {
        data[idx].call = ACT_PROB;
        data[idx].bet = ACT_PROB;
    }
}

__global__
void next_strategies(Actions* regret_sums, Actions* strategies, Actions* strategies_sums, double* reach_prs, double* reach_prs_sums) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_INFOSETS) {
        strategies_sums[idx].call += reach_prs[idx] * strategies[idx].call;
        strategies_sums[idx].bet += reach_prs[idx] * strategies[idx].bet;
    
        strategies[idx].call = fmax(regret_sums[idx].call, 0.);
        strategies[idx].bet = fmax(regret_sums[idx].bet, 0.);

        double total = strategies[idx].call + strategies[idx].bet;

        strategies[idx].call = ((total > 0) ? strategies[idx].call / total : ACT_PROB);
        strategies[idx].bet  = ((total > 0) ? strategies[idx].bet  / total : ACT_PROB);

        reach_prs_sums[idx] += reach_prs[idx];
        reach_prs[idx] = 0.;
    }
}

struct InfoSetMap {
    Actions* regret_sums;
    Actions* strategies;
    Actions* strategies_sums;

    double* reach_prs;
    double* reach_prs_sums;

    InfoSetMap() {
        std::size_t v_size = N_INFOSETS * sizeof(Actions);
        std::size_t s_size = N_INFOSETS * sizeof(double);

        cudaMalloc(&regret_sums, v_size);
        cudaMalloc(&strategies, v_size);
        cudaMalloc(&strategies_sums, v_size);

        cudaMalloc(&reach_prs, s_size);
        cudaMalloc(&reach_prs_sums, s_size);

        cudaMemset(regret_sums, 0, v_size);
        cudaMemset(strategies_sums, 0, v_size);

        cudaMemset(reach_prs, 0, s_size);
        cudaMemset(reach_prs_sums, 0, s_size);

        fill_strategies<<<1, N_INFOSETS>>>(strategies);
        cudaDeviceSynchronize();
    }

    void next_strategy() {
        next_strategies<<<1, N_INFOSETS>>>(regret_sums, strategies, strategies_sums, reach_prs, reach_prs_sums);
        cudaDeviceSynchronize();
    }

    ~InfoSetMap() {
        cudaFree(regret_sums);
        cudaFree(strategies);
        cudaFree(strategies_sums);

        cudaFree(reach_prs);
        cudaFree(reach_prs_sums);
    }
};

class History {
public:
    static const int MAX_LEN = 3;

    char seq[MAX_LEN];
    int len;

    __device__ 
    History() : len(0) {}

    __device__
        History(char action) : len(1) {
        seq[0] = action;
    }

    __device__ 
    History operator+(char action) const {
        History result = *this;
        if (result.len < MAX_LEN) {
            result.seq[result.len++] = action;
        }
        return result;
    }

    __device__ 
    bool is_terminal() const {
        if (len == 2) {
            if (seq[0] == 'c' && seq[1] == 'c') return true;
            if (seq[0] == 'b' && seq[1] == 'c') return true;
            if (seq[0] == 'b' && seq[1] == 'b') return true;
        }
        else if (len == 3) {
            if (seq[0] == 'c' && seq[1] == 'b' && (seq[2] == 'c' || seq[2] == 'b')) return true;
        }
        return false;
    }

    __device__ 
    double terminal_utils(size_t card_1, size_t card_2) const {
        int card_player = (len % 2 == 0) ? card_1 : card_2;
        int card_opponent = (len % 2 == 0) ? card_2 : card_1;

        if ((len == 3 && seq[0] == 'c' && seq[1] == 'b' && seq[2] == 'c') ||
            (len == 2 && seq[0] == 'b' && seq[1] == 'c')) {
            return 1.0;
        }
        else if (len == 2 && seq[0] == 'c' && seq[1] == 'c') {
            return (card_player > card_opponent) ? 1.0 : -1.0;
        }

        return (card_player > card_opponent) ? 2.0 : -2.0;
    }

    __device__
    size_t infoset_idx(size_t card_1, size_t card_2) const {
        size_t card = (len % 2 == 0) ? card_1 : card_2;

        if (len == 0) {
            return card * 4;
        }
        else if (len == 1) {
            return card * 4 + 1 + (seq[0] == 'b');
        }
        else {
            return card * 4 + 3;
        }
    }

    __device__
    bool is_player_1() const {
        return len % 2 == 0;
    }
};

__device__
double cfr_recursive(double* reach_prs, Actions* strategies, Actions* regret_sums, History history, double pr_1, double pr_2, double pr_c) {
    size_t cards_idx = threadIdx.x;
    size_t card_1 = cards_idx / 2;
    size_t card_2 = (card_1 + (cards_idx % 2) + 1) % 3;

    if (history.is_terminal())
        return history.terminal_utils(card_1, card_2);

    size_t is_idx = history.infoset_idx(card_1, card_2);

    Actions actions_utils;
    double util;

    if (history.is_player_1()) {
        atomicAdd(&reach_prs[is_idx], pr_1);
        actions_utils.call  = -1. * cfr_recursive(reach_prs, strategies, regret_sums, history + 'c', pr_1 * strategies[is_idx].call, pr_2, pr_c);
        actions_utils.bet   = -1. * cfr_recursive(reach_prs, strategies, regret_sums, history + 'b', pr_1 * strategies[is_idx].bet, pr_2, pr_c);

        util = actions_utils.call * strategies[is_idx].call + actions_utils.bet * strategies[is_idx].bet;

        atomicAdd(&regret_sums[is_idx].call, pr_2 * pr_c * (actions_utils.call - util));
        atomicAdd(&regret_sums[is_idx].bet, pr_2 * pr_c * (actions_utils.bet - util));
    }
    else {
        atomicAdd(&reach_prs[is_idx], pr_2);
        actions_utils.call  = -1. * cfr_recursive(reach_prs, strategies, regret_sums, history + 'c', pr_1, pr_2 * strategies[is_idx].call, pr_c);
        actions_utils.bet   = -1. * cfr_recursive(reach_prs, strategies, regret_sums, history + 'b', pr_1, pr_2 * strategies[is_idx].bet, pr_c);

        util = actions_utils.call * strategies[is_idx].call + actions_utils.bet * strategies[is_idx].bet;

        atomicAdd(&regret_sums[is_idx].call, pr_1 * pr_c * (actions_utils.call - util));
        atomicAdd(&regret_sums[is_idx].bet, pr_1 * pr_c * (actions_utils.bet - util));
    }

    return util;
}

__global__
void cfr_enterence(double* reach_prs, Actions* strategies, Actions* regret_sums, double* util) {
    size_t idx = threadIdx.x;
    bool is_call = (idx % 2 == 0);

    __shared__ double s_reach_prs[N_INFOSETS];
    __shared__ Actions s_strategies[N_INFOSETS];
    __shared__ Actions s_regret_sums[N_INFOSETS];
    __shared__ double utils[N_POSS];
    
    if (idx < N_INFOSETS) {
        s_reach_prs[idx] = reach_prs[idx];
        s_strategies[idx] = strategies[idx];
        s_regret_sums[idx] = regret_sums[idx];
    }

    __syncthreads();

    if (idx < N_POSS)
        utils[idx] = cfr_recursive(s_reach_prs, s_strategies, s_regret_sums, History(), 1., 1., POSS_PROB);

    __syncthreads();

    if (idx < N_INFOSETS) {
        reach_prs[idx] = s_reach_prs[idx];
        regret_sums[idx] = s_regret_sums[idx];
    }

    if (idx == 0) {
        *util = 0;
        for (int i = 0; i < N_POSS; ++i)
            *util += utils[i];
    } 
}

std::vector<double> get_average_strategy(Actions* strategies_sums, double*  reach_prs_sums) {
    std::vector<double> average_strategy(N_INFOSETS * N_ACTIONS, 0.0);

    for (int i = 0; i < N_INFOSETS; ++i) {
        double reach = reach_prs_sums[i];

        double call = (reach > 0.0) ? strategies_sums[i].call / reach : 0.0;
        double bet = (reach > 0.0) ? strategies_sums[i].bet / reach : 0.0;

        if (call < 0.001) call = 0.0;
        if (bet < 0.001) bet = 0.0;

        double total = call + bet;
        if (total > 0) {
            call /= total;
            bet /= total;
        }

        average_strategy[i * N_ACTIONS + 0] = call;
        average_strategy[i * N_ACTIONS + 1] = bet;
    }

    return average_strategy;
}

void display_result(InfoSetMap& i_map, double expected_game_value) {// выгружаем данные
    std::cout << "Player 1 expected value: " << expected_game_value << std::endl;
    std::cout << "Player 2 expected value: " << -expected_game_value << std::endl << std::endl;

    Actions h_strategies_sums[N_INFOSETS];
    double  h_reach_prs_sums[N_INFOSETS];

    cudaMemcpy(h_strategies_sums, i_map.strategies_sums, sizeof(Actions) * N_INFOSETS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reach_prs_sums, i_map.reach_prs_sums, sizeof(double) * N_INFOSETS, cudaMemcpyDeviceToHost);

    std::vector<double> avg_strategy = get_average_strategy(h_strategies_sums, h_reach_prs_sums);

    static const char* histories[4] = { "", "c", "b", "cb" };
    static const char* cards[3] = { "J", "Q", "K" };

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Player 1 strategies:" << std::endl;

    for (int card = 0; card < 3; ++card) {
        for (int h = 0; h < 4; h+=3) {
            int idx = card * 4 + h;
            double call = avg_strategy[idx * N_ACTIONS + 0];
            double bet = avg_strategy[idx * N_ACTIONS + 1];

            std::string info = cards[card] + std::string("rr") + histories[h];

            std::cout << std::left << std::setw(8) << info << " [" << call << ", " << bet << "]\n";
        }
    }

    std::cout << std::endl << "Player 2 strategies:" << std::endl;

    for (int card = 0; card < 3; ++card) {
        for (int h = 1; h < 3; ++h) {
            int idx = card * 4 + h;
            double call = avg_strategy[idx * N_ACTIONS + 0];
            double bet = avg_strategy[idx * N_ACTIONS + 1];

            std::string info = cards[card] + std::string("rr") + histories[h];

            std::cout << std::left << std::setw(8) << info << " [" << call << ", " << bet << "]\n";
        }
    }
}

int main() {
    double expected_game_value = 0.;

    InfoSetMap i_map;

    double* d_util = nullptr;
    double  h_util;

    cudaMalloc(&d_util, sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < N_ITERATIONS; ++i) {
        cfr_enterence<<<1, N_INFOSETS>>>(i_map.reach_prs, i_map.strategies, i_map.regret_sums, d_util);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_util, d_util, sizeof(double), cudaMemcpyDeviceToHost);

        expected_game_value += h_util / 6.0;

        i_map.next_strategy();
    }

    cudaEventRecord(stop); // конец измерения
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    expected_game_value /= N_ITERATIONS;

    display_result(i_map, expected_game_value);

    cudaFree(d_util);

    return 0;
}
#include "CFR.cuh"

#include <iostream>
#include <iomanip>

#include "Constants.h"
#include "CFRKernels.cuh"

double cfr(InfoSetMap& i_map, const size_t n_iterations) {
    double expected_game_value = 0.;

    double* d_utils = nullptr;
    double  h_utils[N_POSS];

    cudaMalloc(&d_utils, N_POSS * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < n_iterations; ++i) {
        cfr_enter << <1, N_INFOSETS >> > (i_map.reach_prs, i_map.strategies, i_map.regret_sums, d_utils);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_utils, d_utils, N_POSS * sizeof(double), cudaMemcpyDeviceToHost);

        double util = 0.0;

        for (const double h_util : h_utils)
            util += h_util;

        expected_game_value += (util / 6.0);

        i_map.next_strategy();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_utils);

    expected_game_value /= n_iterations;

    return expected_game_value;
}

std::vector<double> get_average_strategy(const Actions* strategies_sums, const double* reach_prs_sums) {
    std::vector<double> average_strategy(N_INFOSETS * N_ACTIONS, 0.0);

    for (int i = 0; i < N_INFOSETS; ++i) {
        const double reach = reach_prs_sums[i];

        double call = (reach > 0.0) ? strategies_sums[i].call / reach : 0.0;
        double bet = (reach > 0.0) ? strategies_sums[i].bet / reach : 0.0;

        if (call < 0.001) call = 0.0;
        if (bet < 0.001) bet = 0.0;

        if (const double total = call + bet; total > 0) {
            call /= total;
            bet /= total;
        }

        average_strategy[i * N_ACTIONS + 0] = call;
        average_strategy[i * N_ACTIONS + 1] = bet;
    }

    return average_strategy;
}

void display_result(const InfoSetMap& i_map, const double expected_game_value) {
    std::cout << "Player 1 expected value: " << expected_game_value << std::endl;
    std::cout << "Player 2 expected value: " << -expected_game_value << std::endl << std::endl;

    Actions h_strategies_sums[N_INFOSETS];
    double  h_reach_prs_sums[N_INFOSETS];

    cudaMemcpy(h_strategies_sums, i_map.strategies_sums, sizeof(Actions) * N_INFOSETS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reach_prs_sums, i_map.reach_prs_sums, sizeof(double) * N_INFOSETS, cudaMemcpyDeviceToHost);

    const std::vector<double> avg_strategy = get_average_strategy(h_strategies_sums, h_reach_prs_sums);

    static const char* histories[4] = { "", "c", "b", "cb" };
    static const char* cards[3] = { "J", "Q", "K" };

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Player 1 strategies:" << std::endl;

    for (int card = 0; card < 3; ++card) {
        for (int h = 0; h < 4; h += 3) {
            const int idx = card * 4 + h;
            const double call = avg_strategy[idx * N_ACTIONS + 0];
            const double bet = avg_strategy[idx * N_ACTIONS + 1];

            std::string info = cards[card] + std::string("rr") + histories[h];

            std::cout << std::left << std::setw(8) << info << " [" << call << ", " << bet << "]\n";
        }
    }

    std::cout << std::endl << "Player 2 strategies:" << std::endl;

    for (int card = 0; card < 3; ++card) {
        for (int h = 1; h < 3; ++h) {
            const int idx = card * 4 + h;
            const double call = avg_strategy[idx * N_ACTIONS + 0];
            const double bet = avg_strategy[idx * N_ACTIONS + 1];

            std::string info = cards[card] + std::string("rr") + histories[h];

            std::cout << std::left << std::setw(8) << info << " [" << call << ", " << bet << "]\n";
        }
    }
}
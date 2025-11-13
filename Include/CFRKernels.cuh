#include "cuda_runtime.h"

#include "InfoSetMap.cuh"
#include "History.cuh"

__device__
double cfr_rec(double* reach_prs, Actions* strategies, Actions* regret_sums, const History history, const double pr_1, const double pr_2, const double pr_c) {
    const size_t cards_idx = threadIdx.x;
    const size_t card_1 = cards_idx / 2;
    const size_t card_2 = (card_1 + (cards_idx % 2) + 1) % 3;

    if (history.is_terminal())
        return history.terminal_utils(card_1, card_2);

    const size_t is_idx = history.infoset_idx(card_1, card_2);

    Actions actions_utils;
    double util;

    if (history.is_player_1()) {
        atomicAdd(&reach_prs[is_idx], pr_1);
        actions_utils.call = -1. * cfr_rec(reach_prs, strategies, regret_sums, history + 'c', pr_1 * strategies[is_idx].call, pr_2, pr_c);
        actions_utils.bet = -1. * cfr_rec(reach_prs, strategies, regret_sums, history + 'b', pr_1 * strategies[is_idx].bet, pr_2, pr_c);

        util = actions_utils.call * strategies[is_idx].call + actions_utils.bet * strategies[is_idx].bet;

        atomicAdd(&regret_sums[is_idx].call, pr_2 * pr_c * (actions_utils.call - util));
        atomicAdd(&regret_sums[is_idx].bet, pr_2 * pr_c * (actions_utils.bet - util));
    }
    else {
        atomicAdd(&reach_prs[is_idx], pr_2);
        actions_utils.call = -1. * cfr_rec(reach_prs, strategies, regret_sums, history + 'c', pr_1, pr_2 * strategies[is_idx].call, pr_c);
        actions_utils.bet = -1. * cfr_rec(reach_prs, strategies, regret_sums, history + 'b', pr_1, pr_2 * strategies[is_idx].bet, pr_c);

        util = actions_utils.call * strategies[is_idx].call + actions_utils.bet * strategies[is_idx].bet;

        atomicAdd(&regret_sums[is_idx].call, pr_1 * pr_c * (actions_utils.call - util));
        atomicAdd(&regret_sums[is_idx].bet, pr_1 * pr_c * (actions_utils.bet - util));
    }

    return util;
}

__global__
void cfr_enter(double* reach_prs, const Actions* strategies, Actions* regret_sums, double* utils) {
    const size_t idx = threadIdx.x;

    __shared__ double s_reach_prs[N_INFOSETS];
    __shared__ Actions s_strategies[N_INFOSETS];
    __shared__ Actions s_regret_sums[N_INFOSETS];

    if (idx < N_INFOSETS) {
        s_reach_prs[idx] = reach_prs[idx];
        s_strategies[idx] = strategies[idx];
        s_regret_sums[idx] = regret_sums[idx];
    }

    __syncthreads();

    if (idx < N_POSS)
        utils[idx] = cfr_rec(s_reach_prs, s_strategies, s_regret_sums, History(), 1., 1., POSS_PROB);

    __syncthreads();

    if (idx < N_INFOSETS) {
        reach_prs[idx] = s_reach_prs[idx];
        regret_sums[idx] = s_regret_sums[idx];
    }
}

#include "InfoSetMap.cuh"

#include "cuda_runtime.h"

__global__
void fill_strategies(Actions* data) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_INFOSETS) {
        data[idx].call = ACT_PROB;
        data[idx].bet = ACT_PROB;
    }
}

__global__
void next_strategies(const Actions* regret_sums, Actions* strategies, Actions* strategies_sums, double* reach_prs,
    double* reach_prs_sums) {
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

InfoSetMap::InfoSetMap() {
    constexpr size_t v_size = N_INFOSETS * sizeof(Actions);
    constexpr size_t s_size = N_INFOSETS * sizeof(double);

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

void InfoSetMap::next_strategy() const {
    next_strategies<<<1, N_INFOSETS>>>(regret_sums, strategies, strategies_sums, reach_prs, reach_prs_sums);
    cudaDeviceSynchronize();
}

InfoSetMap::~InfoSetMap()
{
    cudaFree(regret_sums);
    cudaFree(strategies);
    cudaFree(strategies_sums);

    cudaFree(reach_prs);
    cudaFree(reach_prs_sums);
}

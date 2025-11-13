#ifndef CFR_KERNELS_CUH
#define CFR_KERNELS_CUH

#include "cuda_runtime.h"

#include <vector>

#include "InfoSetMap.cuh"

double cfr(InfoSetMap& i_map, const size_t n_iterations);

std::vector<double> get_average_strategy(const Actions* strategies_sums, const double* reach_prs_sums);

void display_result(const InfoSetMap& i_map, const double expected_game_value);

#endif //CFR_KERNELS_CUH
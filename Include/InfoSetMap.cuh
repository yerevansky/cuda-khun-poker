#ifndef CUDA_KHUN_POKER_INFOSETMAP_CUH
#define CUDA_KHUN_POKER_INFOSETMAP_CUH

#include "Constants.h"

struct Actions {
    double call = 0.;
    double bet = 0.;
};

struct InfoSetMap {
    Actions* regret_sums{};
    Actions* strategies{};
    Actions* strategies_sums{};

    double* reach_prs{};
    double* reach_prs_sums{};

    InfoSetMap();

    void next_strategy() const;

    ~InfoSetMap();
};

#endif //CUDA_KHUN_POKER_INFOSETMAP_CUH
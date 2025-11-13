#include "InfoSetMap.cuh"
#include "CFR.cuh"

#define N_ITERATIONS 10000

int main() {
    InfoSetMap i_map;

    const double expected_game_value = cfr(i_map, N_ITERATIONS);
    display_result(i_map, expected_game_value);

    return 0;
}
#ifndef CUDA_KHUN_POKER_HISTORY_CUH
#define CUDA_KHUN_POKER_HISTORY_CUH

#include "cuda_runtime.h"

class History {
public:
    static constexpr int MAX_LEN = 3;

    char seq[MAX_LEN]{};
    int len;

    __device__
    History() : seq{}, len(0) {
    }

    __device__
    History operator+(const char action) const {
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
    double terminal_utils(const size_t card_1, const size_t card_2) const {
        const int card_player = (len % 2 == 0) ? card_1 : card_2;
        const int card_opponent = (len % 2 == 0) ? card_2 : card_1;

        if ((len == 3 && seq[0] == 'c' && seq[1] == 'b' && seq[2] == 'c') ||
            (len == 2 && seq[0] == 'b' && seq[1] == 'c')) {
            return 1.0;
            }

        if (len == 2 && seq[0] == 'c' && seq[1] == 'c') {
            return (card_player > card_opponent) ? 1.0 : -1.0;
        }

        return (card_player > card_opponent) ? 2.0 : -2.0;
    }

    __device__
    size_t infoset_idx(const size_t card_1, const size_t card_2) const {
        const size_t card = (len % 2 == 0) ? card_1 : card_2;

        if (len == 0) {
            return card * 4;
        }

        if (len == 1) {
            return card * 4 + 1 + (seq[0] == 'b');
        }

        return card * 4 + 3;
    }

    __device__
    bool is_player_1() const {
        return len % 2 == 0;
    }
};

#endif //CUDA_KHUN_POKER_HISTORY_CUH
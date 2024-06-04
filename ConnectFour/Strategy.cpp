#include "Strategy.h"

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
                           const int lastX, const int lastY, const int noX, const int noY) {
    clock_t start = clock();

    int x = -1, y = -1;
    MCTS mcts(M, N, top, _board, noX, noY);

    if (mcts.needUCT(top, _board)) {
        srand(time(0));
        y = mcts.UCT(start, top, _board);
        x = top[y] - 1;
    } else {
        for (int i = 0; i < N; ++i)
            if (mcts.place[i].second && top[i] > 0) {
                y = i;
                x = top[i] - 1;
            }
    }

    return new Point(x, y);
}

extern "C" void clearPoint(Point *p) {
    delete p;
    return;
}

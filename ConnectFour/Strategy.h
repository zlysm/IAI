#ifndef STRATEGY_H_
#define STRATEGY_H_

#include <ctime>

#include "MCTS.h"
#include "Point.h"

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
                           const int lastX, const int lastY, const int noX, const int noY);

extern "C" void clearPoint(Point *p);

#endif

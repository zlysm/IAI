#ifndef MCTS_H
#define MCTS_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>

#include "Judge.h"

const int MAX_NODE = 1e6;
const double MAX_SEARCH_TIME = 2.85;

using Place = std::pair<bool, bool>;

struct Node {
    int x = 0, y = 0;
    int win = 0, total = 0;
    int team = 0;  // team = 1: AI, 2: human
    bool expandable = true, terminated = true;
    int id = 0, parent = 0;
    std::vector<int> children;
};

class MCTS {
    int M, N;      // row, column
    int noX, noY;  // cannot put here
    int **board, top[15];
    int freeCol;  // free column
    Node *nodes;
    int nodeCnt;

    double calcBelief(int nodeID);
    void move(int y, int team);
    int executeTreePolicy(int nodeID);
    int getWinTeam(int nodeID);
    void initBoard(const int *top, const int *_board);
    void initNodeState(int nodeID);
    int getBestChild(int nodeID);
    int getFinalBestChild(int nodeID);
    int expand(int nodeID);

   public:
    MCTS(const int M, const int N, const int *top, const int *_board, const int noX, const int noY);
    ~MCTS();

    std::vector<Place> place;  // place[i].first: can, place[i].second: cannot
    int UCT(clock_t start, const int *top, const int *_board);
    bool needUCT(const int *top, const int *_board);
};

#endif  // MCTS_H

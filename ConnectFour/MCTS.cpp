#include "MCTS.h"

MCTS::MCTS(const int M, const int N, const int *top, const int *_board,
           const int noX, const int noY) : M(M), N(N), noX(noX), noY(noY) {
    board = new int *[M];
    for (int i = 0; i < M; ++i)
        board[i] = new int[N];
    nodes = new Node[MAX_NODE];
    nodeCnt = 0;
}

MCTS::~MCTS() {
    for (int i = 0; i < M; ++i)
        delete[] board[i];
    delete[] board;
    delete[] nodes;
}

void MCTS::initBoard(const int *top, const int *_board) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            board[i][j] = _board[i * N + j];
    freeCol = 0;
    for (int i = 0; i < N; ++i) {
        this->top[i] = top[i];
        if (top[i] != 0)
            ++freeCol;
    }
}

double MCTS::calcBelief(int nodeID) {
    Node &node = nodes[nodeID];
    return node.win * 1.0 / node.total + sqrt(2 * log(nodes[node.parent].total) / node.total);
}

void MCTS::move(int y, int team) {
    int x = top[y] - 1;
    board[x][y] = team;
    top[y] = (x - 1 == noX && y == noY) ? noX : x;
    if (top[y] == 0)
        --freeCol;
}

void MCTS::initNodeState(int nodeID) {
    Node &node = nodes[nodeID];
    node.win = node.total = 0;
    node.terminated = false;
    node.expandable = (freeCol != 0);
}

int MCTS::getBestChild(int nodeID) {
    Node &node = nodes[nodeID];
    int bestChild = -1;
    double bestBelief = -1;

    for (const auto &childId : node.children) {
        double belief = calcBelief(childId);
        if (belief > bestBelief) {
            bestBelief = belief;
            bestChild = childId;
        }
    }
    return bestChild;
}

int MCTS::getFinalBestChild(int nodeID) {
    Node &node = nodes[nodeID];
    int bestChild = node.children[0];
    double bestRate = -1;

    for (const auto &childID : node.children) {
        if (nodes[childID].terminated)
            return childID;

        bool allChildTerminated = true;
        for (const auto &grandChildID : nodes[childID].children)
            if (nodes[grandChildID].terminated) {
                allChildTerminated = false;
                break;
            }

        if (allChildTerminated) {
            double winRate = nodes[childID].win * 1.0 / nodes[childID].total;
            if (winRate > bestRate) {
                bestRate = winRate;
                bestChild = childID;
            }
        }
    }
    return bestChild;
}

int MCTS::expand(int nodeID) {
    Node &node = nodes[nodeID];
    node.children.push_back(nodeCnt);

    int childNum = node.children.size();
    int startY = (childNum == 1) ? 0 : nodes[node.children[childNum - 2]].y + 1;  // start from the last child's y + 1

    if (nodeID != 0) {
        node.expandable = !(childNum == freeCol);  // if all children are expanded, can't expand anymore
        while (top[startY] == 0) ++startY;         // skip the full column
    } else {
        while (top[startY] == 0 || place[startY].first) ++startY;
        bool expandable = false;
        for (int i = startY + 1; i < N; ++i)  // check if there is a point to expand
            if (top[i] != 0 && !place[i].first) {
                expandable = true;
                break;
            }
        nodes[nodeID].expandable = expandable;
    }

    // init new node
    nodes[nodeCnt].x = top[startY] - 1;
    nodes[nodeCnt].y = startY;
    nodes[nodeCnt].team = 3 - node.team;  // reverse team
    nodes[nodeCnt].parent = nodeID;
    nodes[nodeCnt].id = nodeCnt;

    move(startY, nodes[nodeCnt].team);
    initNodeState(nodeCnt);

    if ((nodes[nodeCnt].team == 1 && userWin(nodes[nodeCnt].x, nodes[nodeCnt].y, M, N, board)) ||
        (nodes[nodeCnt].team == 2 && machineWin(nodes[nodeCnt].x, nodes[nodeCnt].y, M, N, board)) ||
        freeCol == 0) {
        nodes[nodeCnt].terminated = true;
    }
    return nodeCnt++;
}

int MCTS::executeTreePolicy(int nodeID) {
    while (!nodes[nodeID].terminated) {
        if (nodeID != 0)
            move(nodes[nodeID].y, nodes[nodeID].team);
        if (nodes[nodeID].expandable)
            return expand(nodeID);
        else
            nodeID = getBestChild(nodeID);
    }
    return nodeID;
}

int MCTS::getWinTeam(int nodeID) {
    int team = nodes[nodeID].team;
    if (nodes[nodeID].terminated)
        return team;

    while (true) {
        team = 3 - team;  // reverse team
        int randY = rand() % freeCol;
        int newY = 0;
        while (top[newY] == 0) ++newY;
        for (int i = 0; i < randY; ++i)
            while (top[++newY] == 0);

        int newX = top[newY] - 1;
        move(newY, team);
        if ((team == 1 && userWin(newX, newY, M, N, board)) ||
            (team == 2 && machineWin(newX, newY, M, N, board)) ||
            freeCol == 0) {
            return team;
        }
    }
}

int MCTS::UCT(clock_t start, const int *top, const int *_board) {
    initNodeState(0);
    nodes[0].parent = -1;
    nodes[0].team = 1;
    nodeCnt = 1;

    while (clock() - start < MAX_SEARCH_TIME * CLOCKS_PER_SEC && nodeCnt < MAX_NODE) {
        initBoard(top, _board);
        int nodeID = executeTreePolicy(0);
        int result = getWinTeam(nodeID);

        while (nodeID != -1) {
            if (result == nodes[nodeID].team)
                ++nodes[nodeID].win;
            ++nodes[nodeID].total;
            nodeID = nodes[nodeID].parent;
        }
    }
    return nodes[getFinalBestChild(0)].y;
}

bool MCTS::needUCT(const int *top, const int *_board) {
    initBoard(top, _board);
    place = std::vector<Place>(N, std::make_pair(false, false));

    for (int i = 0; i < N; ++i) {
        if (top[i] > 0) {
            board[top[i] - 1][i] = 2;
            if (machineWin(top[i] - 1, i, M, N, board)) {
                place[i].second = true;
                return false;
            }
            board[top[i] - 1][i] = 0;
        }

        if (top[i] > 1) {
            board[top[i] - 2][i] = 2;
            if (machineWin(top[i] - 2, i, M, N, board)) {
                place[i].first = true;
            }
            board[top[i] - 2][i] = 0;
        }
    }

    for (int i = 0; i < N; ++i)
        if (top[i] != 0 && !place[i].first)
            return true;

    for (int i = 0; i < N; ++i)
        place[i].first = false;
    return true;
}

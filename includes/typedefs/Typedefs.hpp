#pragma once


#include <queue>
namespace c_knn {

class cmp {
public:
  bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
    return a.first < b.first;
  }
};

using FeaturesVector = std::vector<float>;

using MinHeap = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, cmp>;

}

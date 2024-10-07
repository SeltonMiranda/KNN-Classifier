#pragma once


#include <queue>
namespace c_knn {

class cmp {
public:
  bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
    return a.first < b.first;
  }
};

using LabelsVec = std::vector<int>;
using FeaturesVec = std::vector<float>;
using MinHeap = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, cmp>;

}

#pragma once

#include <iostream>
#include <memory>
#include <vector>

namespace c_knn {

class IDataHandler {
  public:
    virtual ~IDataHandler() = default;
    virtual void load_sample(const std::string& filename, 
                             std::vector<std::vector<float>>&x,
                             std::vector<int>& y) const = 0;
    virtual void preProcessImageData(const std::string& folder) const = 0;
    virtual const std::vector<std::vector<float>>& getData() const = 0;
    virtual const std::vector<int>&  getLabels() const = 0;
};
}

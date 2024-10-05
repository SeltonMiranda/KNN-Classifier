#pragma once

#include "interfaces/IDataHandler.hpp"

namespace c_knn {
class DataHandler : public IDataHandler {
  public:
    DataHandler() = default;
    virtual ~DataHandler() = default;

    virtual void load_sample(const std::string& filename, 
                             std::vector<std::vector<float>>&x,
                             std::vector<int>& y) const override;

    virtual void preProcessImageData(const std::string& folder) const override;
    virtual const std::vector<std::vector<float>>& getData() const override;
    virtual const std::vector<int>&  getLabels() const override;
};
} 

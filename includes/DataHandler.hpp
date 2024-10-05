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
    virtual void generate_data(const std::string& inputPath, const std::string outputFile,
                               const std::unique_ptr<ILocalBinaryPatterns>& descriptor) const override;
    virtual void save_data(const std::vector<float>&, std::ofstream& filename, int label) const override;  
};
} 

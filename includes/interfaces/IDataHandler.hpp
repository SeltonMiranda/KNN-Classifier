#pragma once

#include "ILocalBinaryPatterns.hpp"
#include "../typedefs/Typedefs.hpp"

#include <iostream>
#include <vector>
#include <memory>

namespace c_knn {

class IDataHandler {
  public:
    virtual ~IDataHandler() = default;
    virtual void load_sample(const std::string& filename, 
                             std::vector<c_knn::FeaturesVec>&x,
                             c_knn::LabelsVec& y) const = 0;
    virtual void preProcessImageData(const std::string& folder) const = 0;
    virtual void generate_data(const std::string& inputPath, const std::string outputFile,
                               const std::unique_ptr<ILocalBinaryPatterns>& descriptor) const = 0;
    virtual void save_data(const c_knn::FeaturesVec&, std::ofstream& filename, int label) const = 0;  
    virtual void copy_directory_structure(const std::string& dir_path) const = 0;
};
}

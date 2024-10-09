#pragma once

#include "interfaces/IDataHandler.hpp"

namespace c_knn {
class DataHandler : public IDataHandler {
  private:
    virtual void copy_directory_structure(const std::string& dir_path) const override;
  public:
    DataHandler() = default;
    virtual ~DataHandler() = default;
    virtual void load_sample(const std::string& filename, 
                             std::vector<c_knn::FeaturesVec>&x,
                             c_knn::LabelsVec& y) const override;
    virtual void preProcessImageData(const std::string& folder, const std::unique_ptr<ICrop>& cropper) const override;
    virtual void generate_data(const std::string& inputPath, const std::string outputFile,
                               const std::unique_ptr<ILocalBinaryPatterns>& descriptor) const override;
    virtual void write_to_csv(const c_knn::FeaturesVec&, std::ofstream& filename, int label) const override;  
};
} 

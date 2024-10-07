#pragma once

#include "./IDataHandler.hpp"
#include "../typedefs/Typedefs.hpp"

#include <vector>
#include <iostream>

namespace c_knn {

class IClassifier {
  public:
    virtual ~IClassifier() = default;
    virtual std::vector<std::vector<int>> confusion_matrix(const c_knn::LabelsVec& classified_labels, 
                                                           const c_knn::LabelsVec& y_test) const = 0;

    virtual std::vector<int> classify(const std::vector<c_knn::FeaturesVec>& x_test) const = 0;
    virtual float accuracy(const std::vector<c_knn::LabelsVec>& confusion_matrix) const = 0;
    virtual void set_sample_train(const std::unique_ptr<IDataHandler>& handler, const std::string& filename) = 0;
    virtual const std::vector<std::vector<float>>& get_X_train() const = 0;
    virtual const std::vector<int>& get_y_train() const = 0;
};
}

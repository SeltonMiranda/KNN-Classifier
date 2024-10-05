#pragma once

#include <iostream>
#include "./IDataHandler.hpp"
#include <vector>

namespace c_knn {

class IClassifier {
  public:
    virtual ~IClassifier() = default;
    virtual std::vector<std::vector<int>> confusion_matrix(std::vector<int> classified_labels, 
                                                           std::vector<int> y_test) const = 0;

    virtual std::vector<int> classify(const std::vector<std::vector<float>>& x_test) const = 0;
    virtual float accuracy(const std::vector<std::vector<int>>& confusion_matrix) const = 0;

    virtual void set_sample_train(const std::unique_ptr<IDataHandler>& handler, const std::string& filename) = 0;
    virtual const std::vector<std::vector<float>>& get_X_train() const = 0;
    virtual const std::vector<int>& get_y_train() const = 0;
};
}

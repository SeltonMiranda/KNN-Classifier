#pragma once

#include <iostream>
#include <vector>

namespace c_knn {

class IClassifier {
  public:
    virtual ~IClassifier() = default;

    virtual void generate_data(const std::string& path, const std::string& filename) const = 0;

    virtual void extract_data_and_labels(const std::string& filename, std::vector<std::vector<float>>& x,
                                         std::vector<int>& y) = 0;

    virtual std::vector<int> classify(const std::vector<std::vector<float>>& x_test) const = 0;

    virtual std::vector<std::vector<int>> confusion_matrix(std::vector<int> classified_labels, 
                                                           std::vector<int> y_test) const = 0;

    virtual float accuracy(const std::vector<std::vector<int>>& confusion_matrix) const = 0;
};
}

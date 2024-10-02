#pragma once

#include <iostream>
#include <vector>

namespace c_knn {

class IClassifier {
  public:
    virtual ~IClassifier() = default;
    virtual void load_sample(const std::string& filename, std::vector<std::vector<float>>& x,
                             std::vector<int>& y) = 0;
    virtual std::vector<std::vector<int>> generate_confusion_matrix(const std::vector<int>& classified_labels,
                                                                    const std::vector<int>& y_test) const = 0;
    virtual void checks_if_data_exists() const = 0;
    virtual void create_cropped_images() = 0;
    virtual void generate_data(const std::string& path, const std::string& filename) const = 0;
    virtual std::vector<int> classify(const std::vector<std::vector<float>>& x_test) const = 0;
    virtual float accuracy(const std::vector<std::vector<int>>& confusion_matrix) const = 0;

    virtual void set_sample_train(const std::string& filename) = 0;
    virtual const std::vector<std::vector<float>>& get_x_train() const = 0;
    virtual const std::vector<int>& get_y_train() const = 0;
  
    virtual void set_sample_test(const std::string& filename) = 0;
    virtual const std::vector<std::vector<float>>& get_x_test() const = 0;
    virtual const std::vector<int>& get_y_test() const = 0;
    
    virtual void set_confusion_matrix(const std::vector<std::vector<int>>& matrix) = 0;
    virtual const std::vector<std::vector<int>>& get_confusion_matrix() const = 0;

    virtual void set_predicted_labels(const std::vector<int>& labels) = 0;
    virtual const std::vector<int>& get_predicted_labels() const = 0;
    
};
}

#include "../includes/KNN.hpp"
#include "../includes/typedefs/Typedefs.hpp"

namespace c_knn {

KNN::KNN(size_t k): k_nearest(k) {}
KNN::KNN(size_t k, const c_knn::FeaturesVec& X_train, const c_knn::LabelsVec& y_train) :
  k_nearest{k}, X_train{X_train}, y_train{y_train} {};

float KNN::euclidean_distance(const c_knn::FeaturesVec& vector_test,
                              const c_knn::FeaturesVec& vector_train) const 
{
  float sum{0.0f};
  for (size_t i = 0; i < vector_test.size(); i++) {
    float diff{vector_test.at(i) - vector_train.at(i)};
    sum += (diff * diff);
  }
  return std::sqrt(sum);
}

void KNN::set_sample_train(const std::unique_ptr<IDataHandler>& handler, const std::string& filename) {
  handler->load_sample(filename, this->X_train, this->y_train);
}

const std::vector<c_knn::FeaturesVec>& KNN::get_X_train() const { return this->X_train; }

const c_knn::LabelsVec& KNN::get_y_train() const { return this->y_train; }

c_knn::LabelsVec KNN::classify(const std::vector<c_knn::FeaturesVec>& x_test) const {
  c_knn::LabelsVec classified_labels;
  for (const auto& feature_vector_test : x_test) {
    c_knn::MinHeap minHeap;

    for (std::size_t i = 0; i < this->X_train.size(); ++i) {
      float distance{this->euclidean_distance(feature_vector_test, this->X_train[i])};
      minHeap.push({distance, i});
      if (minHeap.size() > this->k_nearest) minHeap.pop();
    }

    std::map<int, int> label_count;
    while (!minHeap.empty()) {
      int index{minHeap.top().second};
      int label{this->y_train[index]};
      label_count[label]++;
      minHeap.pop();
    }
    
    int classified_label = (std::max_element(begin(label_count), end(label_count),
                                             [](const std::pair<int, int>& a, const std::pair<int, int>& b)
                                             { return a.second < b.second; }))->first;
    classified_labels.push_back(classified_label);
  }
  return classified_labels;
}

std::vector<c_knn::LabelsVec> KNN::confusion_matrix(const c_knn::LabelsVec& classified_labels, 
                                                             const c_knn::LabelsVec& true_labels) const 
{
  std::vector<c_knn::LabelsVec> matrix{2, c_knn::LabelsVec(2, 0)}; 
  for (size_t i = 0; i < true_labels.size(); i++) {
    int true_label{true_labels[i]};
    int classified_label{classified_labels[i]};
    matrix[true_label][classified_label]++;
  }
  return matrix;
}

float KNN::accuracy(const std::vector<c_knn::LabelsVec>& confusion_matrix) const {
  int correct_classified{0};
  int total{0};

  for (size_t i = 0; i < confusion_matrix.size(); i++)
    correct_classified += confusion_matrix[i][i];

  for (const auto& row: confusion_matrix) {
    for (int val: row) 
    total += val;
  }
  return static_cast<float>(correct_classified) / total;
}

}

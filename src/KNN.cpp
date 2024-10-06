#include "../includes/KNN.hpp"

namespace c_knn {

KNN::KNN(size_t k): k_nearest(k) {}



float KNN::euclidean_distance(const std::vector<float>& vector_test,
                          const std::vector<float>& vector_train) const 
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

const std::vector<std::vector<float>>& KNN::get_X_train() const { return this->X_train; }

const std::vector<int>& KNN::get_y_train() const { return this->y_train; }

std::vector<int> KNN::classify(const std::vector<std::vector<float>>& x_test) const {
  std::vector<int> classified_labels;
  for (const auto& feature_vector_test : x_test) {
    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
      return a.first < b.first;
    };

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> minHeap(cmp);

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

std::vector<std::vector<int>> KNN::confusion_matrix(const std::vector<int>& classified_labels, 
                                                             const std::vector<int>& true_labels) const 
{
  std::vector<std::vector<int>> matrix{2, std::vector<int>(2, 0)}; 
  for (size_t i = 0; i < true_labels.size(); i++) {
    int true_label{true_labels[i]};
    int classified_label{classified_labels[i]};
    matrix[true_label][classified_label]++;
  }
  return matrix;
}

float KNN::accuracy(const std::vector<std::vector<int>>& confusion_matrix) const {
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

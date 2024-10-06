#pragma once

#include "./interfaces/IClassifier.hpp"

namespace c_knn {

class KNN : public IClassifier {
  private:
    size_t k_nearest;
    std::vector<std::vector<float>> X_train;
    std::vector<int> y_train;
    // Calcula a distância (euclidiana) entre dois vetores
    float euclidean_distance(const std::vector<float>& vector_test,
                              const std::vector<float>& vector_train) const;
  public:
   // Constructor
    KNN(size_t k);

    // Destructor
    virtual ~KNN() = default;

    // Classifica o vetor de características teste, retorna um vetor com as classes previstas
    virtual std::vector<int> classify(const std::vector<std::vector<float>>& x_test) const override;
    std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& classified_labels, 
                                                             const std::vector<int>& true_labels) const override;

    // Retorna a acurácia
    virtual float accuracy(const std::vector<std::vector<int>>& confusion_matrix) const override;
    virtual void set_sample_train(const std::unique_ptr<IDataHandler>& handler, const std::string& filename) override;
    virtual const std::vector<std::vector<float>>& get_X_train() const override;
    virtual const std::vector<int>& get_y_train() const override;
};
}

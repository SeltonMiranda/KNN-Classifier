#pragma once

#include "./interfaces/IClassifier.hpp"
#include "typedefs/Typedefs.hpp"

namespace c_knn {

class KNN : public IClassifier {
  private:
    size_t k_nearest;
    std::vector<c_knn::FeaturesVec> X_train;
    c_knn::LabelsVec y_train;
    // Calcula a distância (euclidiana) entre dois vetores
    float euclidean_distance(const c_knn::FeaturesVec& vector_test,
                              const c_knn::FeaturesVec& vector_train) const;
  public:
   // Constructor
    KNN(size_t k);
    KNN(size_t k, const c_knn::FeaturesVec& X_train, const c_knn::LabelsVec& y_train);

    // Destructor
    virtual ~KNN() = default;

    // Classifica o vetor de características teste, retorna um vetor com as classes previstas
    virtual std::vector<int> classify(const std::vector<c_knn::FeaturesVec>& x_test) const override;
    std::vector<std::vector<int>> confusion_matrix(const c_knn::LabelsVec& classified_labels, 
                                                             const c_knn::LabelsVec& true_labels) const override;

    // Retorna a acurácia
    virtual float accuracy(const std::vector<c_knn::LabelsVec>& confusion_matrix) const override;
    virtual void set_sample_train(const std::unique_ptr<IDataHandler>& handler, const std::string& filename) override;
    virtual const std::vector<c_knn::FeaturesVec>& get_X_train() const override;
    virtual const c_knn::LabelsVec& get_y_train() const override;
};
}

#pragma once

#include "./interfaces/IClassifier.hpp"
#include "./interfaces/ILocalBinaryPatterns.hpp"
#include "./interfaces/ICrop.hpp"

namespace c_knn {

class KNN : public IClassifier {
  private:

    // Salva os dados do vetor de características no csv
    void save_data_2_csv(const std::vector<float>&, std::ofstream& filename, int label) const;  

    // Calcula a distância (euclidiana) entre dois vetores
    float euclidean_distance(const std::vector<float>& vector_test,
                              const std::vector<float>& vector_train) const;
  public:
    size_t k_nearest;
    std::unique_ptr<ILocalBinaryPatterns> lbp; 
    std::unique_ptr<ICrop> cropper;

    std::vector<std::vector<float>> x_train, x_test;
    std::vector<int> y_train, y_test, predicted_labels;
    std::vector<std::vector<int>> confusion_matrix;

    // Constructor
    KNN(size_t k, std::unique_ptr<ILocalBinaryPatterns> lbp, std::unique_ptr<ICrop> cropper);

    // Destructor
    virtual ~KNN() = default;

    // Cria um arquivo .csv o qual cada linha é um vetor de características de cada imagem
    // contida no path
    virtual void generate_data(const std::string& path, const std::string& filename) const override;

    // Extrai os dados do .csv passado no parâmetro, na matriz x ficarão os vetores de características
    // e no vetor y, as respectivas classes (0: vazio, 1: ocupado)
    virtual void load_sample(const std::string& filename, std::vector<std::vector<float>>& x,
                                         std::vector<int>& y) override;

    // Classifica o vetor de características teste, retorna um vetor com as classes previstas
    virtual std::vector<int> classify(const std::vector<std::vector<float>>& x_test) const override;

    // Retorna uma matriz de confusão
    virtual std::vector<std::vector<int>> get_confusion_matrix(std::vector<int> classified_labels,
                                                           std::vector<int> y_test) const override;
    
    // Retorna a acurácia
    virtual float accuracy(const std::vector<std::vector<int>>& confusion_matrix) const override;
};
}

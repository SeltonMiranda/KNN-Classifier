#include "../includes/KNN.hpp"
#include "../includes/exception/Exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <ostream>
#include <string>

namespace c_knn {

KNN::KNN(size_t k, std::unique_ptr<ILocalBinaryPatterns> lbp, std::unique_ptr<ICrop> cropper)
  : k_nearest(k), lbp(std::move(lbp)), cropper(std::move(cropper)) {}


void KNN::generate_data(const std::string& path, const std::string& filename) const {
  if (!std::filesystem::exists(path) &&
      std::filesystem::is_directory(path)) {
    throw c_knn::DirectoryException{"Directory " + path + " doesn't exists\n"};
  }

  std::ofstream csv_file{filename};
  if (!csv_file) {
    throw c_knn::FileException{"ERROR: Could not open " + filename + "\n"};
  }

  for (const auto& climate: std::filesystem::directory_iterator(path)) {
    for (const auto& day: std::filesystem::directory_iterator(climate)) {
      for (const auto& vacancy: std::filesystem::directory_iterator(day)) {

        cv::Mat image{cv::imread(vacancy.path().string())};
        if (image.empty()) {
          throw c_knn::ImageException{"Couldn't read " + vacancy.path().string() + "\n"};
        }
        
        cv::Mat histogram{this->lbp->histogram(image)};
        // Converter só para nao colocar um header opencv no .hpp
        std::vector<float> features_vector(histogram.begin<float>(), histogram.end<float>());

        if (vacancy.path().filename() == "occupied") {
          this->save_data(features_vector, csv_file, 1);
        } else {
          this->save_data(features_vector, csv_file, 0);
        }
      }
    }
  }
  csv_file.close();
}

void KNN::save_data(const std::vector<float>& vector, std::ofstream& csv, int label) const {
  std::vector<float>::const_iterator it{vector.begin()};
  for (; it != vector.end(); ++it) {
    csv << *it;
    csv << ",";
  }
  csv << label;
  csv << std::endl;
}  

// Parametros de exemplo
// filename = "PUCPR.csv", x_treino = this->x_treino, y_treino = this->y_treino
void KNN::extract_data_and_labels(const std::string& filename, std::vector<std::vector<float>>& x,
                                  std::vector<int>& y)
{
  std::ifstream csv{filename};
  if (!csv) throw c_knn::FileException{"ERROR: Couldn't open " + filename + "\n"};
  
  std::string row;
  // Pega uma linha do csv
  while (std::getline(csv, row)) {
    std::istringstream rowStream{row};
    std::string value;
    std::vector<float> feature_vector;

    // Pega os valores da linha atual
    while (std::getline(rowStream, value, ',')) 
      feature_vector.push_back(std::stof(value));

    // Retira o ultimo valor o qual é a classe
    int label {static_cast<int>(feature_vector.back())};
    feature_vector.pop_back();

    // Guarda os valores nos vetores
    x.push_back(feature_vector);
    y.push_back(label);
  }
}

float KNN::calculate_dist(const std::vector<float>& vector_test,
                          const std::vector<float>& vector_train) const 
{
  float sum{0.0f};
  for (size_t i = 0; i < vector_test.size(); i++) {
    float diff{vector_test.at(i) - vector_train.at(i)};
    sum += (diff * diff);
  }
  return std::sqrt(sum);
}

std::vector<int> KNN::myArgSort(const std::vector<float>& distances) const {
  std::vector<int> indexes(distances.size());  

  for (size_t i = 0; i < distances.size(); i++) {
    indexes[i] = i;
  }

  // Função lambda
  std::sort(indexes.begin(), indexes.end(),
            [&distances](int i1, int i2) { return distances.at(i1) < distances.at(i2); });

  return indexes;
}

// Retorna um vetor labels de prediçoes
std::vector<int> KNN::classify(const std::vector<std::vector<float>>& x_test) const {
  std::vector<int> labels;

  for (const auto& feature_vector_test: x_test) {
    std::vector<float> distances;

    // Calcula a distancia do vetor de teste com os vetores de treino
    for (const auto& feature_vector_train: this->x_train) {
      distances.push_back(this->calculate_dist(feature_vector_test, feature_vector_train));
    }

    // Ordena as distancias pelos indices
    std::vector<int> indexes{this->myArgSort(distances)};

    // Conta a quantidade dos rótulos nos 3 vizinhos mais próximos
    // first -> rótulo
    // second-> quantidade deste rótulo
    std::map<int, int> label_count;
    for (size_t i = 0; i < this->k_nearest; i++) {
      int label{this->y_train.at(indexes.at(i))};
      label_count[label]++;
    }

    // Encontra o rótulo que mais apareceu
    int classified_label = (std::max_element(label_count.begin(), label_count.end(),
                        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                          return a.second < b.second; }))->first;

    // Insere o rótulo no vetor
    labels.push_back(classified_label);
  }
  return labels;
}

std::vector<std::vector<int>> KNN::confusion_matrix(std::vector<int> classified_labels, std::vector<int> true_labels) const {
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

#include "../includes/KNN.hpp"
#include "../includes/exception/Exceptions.hpp"
#include "../includes/constants/Constants.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <ostream>
#include <queue>
#include <string>

namespace c_knn {

KNN::KNN(size_t k, std::unique_ptr<ILocalBinaryPatterns> lbp, std::unique_ptr<ICrop> cropper)
: k_nearest(k), lbp(std::move(lbp)), cropper(std::move(cropper)) {}


void KNN::create_cropped_images() { this->cropper->makeCrop(this->cropper->getFolder()); }

void KNN::set_sample_train(const std::string& filename) {
  this->load_sample(filename, this->x_train, this->y_train);
}

const std::vector<std::vector<float>>& KNN::get_x_train() const { return this->x_train; }

const std::vector<int>& KNN::get_y_train() const { return this->y_train; }

void KNN::set_sample_test(const std::string& filename) {
  this->load_sample(filename, this->x_test, this->y_test);
}

const std::vector<std::vector<float>>& KNN::get_x_test() const { return this->x_test; } 

const std::vector<int>& KNN::get_y_test() const { return this->y_test; }

void KNN::set_confusion_matrix(const std::vector<std::vector<int>>& matrix) {
  this->confusion_matrix = matrix;
}

const std::vector<std::vector<int>>& KNN::get_confusion_matrix() const {
  return this->confusion_matrix;
}

void KNN::set_predicted_labels(const std::vector<int>& labels) {
  this->predicted_labels = labels;
}

const std::vector<int>& KNN::get_predicted_labels() const {
  return this->predicted_labels;
}

void KNN::generate_data(const std::string& path, const std::string& filename) const {
  if (!std::filesystem::exists(path))
    throw c_knn::DirectoryException{"Directory " + path + " doesn't exists\n"};

  std::ofstream csv_file{filename};
  if (!csv_file.is_open())
    throw c_knn::FileException{"ERROR: Could not open " + filename + "\n"};

  for (const auto& entry: std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file()) {
      cv::Mat image{cv::imread(entry.path().string())};

      if (image.empty())
        throw c_knn::ImageException{"Couldn't read " + entry.path().string() + "\n"};

      cv::Mat histogram{this->lbp->histogram(image)};
      // Converter só para nao colocar um header opencv no .hpp
      std::vector<float> features_vector(histogram.begin<float>(), histogram.end<float>());

      if (entry.path().parent_path().filename().string() == "Occupied") {
        this->save_data_2_csv(features_vector, csv_file, 1);
      } else {
        this->save_data_2_csv(features_vector, csv_file, 0);
      }
    }
  }
  csv_file.close();
}

void KNN::save_data_2_csv(const std::vector<float>& vector, std::ofstream& csv, int label) const {
  std::vector<float>::const_iterator it{vector.begin()};
  for (; it != vector.end(); ++it) {
    csv << *it;
    csv << ",";
  }
  csv << label;
  csv << std::endl;
}  

void KNN::load_sample(const std::string& filename, std::vector<std::vector<float>>& x,
                      std::vector<int>& y) {
  std::ifstream csv{filename};
  if (!csv) throw c_knn::FileException{"ERROR: Couldn't open " + filename + "\n"};

  std::string row;
  // Pega uma linha do csv
  while (std::getline(csv, row)) {
    std::istringstream rowStream{row};
    std::string value;
    std::vector<float> features_vector;

    // Pega os valores da linha atual
    while (std::getline(rowStream, value, ',')) 
      features_vector.push_back(std::stof(value));

    // Retira o ultimo valor o qual é a classe
    int label {static_cast<int>(features_vector.back())};
    features_vector.pop_back();

    // Guarda os valores nos vetores
    x.push_back(features_vector);
    y.push_back(label);
  }
  csv.close();
}

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

// Retorna um vetor labels de prediçoes
std::vector<int> KNN::classify(const std::vector<std::vector<float>>& x_test) const {
  std::vector<int> labels;
  for (const auto& feature_vector_test: x_test) {
    // Função lambda para comparação dos valores na min_heap
    auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
      return a.first < b.first;
    };  

    // Cria uma min_heap para armazenar as "k" distâncias mais próximas
    std::priority_queue<std::pair<float, int>,
    std::vector<std::pair<float, int>>,
    decltype(compare)> min_heap(compare);

    // Calcula a distancia do vetor de teste com os vetores de treino
    for (std::size_t i = 0; i < this->x_train.size(); i++) {
      float distance{this->euclidean_distance(feature_vector_test, this->x_train[i])};
      min_heap.push({distance, i});

      if (min_heap.size() > this->k_nearest) min_heap.pop();
    }

    // Conta a quantidade dos rótulos nos 3 vizinhos mais próximos
    // first -> rótulo
    // second-> quantidade deste rótulo
    std::map<int, int> label_count;
    while (!min_heap.empty()) {
      int index{min_heap.top().second};
      int label{this->y_train.at(index)};
      label_count[label]++;
      min_heap.pop();
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

std::vector<std::vector<int>> KNN::generate_confusion_matrix(const std::vector<int>& classified_labels, 
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

void KNN::checks_if_data_exists() const {
  if (!std::filesystem::exists(c_knn::Constants::PKLOTSEGMENTED_DIR)) {
      this->cropper->makeCrop(this->cropper->getFolder());
    }

    if (!std::filesystem::exists(c_knn::Constants::PUCPR_CSV)) {
      this->generate_data(c_knn::Constants::PUCPR, c_knn::Constants::PUCPR_CSV);
    }

    if (!std::filesystem::exists(c_knn::Constants::UFPR04_CSV)) {
      this->generate_data(c_knn::Constants::UFPR04, c_knn::Constants::UFPR04_CSV);
    }

    //if (!std::filesystem::exists(c_knn::Constants::UFPR05_CSV) {
    //  classifier.generate_data(c_knn::Constants::UFPR05, c_knn::Constants::UFPR05_CSV);
    //}
}

}

#include "../includes/KNN.hpp"
#include "../includes/DataHandler.hpp"
#include "../includes/Descriptor.hpp"
#include <filesystem>

int main() {
  std::unique_ptr<c_knn::IClassifier> knn{std::make_unique<c_knn::KNN>(3)};
  std::unique_ptr<c_knn::IDataHandler> handler{std::make_unique<c_knn::DataHandler>()};
  std::unique_ptr<c_knn::ILocalBinaryPatterns> descriptor{std::make_unique<c_knn::Descriptor>()};
  std::vector<std::vector<float>> X_test;
  std::vector<int> y_test;

  try {
    if (!std::filesystem::exists("./PKLotSegmented"))
      handler->preProcessImageData("./PKLotSegmented");
    
    if (!std::filesystem::exists("pucpr_norm.csv")) 
      handler->generate_data("./PKLotSegmented/PUCPR", "pucpr_norm.csv", descriptor);

    if (!std::filesystem::exists("ufpr04_norm.csv")) 
      handler->generate_data("./PKLotSegmented/UFPR04", "ufpr04_norm.csv", descriptor);

    //if (!std::filesystem::exists("ufpr05_norm.csv")) 
    //  handler->generate_data("./PKLotSegmented/PUCPR", "ufpr05_norm.csv", descriptor);

    knn->set_sample_train(handler, "pucpr_norm.csv");
    handler->load_sample("ufpr04_norm.csv", X_test, y_test); 
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  std::vector<int> predicted_labels{knn->classify(X_test)};
  std::vector<std::vector<int>> matrix{knn->confusion_matrix(predicted_labels, y_test)};
  
  float accuracy{knn->accuracy(matrix)};
  std::cout << "Accuracy = " << accuracy << std::endl;
  
  return 0;
}

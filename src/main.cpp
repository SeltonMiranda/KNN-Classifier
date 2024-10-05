#include "../includes/KNN.hpp"
#include "../includes/DataHandler.hpp"
#include "../includes/Descriptor.hpp"
#include "../includes/exception/Exceptions.hpp"
#include "../includes/constants/Constants.hpp"
#include <filesystem>

int main() {
  std::unique_ptr<c_knn::IClassifier> knn{std::make_unique<c_knn::KNN>(3)};
  std::unique_ptr<c_knn::IDataHandler> handler{std::make_unique<c_knn::DataHandler>()};
  std::unique_ptr<c_knn::ILocalBinaryPatterns> descriptor{std::make_unique<c_knn::Descriptor>()};
  std::vector<std::vector<float>> X_test;
  std::vector<int> y_test;

  try {
    if (!std::filesystem::exists(c_knn::Constants::PKLOTSEGMENTED_DIR)) {
      std::cout << "Processing images..." << std::endl;
      handler->preProcessImageData(c_knn::Constants::PKLOT_DIR);
      std:: cout << "Done!" << std::endl;
    }
    
    if (!std::filesystem::exists(c_knn::Constants::PUCPR_CSV)) {
      std::cout << "Creating pucpr_norm.csv..." << std::endl;
      handler->generate_data(c_knn::Constants::PUCPR, c_knn::Constants::PUCPR_CSV, descriptor);
      std:: cout << "Done!" << std::endl;
    }

    if (!std::filesystem::exists(c_knn::Constants::UFPR04_CSV)) {
      std::cout << "Creating ufpr04_norm.csv..." << std::endl;
      handler->generate_data(c_knn::Constants::UFPR04, c_knn::Constants::UFPR04_CSV, descriptor);
      std:: cout << "Done!" << std::endl;
    }

    //if (!std::filesystem::exists(c_knn::Constants::UFPR05_CSV)) 
    //  handler->generate_data(c_knn::Constants::UFPR05, c_knn::Constants::UFPR05_CSV, descriptor);
    
    knn->set_sample_train(handler, c_knn::Constants::PUCPR_CSV);
    handler->load_sample(c_knn::Constants::UFPR04_CSV, X_test, y_test); 

  } catch (c_knn::FileException& e) {
    std::cout << e.what() << std::endl;
  } catch (c_knn::ImageException& e) {
    std::cout << e.what() << std::endl;
  } catch (c_knn::DirectoryException& e) {
    std::cout << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cout << "Unknown error!" << std::endl;
    std::cout << e.what() << std::endl;
  }

  std::cout << "Classifying testing sample..." << std::endl;
  std::vector<int> predicted_labels{knn->classify(X_test)};
  std::vector<std::vector<int>> matrix{knn->confusion_matrix(predicted_labels, y_test)};
  
  float accuracy{knn->accuracy(matrix)};
  std::cout << "Accuracy = " << accuracy << std::endl;
  
  return 0;
}

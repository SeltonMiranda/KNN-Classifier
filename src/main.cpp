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
  std::vector<c_knn::FeaturesVec> X_test_pucpr, X_test_ufpr04, X_test_ufpr05;
  c_knn::LabelsVec y_test_pucpr, y_test_ufpr04, y_test_ufpr05;

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

    if (!std::filesystem::exists(c_knn::Constants::UFPR05_CSV)) {
      std::cout << "Creating ufpr05_norm.csv..." << std::endl;
      handler->generate_data(c_knn::Constants::UFPR05, c_knn::Constants::UFPR05_CSV, descriptor);
      std:: cout << "Done!" << std::endl;
    }

    //knn->set_sample_train(handler, c_knn::Constants::PUCPR_CSV);
    //handler->load_sample(c_knn::Constants::UFPR04_CSV, X_test_ufpr04, y_test_ufpr04); 

    knn->set_sample_train(handler, "./test/70_subset_shuffled.csv");
    handler->load_sample("./test/30_subset_shuffled.csv", X_test_pucpr, y_test_pucpr);

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

  std::cout << "Classifying -> training: 70% subset X testing: 30% subset" << std::endl;
  c_knn::LabelsVec predicted_labels{knn->classify(X_test_pucpr)};
  std::vector<c_knn::LabelsVec> matrix{knn->confusion_matrix(predicted_labels, y_test_pucpr)};
  float accuracy{knn->accuracy(matrix)};
  std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;;

  //std::cout << "Classifying -> training: UFPR04 X testing: PUCPR" << std::endl;
  //c_knn::LabelsVec predicted_labels{knn->classify(X_test_pucpr)};
  //std::vector<c_knn::LabelsVec> matrix{knn->confusion_matrix(predicted_labels, y_test_pucpr)};
  //float accuracy{knn->accuracy(matrix)};
  //std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;;

  //try {
  //  //handler->load_sample(c_knn::Constants::UFPR05_CSV, X_test, y_test); 
  //  handler->load_sample("./test/ufpr05_norm_subset.csv", X_test_ufpr05, y_test_ufpr05);
  //  std::cout << "Classifying training: UFPR04 X testing: UFPR05" << std::endl;
  //  predicted_labels = knn->classify(X_test_ufpr05);
  //  matrix = knn->confusion_matrix(predicted_labels, y_test_ufpr05);
  //  accuracy = knn->accuracy(matrix);
  //  std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;;
  //} catch (std::exception& e) {
  //  std::cout << e.what() << std::endl;
  //}

  //try {
  //  handler->load_sample("./test/ufpr04_norm_subset.csv", X_test_ufpr04, y_test_ufpr04);
  //  std::cout << "Classifying -> training: UFPR04 X testing: UFPR04" << std::endl;
  //  predicted_labels = knn->classify(X_test_ufpr04);
  //  matrix = knn->confusion_matrix(predicted_labels, y_test_ufpr04);
  //  accuracy = knn->accuracy(matrix);
  //  std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;;
  //} catch (std::exception& e) {
  //  std::cout << e.what() << std::endl;
  //}

  return 0;
}

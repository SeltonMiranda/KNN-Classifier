#include "../includes/KNN.hpp"
#include "../includes/DataHandler.hpp"
#include "../includes/Cropper.hpp"
#include "../includes/Descriptor.hpp"
#include "../includes/exception/Exceptions.hpp"
#include "../includes/constants/Constants.hpp"
#include <filesystem>

int main() {
  std::unique_ptr<c_knn::ICrop> cropper{std::make_unique<c_knn::Cropper>()};
  std::unique_ptr<c_knn::IClassifier> knn{std::make_unique<c_knn::KNN>(3)};
  std::unique_ptr<c_knn::IDataHandler> handler{std::make_unique<c_knn::DataHandler>()};
  std::unique_ptr<c_knn::ILocalBinaryPatterns> descriptor{std::make_unique<c_knn::Descriptor>()};

  std::vector<c_knn::FeaturesVec> X_test;
  c_knn::LabelsVec y_test;

  try {
    if (!std::filesystem::exists(c_knn::Constants::PKLOTSEGMENTED_DIR)) {
      std::cout << "Processing images..." << std::endl;
      handler->preProcessImageData(c_knn::Constants::PKLOT_DIR, cropper);
      std:: cout << "Done!" << std::endl;
    }
    
    if (!std::filesystem::exists("normalized_data.csv")) {
      std::cout << "Creating normalized_data.csv..." << std::endl;
      handler->generate_data(c_knn::Constants::PKLOTSEGMENTED_DIR, "normalized_data.csv", descriptor);
      std:: cout << "Done!" << std::endl;
    }

    knn->set_sample_train(handler, "./test/train_subset.csv");
    handler->load_sample("./test/test_subset.csv", X_test, y_test);
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
  c_knn::LabelsVec predicted_labels{knn->classify(X_test)};
  std::vector<c_knn::LabelsVec> matrix{knn->confusion_matrix(predicted_labels, y_test)};
  float accuracy{knn->accuracy(matrix)};
  std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;;

  return 0;
}

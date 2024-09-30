#include "../includes/factory/LBPFactory.hpp"
#include "../includes/factory/CropperFactory.hpp"
#include "../includes/KNN.hpp"


int main() {
  c_knn::KNN classifier{3, c_knn::LBPFactory::createBasicLBP(), c_knn::CropperFactory::createPKLotCropper("PKLot")};
  try {
    classifier.cropper->makeCrop(classifier.cropper->getFolder());
    classifier.generate_data("./PKLot/PKLotSegmented/PUCPR", "PUCPR_NORM.csv");
    classifier.generate_data("./PKLot/PKLotSegmented/UFPR04", "UFPR04_NORM.csv");
    classifier.generate_data("./PKLot/PKLotSegmented/UFPR05", "UFPR05_NORM.csv");

    std::vector<std::vector<float>> x_test;
    std::vector<std::vector<int>> confusion_matrix;
    std::vector<int> y_test, predicted_labels;
    float accuracy{0};
  
    classifier.extract_data_and_labels("PUCPR_NORM.csv", classifier.x_train, classifier.y_train);
    classifier.extract_data_and_labels("UFPR04_NORM.csv", x_test, y_test);

    predicted_labels = classifier.classify(x_test);
    confusion_matrix = classifier.confusion_matrix(predicted_labels, y_test);

    accuracy = classifier.accuracy(confusion_matrix);

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  
  return 0;
}

#include "../includes/factory/LBPFactory.hpp"
#include "../includes/factory/CropperFactory.hpp"
#include "../includes/KNN.hpp"

#include <filesystem>

int main() {
  c_knn::KNN classifier{3, c_knn::LBPFactory::createBasicLBP(),
                        c_knn::CropperFactory::createPKLotCropper("PKLot/PKLot")};

  try {

    //if (!std::filesystem::exists("/PKLotSegmented")) {
    //  classifier.cropper->makeCrop(classifier.cropper->getFolder());
    //}

    if (!std::filesystem::exists("pucpr_norm.csv")) {
      classifier.generate_data("./PKLotSegmented/PUCPR", "pucpr_norm.csv");
    }

    if (!std::filesystem::exists("ufpr04_norm.csv")) {
      classifier.generate_data("./PKLotSegmented/UFPR04", "ufpr04_norm.csv");
    }

    //if (!std::filesystem::exists("ufpr05_norm.csv")) {
    //  classifier.generate_data("./PKLot/PKLotSegmented/UFPR05", "ufpr05_norm.csv");
    //}

    float accuracy{0};
    classifier.load_sample("pucpr_norm.csv", classifier.x_train, classifier.y_train);
    classifier.load_sample("ufpr04_norm.csv", classifier.x_test, classifier.y_test);

    classifier.predicted_labels = classifier.classify(classifier.x_test);
    classifier.confusion_matrix =
              classifier.get_confusion_matrix(classifier.predicted_labels, classifier.y_test);

    accuracy = classifier.accuracy(classifier.confusion_matrix);
    std::cout << "Accuracy: " << accuracy << std::endl;

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}

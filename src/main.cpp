#include "../includes/factory/LBPFactory.hpp"
#include "../includes/factory/CropperFactory.hpp"
#include "../includes/KNN.hpp"

int main() {
  c_knn::KNN classifier{3, c_knn::LBPFactory::createBasicLBP(),
                        c_knn::CropperFactory::createPKLotCropper("./PKLot/PKLot")};

  try {
    classifier.checks_if_data_exists();

    float accuracy{0};
    classifier.set_sample_train("pucpr_norm.csv");
    classifier.set_sample_test("ufpr04_norm.csv");

    classifier.set_predicted_labels(classifier.classify(classifier.get_x_test()));
    classifier.set_confusion_matrix(classifier.generate_confusion_matrix(classifier.get_predicted_labels(),
                                                                         classifier.get_y_test()));

    accuracy = classifier.accuracy(classifier.get_confusion_matrix());
    std::cout << "Accuracy: " << accuracy << std::endl;

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}

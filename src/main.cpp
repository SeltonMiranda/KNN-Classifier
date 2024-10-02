#include "../includes/factory/LBPFactory.hpp"
#include "../includes/factory/CropperFactory.hpp"
#include "../includes/constants/Constants.hpp"
#include "../includes/KNN.hpp"

int main() {
  c_knn::KNN classifier{3, c_knn::LBPFactory::createBasicLBP(),
          c_knn::CropperFactory::createPKLotCropper(c_knn::Constants::PKLOT_DIR)};

  try {
    classifier.checks_if_data_exists();

    float accuracy{0};
    classifier.set_sample_train(c_knn::Constants::PUCPR_CSV);
    classifier.set_sample_test(c_knn::Constants::UFPR04_CSV);

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

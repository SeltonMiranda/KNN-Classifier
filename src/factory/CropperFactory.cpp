#include "../../includes/factory/CropperFactory.hpp"
#include "../../includes/Cropper.hpp"

namespace c_knn {

std::unique_ptr<ICrop> CropperFactory::createPKLotCropper(const std::string &path) {
  return std::make_unique<Cropper>(path);
}

}

#pragma once

#include <memory>
#include "../interfaces/ICrop.hpp"

namespace c_knn {

class CropperFactory {
  public:
    CropperFactory() = delete;
    ~CropperFactory() = default;
    
    static std::unique_ptr<ICrop> createPKLotCropper(const std::string& path);
};
}

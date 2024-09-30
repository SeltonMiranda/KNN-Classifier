#pragma once

#include "../interfaces/ILocalBinaryPatterns.hpp"

namespace c_knn {

class LBPFactory {
  public:
    virtual ~LBPFactory() = default;
    static std::unique_ptr<ILocalBinaryPatterns> createBasicLBP();
};

}

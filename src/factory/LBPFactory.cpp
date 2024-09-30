#include "../../includes/factory/LBPFactory.hpp"
#include "../../includes/LBP.hpp"

namespace c_knn {

std::unique_ptr<ILocalBinaryPatterns> LBPFactory::createBasicLBP() {
  return std::make_unique<LBP>();
}

}

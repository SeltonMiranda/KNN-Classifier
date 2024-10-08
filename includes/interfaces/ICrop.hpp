#pragma once

#include <iostream>

namespace c_knn {

class ICrop {
  public:
    virtual ~ICrop() = default; 
    virtual void makeCrop(const std::string& path) = 0;
};
}

#pragma once

#include "./interfaces/ILocalBinaryPatterns.hpp"

namespace c_knn {

class Descriptor : public ILocalBinaryPatterns {
  public:
    Descriptor() = default;
    ~Descriptor() = default;
    cv::Mat histogram(const cv::Mat& img) const override;
    cv::Mat applyLBP(const cv::Mat& img) const override;
    cv::Mat normalize(const cv::Mat& histogram) const override;
};
}

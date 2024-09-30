#pragma once

#include "./interfaces/ILocalBinaryPatterns.hpp"

namespace c_knn {
class LBP : public ILocalBinaryPatterns {
  public:
    LBP() = default;
    virtual ~LBP() = default; 
    virtual cv::Mat histogram(const cv::Mat& img) const override;

  protected:
    virtual cv::Mat normalize(const cv::Mat& histogram) const override;
    virtual cv::Mat applyLBP(const cv::Mat& img) const override;
};
}

#pragma once

#include <opencv4/opencv2/opencv.hpp>

namespace c_knn {

class ILocalBinaryPatterns {
  public:
    ~ILocalBinaryPatterns() = default;
    virtual cv::Mat histogram(const cv::Mat& img) const = 0;
    virtual cv::Mat normalize(const cv::Mat& histogram) const = 0;
    virtual cv::Mat applyLBP(const cv::Mat& img) const = 0;
};
}

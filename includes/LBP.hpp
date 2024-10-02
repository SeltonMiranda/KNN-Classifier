#pragma once

#include "./interfaces/ILocalBinaryPatterns.hpp"

namespace c_knn {
class LBP : public ILocalBinaryPatterns {
  public:
    LBP() = default;
    virtual ~LBP() = default; 
    // Gera o histograma a partir da imagem LBP
    virtual cv::Mat histogram(const cv::Mat& img) const override;
    // Normaliza o histograma
    virtual cv::Mat normalize(const cv::Mat& histogram) const override;
    // Aplica o LBP na imagem
    virtual cv::Mat applyLBP(const cv::Mat& img) const override;
};
}

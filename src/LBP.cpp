#include "../includes/LBP.hpp"
#include <opencv2/core/base.hpp>
#include <stdexcept>

namespace c_knn {

cv::Mat LBP::histogram(const cv::Mat& img) const {
  cv::Mat grayImage;
  cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

  if (grayImage.empty()) {
    throw std::invalid_argument{"Imagem nao existe\n"};
  }

  const int histSize{256};
  float range[] = {0.0f, 256.0f};
  const float* histRange = {range};

  cv::Mat hist;
  cv::Mat lbpImage{this->applyLBP(grayImage)};
  cv::calcHist(&lbpImage, 1, nullptr, cv::Mat(), hist, 1,
               &histSize, &histRange, true, false);
  return this->normalize(hist);
}

cv::Mat LBP::applyLBP(const cv::Mat& img) const {
  cv::Mat lbpImage{cv::Mat::zeros(img.rows, img.cols, CV_8UC1)};
  
  for (int i = 1; i < img.rows - 1; i++) {
    for (int j = 1; j < img.cols - 1; j++) {
      uchar center{img.at<uchar>(i, j)};
      unsigned char pattern = 0;
      pattern |= (img.at<uchar>(i - 1, j - 1) > center) << 7;
      pattern |= (img.at<uchar>(i - 1, j + 0) > center) << 6;
      pattern |= (img.at<uchar>(i - 1, j + 1) > center) << 5;
      pattern |= (img.at<uchar>(i + 0, j + 1) > center) << 4;
      pattern |= (img.at<uchar>(i + 1, j + 1) > center) << 3;
      pattern |= (img.at<uchar>(i + 1, j + 0) > center) << 2;
      pattern |= (img.at<uchar>(i + 1, j - 1) > center) << 1;
      pattern |= (img.at<uchar>(i + 0, j - 1) > center) << 0;
      lbpImage.at<uchar>(i, j) = pattern;
    }
  }
  return lbpImage;
}


cv::Mat LBP::normalize(const cv::Mat& histogram) const {
  cv::Mat normalizedHist;
  cv::normalize(histogram, normalizedHist, 0, 1, cv::NORM_MINMAX);
  return normalizedHist;
}

}

#pragma once
#include <stdexcept>

namespace c_knn {

class FileException : public std::runtime_error {
  public:
    explicit FileException(const std::string& message) : 
      std::runtime_error(message) {}
};

class DirectoryException : public std::runtime_error {
  public:
    explicit DirectoryException(const std::string& message) :
      std::runtime_error(message) {}
};

class ImageException : public std::runtime_error {
  public:
    explicit ImageException(const std::string& message) :
      std::runtime_error(message) {}
};

}

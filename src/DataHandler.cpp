#include "../includes/DataHandler.hpp"
#include "../includes/exception/Exceptions.hpp"
#include "../includes/Cropper.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace c_knn {

void DataHandler::load_sample(const std::string& filename, 
                              std::vector<c_knn::FeaturesVec>& x,
                              c_knn::LabelsVec& y) const 
{
  std::ifstream csv{filename};
  if (!csv)
    throw c_knn::FileException{"Error: Couldn´t open " + filename};

  std::string row;
  while (std::getline(csv, row)) {
    std::istringstream rowStream(row);
    std::string value;
    c_knn::FeaturesVec fv;

    while (std::getline(rowStream, value, ','))
      fv.push_back(std::stof(value));

    int label{static_cast<int>(fv.back())};
    fv.pop_back();
    x.push_back(fv);
    y.push_back(label);
  }
  csv.close();
}

void DataHandler::copy_directory_structure(const std::string& dir_path) const {
  if (!std::filesystem::exists(dir_path)) {
    throw c_knn::DirectoryException{"Error: Directory " + dir_path + " doesn't exists"};
  }

  const std::string new_dir{"PKLotSegmented"};
  std::filesystem::create_directory(new_dir);

  for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
    if (entry.is_directory()) {
      const std::string relative_path{std::filesystem::relative(entry.path(), dir_path).string()};
      const std::string new_path{new_dir + "/" + relative_path};
      std::filesystem::create_directory(new_path);
    }
  }
}

void DataHandler::preProcessImageData(const std::string& folder) const {
  std::unique_ptr<ICrop> cropper{std::make_unique<Cropper>()};
  this->copy_directory_structure(folder);
  cropper->makeCrop(folder);
}

void DataHandler::generate_data(const std::string& inputPath, const std::string outputFile,
                                const std::unique_ptr<ILocalBinaryPatterns>& descriptor) const {
  if (!std::filesystem::exists(inputPath)) 
    throw c_knn::DirectoryException{"Directory " + inputPath + " doesn't exists"};

  std::ofstream output{outputFile};
  if (!output) throw c_knn::FileException{"Error: Couldn't open " + outputFile};

  for (const auto& entry : std::filesystem::recursive_directory_iterator(inputPath)) {
    if (entry.is_regular_file()) {
      cv::Mat image{cv::imread(entry.path().string())};
      if (image.empty()) {
        throw c_knn::ImageException{"Couldn't read " + entry.path().string()};
      }

      cv::Mat histogram{descriptor->histogram(image)};
      c_knn::FeaturesVec fv(histogram.begin<float>(), histogram.end<float>());
      if (entry.path().parent_path().filename() == "Occupied")
        this->save_data(fv, output, 1);
      else 
        this->save_data(fv, output, 0);
    }
  }
}

void DataHandler::save_data(const c_knn::FeaturesVec& vector, std::ofstream& filename, int label) const {
  c_knn::FeaturesVec::const_iterator it{begin(vector)};
  for (; it != end(vector); ++it) {
    filename << *it;
    filename << ",";
  }
  filename << label;
  filename << std::endl;
} 

}

#include "../includes/Cropper.hpp"
#include "../includes/exception/Exceptions.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace c_knn {

void Cropper::makeCrop(const std::string& path) {
  if (!std::filesystem::exists(path))
    throw c_knn::DirectoryException{"Could not find directory " + path + "\n"};

  const std::string new_dir{"PKLotSegmented"};
  std::filesystem::create_directory(new_dir);

  std::vector<std::string> xml;
  std::vector<std::string> jpg;
  for (const auto& file : std::filesystem::recursive_directory_iterator(path)) {
    if (file.is_regular_file()) {

      if (file.path().extension().string() == ".xml") xml.push_back(file.path().string());
      else if (file.path().extension().string() == ".jpg") jpg.push_back(file.path().string());

    } else if (file.is_directory()) {

      const std::string relative_path{std::filesystem::relative(file.path(), path).string()};
      const std::string new_path{new_dir + "/" + relative_path};
      std::filesystem::create_directory(new_path);

    }
  }

  std::sort(xml.begin(), xml.end());
  std::sort(jpg.begin(), jpg.end());

  for (size_t i = 0; i < xml.size(); i++) {
    std::cout << "xml: " << xml[i] << " jpg: " << jpg[i] << std::endl;
  }
  exit(1);

  //for (size_t i = 0; i < xml.size(); i++) {
  //  const std::string relative{std::filesystem::relative(xml[i], path).parent_path().string()};
  //  const std::string date_new_path{new_dir + "/" + relative};
  //  const std::string emptyDir{date_new_path + "/Empty"};
  //  const std::string occupiedDir{date_new_path + "/Occupied"};

  //  std::filesystem::create_directory(emptyDir);
  //  std::filesystem::create_directory(occupiedDir);
  //  this->cropImages(jpg[i], xml[i], emptyDir, occupiedDir);
  //}
}

void Cropper::cropImages(const std::string& imgPath, const std::string xmlPath,
                        const std::string& emptyDir, const std::string& occupiedDir) {
  // Abrindo o xml
  std::unique_ptr<tinyxml2::XMLDocument> doc{std::make_unique<tinyxml2::XMLDocument>()};
  if (doc->LoadFile(xmlPath.c_str()) != tinyxml2::XML_SUCCESS)
    throw c_knn::FileException{"Could not open file " + xmlPath + "\n"};
  
  // Abrindo a imagem
  cv::Mat image{cv::imread(imgPath.c_str())};
  if (image.empty())
    throw c_knn::ImageException{"Could not open image " + imgPath + "\n"};

  tinyxml2::XMLElement* parkId{doc->FirstChildElement("parking")};
  tinyxml2::XMLElement* space{parkId->FirstChildElement("space")};

  // Iterando pelo campo "space" do xml
  for (; space != nullptr; space = space->NextSiblingElement("space")) {
    this->rect = this->extractXML(space); // Um nome melhor seria parseXML(), provavelmente
    cv::Mat rotatedImage;
    cv::Mat outputImage;

    // Buscando as coordenadas do retângulo rotacionado
    cv::Point2f rectCenter(rect->center_x, rect->center_y);
    cv::Size2f  rectSize(rect->width, rect->height);
    cv::RotatedRect rotatedRect(rectCenter, rectSize, rect->angle);

    // Recortando, na imagem, o retângulo rotacionado na posição extraída
    cv::Mat rotationMatrix{cv::getRotationMatrix2D(rectCenter, rect->angle, 1.0)};
    cv::warpAffine(image, rotatedImage, rotationMatrix, image.size(), cv::INTER_CUBIC);
    cv::getRectSubPix(rotatedImage, rotatedRect.size, rotatedRect.center, outputImage);
    
    // Nomeia a imagem recortada
    std::ostringstream finalNamestream;
    finalNamestream << std::filesystem::path(imgPath).stem().string() << "#"
                    << std::setw(3) << std::setfill('0') << rect->id << ".jpg";
    const std::string finalName{finalNamestream.str()};

    // Insere no diretório correspondente ao id extraído
    if (rect->occupied == 1) {
      cv::imwrite(occupiedDir + "/" + finalName, outputImage);
      std::cout << occupiedDir + "/" + finalName << " occupied: " << rect->occupied << std::endl;
    } else if (rect->occupied == 0) {
      std::cout << emptyDir + "/" + finalName << " occupied: " << rect->occupied << std::endl;
      cv::imwrite(emptyDir + "/" + finalName, outputImage);
    }
  }
}

std::unique_ptr<RectanglePrototype> Cropper::extractXML(tinyxml2::XMLElement* space) {
  std::unique_ptr<RectanglePrototype> rect{std::make_unique<RectanglePrototype>()};

  tinyxml2::XMLElement* rotatedRect{space->FirstChildElement("rotatedRect")};
  tinyxml2::XMLElement* center{rotatedRect->FirstChildElement("center")};
  tinyxml2::XMLElement* size{rotatedRect->FirstChildElement("size")};
  tinyxml2::XMLElement* angle{rotatedRect->FirstChildElement("angle")};

  // Aquisição dos atributos no xml
  space->QueryIntAttribute("id", &rect->id);
  space->QueryIntAttribute("occupied", &rect->occupied);
  center->QueryIntAttribute("x", &rect->center_x);
  center->QueryIntAttribute("y", &rect->center_y);
  size->QueryIntAttribute("w", &rect->width);
  size->QueryIntAttribute("h", &rect->height);
  angle->QueryIntAttribute("d", &rect->angle);

  // Rotaciona o retângulo de acordo com o artigo  
  if (rect->angle <= -45) {
    rect->angle = 90 - std::abs(rect->angle);
    std::swap(rect->width, rect->height);
  } else {
    rect->angle = std::abs(rect->angle);
  }

  return rect;
}
}

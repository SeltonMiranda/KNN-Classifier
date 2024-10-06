#pragma once

#include "./interfaces/ICrop.hpp"
#include "RectanglePrototype.hpp"
#include <memory>
#include <tinyxml2.h>

namespace c_knn {

class Cropper : public ICrop {
  private:
    std::unique_ptr<RectanglePrototype> rect; 

    // Essa função extrai os dados do xml e retorna uma classe que os contêm.
    // Assim fica mais organizado. 
    std::unique_ptr<RectanglePrototype> extractXML(tinyxml2::XMLElement* space);

    // Corta as imagens das vagas, recebe como parâmetros: caminho da imagem, do xml, e as pastas de destino
    // conforme a classificação da imagem no xml, "empty" ou "occupied".
    void cropImages(const std::string& imgPath, const std::string xmlPath,
                    const std::string& emptyPath, const std::string& occupiedPath);
  public:
    Cropper() = default;
    virtual ~Cropper() = default;

    // Recorta as imagens 
    virtual void makeCrop(const std::string& path) override;
};
}

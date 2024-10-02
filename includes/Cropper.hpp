#pragma once

#include "./interfaces/ICrop.hpp"
#include "RectanglePrototype.hpp"
#include <memory>
#include <tinyxml2.h>

namespace c_knn {

class Cropper : public ICrop {
  private:
    std::string folder;
    std::unique_ptr<RectanglePrototype> rect; 

    // Essa função extrai os dados do xml e retorna uma classe que os contêm.
    // Assim fica mais organizado. 
    std::unique_ptr<RectanglePrototype> extractXML(tinyxml2::XMLElement* space);

    // Corta as imagens das vagas, recebe como parâmetros: caminho da imagem, do xml, e as pastas de destino
    // conforme a classificação da imagem no xml, "empty" ou "occupied".
    void cropImages(const std::string& imgPath, const std::string xmlPath,
                    const std::string& emptyPath, const std::string& occupiedPath);
  public:
    // Construtor
    Cropper(const std::string& inputPath);
    // Destrutor
    virtual ~Cropper() = default;

    // Configura o ambiente para o recorte das imagens na pasta passada como parâmetro
    virtual void makeCrop(const std::string& path) override;

    // Retorna o nome da pasta
    virtual const std::string& getFolder() const override;
};
}

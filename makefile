#Flags de compilacao
CFLAGS=-Wall -Wextra -pedantic -std=c++23 -g
LDFLAGS = $(shell pkg-config --cflags --libs opencv4 tinyxml2)

# Diretorios
SRC_DIR =src
INCLUDE_DIR =includes
BUILD_DIR = build

# Nome do executavel
MAIN=treinamento

# Arquivos fonte
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
SRCS_FACTORY = $(wildcard $(SRC_DIR)/*/*.cpp)

# Arquivos objetos
OBJ_FILES=$(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
OBJ_FACTORY = $(patsubst $(SRC_DIR)/*/%.cpp, $(BUILD_DIR)/%.o, $(SRCS_FACTORY))
OBJS = $(OBJ_FILES) $(OBJ_FACTORY)


all:$(MAIN)

$(MAIN):$(OBJS)
	g++ $(CFLAGS) -o $@ $^ $(LDFLAGS) -g

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	g++ $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@ -g

$(BUILD_DIR)/%.o: $(SRCS_FACTORY)/%.cpp
	@mkdir -p $(dir $@)
	g++ $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@ -g

clean:
	rm -rf $(BUILD_DIR)/*.o $(MAIN) *.o

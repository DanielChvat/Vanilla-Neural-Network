CC = g++

INC_DIR = -Iinclude

SRC = $(wildcard src/*.cpp)

PROJECTNAME = MyProject

OUTPUT_DIR = build

default:
	$(CC) $(INC_DIR) $(SRC) -o $(OUTPUT_DIR)/$(PROJECTNAME) -Wall -g
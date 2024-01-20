# Compiler
CC = gcc

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
# source files
SRCS = $(wildcard $(SRC_DIR)/*.c )
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS) )

# Executable
EXECUTABLE = $(BIN_DIR)/dla_omp

# Config file
CONFIG_H = src/config.h

# Compiler flags
CFLAGS = -c

# Linker flags
LDFLAGS = -lm

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS) -o $@ -fopenmp -lm -O3 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(CONFIG_H)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $< -o $@ -fopenmp -O3 -lm -O3


clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
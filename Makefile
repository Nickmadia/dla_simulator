# Compiler
CC = gcc
NVCC = /opt/cuda/bin/nvcc

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
# source files
SRCS = $(wildcard $(SRC_DIR)/*.c $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS)))

# Executable
EXECUTABLE = $(BIN_DIR)/dla_simulation

# Config file
CONFIG_H = src/config.h

# Compiler flags
CFLAGS = -Wall  -c
NVCCFLAGS = -c

# Linker flags
LDFLAGS = -lm

# Targets
all: print_vars $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS) -ccbin /opt/cuda/bin

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(CONFIG_H)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(CONFIG_H) 
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ -ccbin /opt/cuda/bin 

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
print_vars:
    @echo "SRCS: $(SRCS)"
    @echo "OBJS: $(OBJS)"
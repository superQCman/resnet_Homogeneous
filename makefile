# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/resnet
INTERCHIPLET=../../interchiplet/bin/interchiplet

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLET_C_LIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a

# C/C++ Source files
C_SRCS=$(wildcard *.cpp)
C_OBJS=$(patsubst %.cpp, obj/%.o, $(filter-out resnet.cpp, $(C_SRCS)))
RESNET_OBJ=obj/resnet.o
C_TARGET=bin/resnet_c

all: bin_dir obj_dir C_target resnet_target

# C language target
C_target: $(C_OBJS)
	$(CC) $(C_OBJS) $(INTERCHIPLET_C_LIB) -o $(C_TARGET) -pthread

# resnet.cpp target
resnet_target: $(RESNET_OBJ)
	$(CC) $(RESNET_OBJ) $(INTERCHIPLET_C_LIB) -o bin/resnet -pthread

# Rule for resnet.cpp object
$(RESNET_OBJ): resnet.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for other C objects
obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Directory for binary files.
bin_dir:
	mkdir -p bin

# Directory for object files for C.
obj_dir:
	mkdir -p obj

run:
	$(INTERCHIPLET) resnet.yml

gdb:
	gdb $(C_TARGET)

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj bin
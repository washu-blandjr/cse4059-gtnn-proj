NVCC      := nvcc
CXXFLAGS  := -std=c++17
INCLUDES  := -I.
TARGET    := sim
SRCS      := main.cu kernel.cu
OBJS      := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean

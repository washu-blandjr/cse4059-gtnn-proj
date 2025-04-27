NVCC      := nvcc
CXXFLAGS  := -std=c++17
INCLUDES  := -I.
TARGET1   := sim
TARGET2   := simO1
TARGET3   := simO2
TARGET4   := simO3
SRCS1     := main.cu kernel.cu
SRCS2     := mainO1.cu KernelO1.cu
SRCS3     := mainO2.cu KernelO2.cu
SRCS4     := mainO3.cu KernelO3.cu
OBJS1     := $(SRCS1:.cu=.o)
OBJS2     := $(SRCS2:.cu=.o)
OBJS3     := $(SRCS3:.cu=.o)
OBJS4     := $(SRCS4:.cu=.o)

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): $(OBJS1)
	$(NVCC) $(CXXFLAGS) -o $@ $(OBJS1)

$(TARGET2): $(OBJS2)
	$(NVCC) $(CXXFLAGS) -o $@ $(OBJS2)

$(TARGET3): $(OBJS3)
	$(NVCC) $(CXXFLAGS) -o $@ $(OBJS3)

$(TARGET4): $(OBJS4)
	$(NVCC) $(CXXFLAGS) -o $@ $(OBJS4)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(OBJS1) $(OBJS2) $(OBJS3) $(OBJS4)

.PHONY: all clean

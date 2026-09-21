CXX = g++

CXXFLAGS = -fstack-protector-all -Wl,-z,now -g -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -std=gnu++17 \
           -L$(CANN_INSTALL_PATH)/lib64 \
		   -L$(CANN_INSTALL_PATH)/tools/tikicpulib/lib \
		   -L$(CANN_INSTALL_PATH)/tools/tikicpulib/lib/Ascend910B1 \
		   -L$(CANN_INSTALL_PATH)/tools/tikicpulib/../simulator/Ascend910B1/lib \
		   -L$(CANN_INSTALL_PATH)/tools/tikicpulib/../../lib64 \
		   -Wl,-rpath,$(CANN_INSTALL_PATH)/lib64:$(CANN_INSTALL_PATH)/tools/tikicpulib/lib:$(CANN_INSTALL_PATH)/tools/tikicpulib/lib/Ascend910B1:$(CANN_INSTALL_PATH)/tools/tikicpulib/../simulator/Ascend910B1/lib:$(CANN_INSTALL_PATH)/tools/tikicpulib/../../lib64: -Wl,--no-as-needed $(CANN_INSTALL_PATH)/tools/tikicpulib/lib/libtikicpulib_cceprint.so $(CANN_INSTALL_PATH)/tools/tikicpulib/lib/libtikicpulib_npuchk.so $(CANN_INSTALL_PATH)/tools/tikicpulib/lib/libtikicpulib_stubreg.so -lc_sec -lpem_davinci -ltikcpp_debug -lpthread \
		   -DASCENDC_CPU_DEBUG=1 -DASCENDC_DEBUG=1 -DGTEST_LINKED_AS_SHARED_LIBRARY=1 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CCE_AICORE__=220 -D__CCE_KT_TEST__=1 -D__DAV_C220__ \
           -D__NPU_ARCH__=2201 \
           -I$(CANN_INSTALL_PATH)/compiler/tikcpp/tikcfw/  \
           -I$(CANN_INSTALL_PATH)/compiler/tikcpp/tikcfw/impl/ \
           -I$(CANN_INSTALL_PATH)/compiler/tikcpp/tikcfw/interface/ \
           -I$(CANN_INSTALL_PATH)/tools/tikicpulib/lib/include/ \
           -I$(CANN_INSTALL_PATH)/include/ \
           -I$(CANN_INSTALL_PATH)/pkg_inc/base

TARGET = test_kernel


SOURCES = kernel.cpp tiling_func_asc_graph0_schedule_result0_g0.cpp tiling_func_schedule_group_tail.cpp tiling_func_solver_func.cpp tiling_func_tiling_def_and_tiling_const.cpp main.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -rf $(OBJECTS) $(TARGET) cceprint npuchk simlog stub_reg.log
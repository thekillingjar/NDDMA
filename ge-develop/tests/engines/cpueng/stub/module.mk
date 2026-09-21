LOCAL_PATH := $(call my-dir)

engine_common_stub_files := error_manager_stub.cpp \
							platform_info_stub.cpp \
							cpu_kernel_context_stub.cpp \
							stub.cpp \
							engine/base_engine_stub.cpp \
							optimizer/optimizer_stub.cpp \
							ops_kernel_builder_registry_stub.cpp \
							ops_kernel_info_store/ops_kernel_info_store_stub.cpp \
							util/util_stub.cpp \
                            ../../../../metadef/graph/normal_graph/gnode.cc \
                            ../../../../metadef/graph/type/ascend_string.cc \

engine_common_includes :=   ${LOCAL_PATH} \
							${TOPDIR}aicpu/aicpu_host \
							${TOPDIR}aicpu/aicpu_host/common \
							${TOPDIR}inc/ \
							${TOPDIR}inc/external \
							${TOPDIR}metadef/inc \
                        ${TOPDIR}graphengine/inc \
                        ${TOPDIR}metadef/inc/external \
                        ${TOPDIR}graphengine/inc/external \
                        ${TOPDIR}graphengine/inc/framework \
                            $(TOPDIR)metadef/inc/graph \
                             $(TOPDIR)metadef/inc/graph/utils \
							${TOPDIR}inc/aicpu/cpu_kernels \
							${TOPDIR}inc/external/aicpu \
							proto/task.proto \
							${TOPDIR}common/ \
                            ${TOPDIR}metadef/third_party/graphengine/inc \
                            ${TOPDIR}metadef/ \
                                                        ${TOPDIR}metadef/graph/ \
							${TOPDIR}third_party/json/include \
							${TOPDIR}open_source/json/include \
							${TOPDIR}third_party/protobuf/include \

                          

include $(CLEAR_VARS)
LOCAL_MODULE := libstub_for_aicpu

LOCAL_SRC_FILES := $(engine_common_stub_files)

LOCAL_C_INCLUDES := $(engine_common_includes)

LOCAL_CFLAGS := -std=c++11

LOCAL_STATIC_LIBRARIES :=

include $(BUILD_LLT_SHARED_LIBRARY)

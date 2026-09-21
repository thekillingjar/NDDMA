LOCAL_PATH := $(call my-dir)


local_lib_src_files := engine/base_engine.cc \
                       aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.cc \
                       aicpu_ops_kernel_info_store/kernel_info.cc \
                       aicpu_graph_optimizer/aicpu_graph_optimizer.cc \
                       aicpu_graph_optimizer/optimizer.cc \
                       aicpu_graph_optimizer/graph_optimizer_utils.cc \
                       aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.cc \
                       aicpu_ops_kernel_builder/kernel_builder.cc \
                       config/config_file.cc \
                       config/ops_json_file.cc \
                       error_code/error_code.cc \
                       util/util.cc \
                       config/ops_parallel_rule_json_file.cpp\

local_lib_builder_src_files := aicpu_ops_kernel_builder/kernel_builder.cc \
                              error_code/error_code.cc \
                              util/util.cc \
                              config/config_file.cc \
                              config/ops_json_file.cc \
                              config/ops_parallel_rule_json_file.cpp\

local_lib_inc_path := ${LOCAL_PATH} \
                      ${TOPDIR}metadef/inc \
                      ${TOPDIR}graphengine/inc \
                      ${TOPDIR}inc \
                      ${TOPDIR}inc/external \
                      ${TOPDIR}metadef/inc/external \
                      ${TOPDIR}graphengine/inc/external \
                      ${TOPDIR}graphengine/inc/framework \
                      ${TOPDIR}libc_sec/include \
                      ${TOPDIR}third_party/json/include \
                      ${TOPDIR}open_source/json/include \
                      ${TOPDIR}cann/ops/built-in/op_proto/inc \
                      ${TOPDIR}third_party/protobuf/include \
                      proto/task.proto \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_engine_common

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libgraph \
                          libplatform \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_builder_common

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libc_sec \
                          libgraph \

LOCAL_SRC_FILES := $(local_lib_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################
#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_builder_common

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations
LOCAL_LDFLAGS += -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libc_sec \
                          libgraph \

LOCAL_SRC_FILES := $(local_lib_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_SHARED_LIBRARY}

######################################
#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libaicpu_engine_common

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          libgraph \
                          liberror_manager \
                          libplatform \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################




# add init.conf to host
include $(CLEAR_VARS)

LOCAL_MODULE := init.conf
LOCAL_SRC_FILES := config/init.conf
LOCAL_MODULE_CLASS := ETC
LOCAL_INSTALLED_PATH := $(HOST_OUT_ROOT)/init.conf
include $(BUILD_HOST_PREBUILT)

# add aicpu_ops_parallel_rule.json to host
include $(CLEAR_VARS)

LOCAL_MODULE := aicpu_ops_parallel_rule.json
LOCAL_SRC_FILES := config/aicpu_ops_parallel_rule.json
LOCAL_MODULE_CLASS := ETC
LOCAL_INSTALLED_PATH := $(HOST_OUT_ROOT)/aicpu_ops_parallel_rule.json
include $(BUILD_HOST_PREBUILT)
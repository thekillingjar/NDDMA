LOCAL_PATH := $(call my-dir)


local_lib_src_files := engine/aicpu_engine.cpp \
		               kernel_info/aicpu_kernel_info.cpp \
		               kernel_info/aicpu_cust_kernel_info.cpp \
		               optimizer/aicpu_optimizer.cpp \

local_lib_inc_path := ${LOCAL_PATH} \
                      ${TOPDIR}air/compiler/engines/cpu_engine \
		              ${TOPDIR}air/compiler/engines/cpu_engine/common \
					  ${TOPDIR}metadef/inc \
                      ${TOPDIR}graphengine/inc \
		              ${TOPDIR}inc \
		              ${TOPDIR}inc/external \
					  ${TOPDIR}metadef/inc/external \
                      ${TOPDIR}graphengine/inc/external \
		              ${TOPDIR}graphengine/inc/framework \
		              ${TOPDIR}third_party/json/include \
		              ${TOPDIR}open_source/json/include \
		              proto/task.proto \
		              ${TOPDIR}third_party/protobuf/include \
		              ${TOPDIR}metadef/graph/ \
                      ${TOPDIR}inc/aicpu/cpu_kernels \
                      ${TOPDIR}inc/external/aicpu \
                      ${TOPDIR}libc_sec/include \

local_builder_src_files := kernel_builder/aicpu_kernel_builder.cpp \
						   kernel_builder/aicpu_cust_kernel_builder.cpp \
						   kernel_builder/aicpu_ascend_ops_kernel_builder.cpp \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_ascend_engine

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES := 

LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libgraph \
                          libplatform \
			              libaicpu_engine_common \
			              libascend_protobuf \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}
######################################

#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libaicpu_ascend_engine

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES := 

LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libgraph \
                          libplatform \
			              libascend_protobuf \
		                  atclib/libaicpu_engine_common \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_ascend_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES :=



LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libgraph \
                          libplatform \
						  libaicpu_engine_common \
						  libaicpu_ascend_engine \
						  libascend_protobuf \
						  libregister \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################
#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := acl/libaicpu_ascend_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_UNINSTALLABLE_MODULE := false
LOCAL_STATIC_LIBRARIES :=
                           

LOCAL_SHARED_LIBRARIES := libaicpu_builder_common \
                          libregister \
                          libgraph \
                          libslog \
                          liberror_manager \
						  libascend_protobuf \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}
######################################
#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := acl/libaicpu_ascend_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_UNINSTALLABLE_MODULE := false
LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libslog \
                          libaicpu_builder_common \
                          libregister \
						  libgraph \
                          liberror_manager \
						  libascend_protobuf \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_SHARED_LIBRARY}
######################################

#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libaicpu_ascend_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl


LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libslog \
                          liberror_manager \
                          libgraph \
                          libplatform \
						  libascend_protobuf \
						  libregister \
						  atclib/libaicpu_engine_common \
						  atclib/libaicpu_ascend_engine \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}
######################################

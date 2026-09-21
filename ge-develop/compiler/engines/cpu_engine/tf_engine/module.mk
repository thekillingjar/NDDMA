LOCAL_PATH := $(call my-dir)

local_lib_src_files :=	engine/tf_engine.cc \
                        tf_kernel_info/tf_kernel_info.cc \
						tf_optimizer/tf_optimizer.cc \
						tf_optimizer/tensorflow_util.cc \
						tf_optimizer/tf_optimizer_utils.cc \
						tf_optimizer/tf_function_builder.cc \
						util/tf_util.cc \
						ir2tf/ir2tf_parser_factory.cc \
						ir2tf/ir2tf_base_parser.cc \
						config/ir2tf_json_file.cc \

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
					  ${TOPDIR}libc_sec/include \
					  ${TOPDIR}third_party/json/include \
					  ${TOPDIR}open_source/json/include \
					  ${TOPDIR}cann/ops/built-in/op_proto/inc \
					  ${TOPDIR}third_party/protobuf/include \
					  proto/fwk_adapter.proto \
					  proto/tensorflow/graph.proto \
					  proto/tensorflow/node_def.proto \
					  proto/tensorflow/attr_value.proto \
					  proto/tensorflow/tensor.proto \
					  proto/tensorflow/resource_handle.proto \
					  proto/tensorflow/tensor_shape.proto \
					  proto/tensorflow/types.proto \
					  proto/tensorflow/function.proto \
					  proto/tensorflow/op_def.proto \
					  proto/tensorflow/versions.proto \
					  proto/task.proto \

local_builder_src_files := tf_kernel_builder/tf_kernel_builder.cc \
                           tf_kernel_builder/tf_ops_kernel_builder.cc \

local_builder_acl_src_files := tf_kernel_builder/tf_kernel_builder.cc \
                            	  tf_kernel_builder/tf_ops_kernel_builder.cc \
								  util/tf_util.cc \
								  ir2tf/ir2tf_parser_factory.cc \
								  ir2tf/ir2tf_base_parser.cc \
								  config/ir2tf_json_file.cc \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_tf_engine

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libslog \
						  libgraph \
						  liberror_manager \
						  libruntime \
						  libregister \
						  libplatform \
						  libaicpu_engine_common \
                          libascend_protobuf \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libaicpu_tf_engine

LOCAL_CFLAGS += -DAICPU_PLUGIN -DCOMPILE_OMG_PACKAGE -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
						  libgraph \
						  liberror_manager \
						  libascend_protobuf \
						  libruntime_compile \
						  libregister \
						  libplatform \
						  atclib/libaicpu_engine_common \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

# add ir2tf_op_mapping_lib.pbtxt to host
include $(CLEAR_VARS)

LOCAL_MODULE := ir2tf_op_mapping_lib.json
LOCAL_SRC_FILES := config/ir2tf/ir2tf_op_mapping_lib.json
LOCAL_MODULE_CLASS := ETC
LOCAL_INSTALLED_PATH := $(HOST_OUT_ROOT)/ir2tf_op_mapping_lib.json
include $(BUILD_HOST_PREBUILT)

######################################
#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libaicpu_tf_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libslog \
						  libgraph \
						  liberror_manager \
						  libruntime \
						  libregister \
						  libplatform \
						  libaicpu_engine_common \
						  libaicpu_tf_engine \
                          libascend_protobuf \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/libaicpu_tf_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -DCOMPILE_OMG_PACKAGE -Dgoogle=ascend_private
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
						  libgraph \
						  liberror_manager \
						  libascend_protobuf \
						  libruntime_compile \
						  libregister \
						  libplatform \
						  atclib/libaicpu_engine_common \
						  atclib/libaicpu_tf_engine \

LOCAL_SRC_FILES := $(local_builder_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := acl/libaicpu_tf_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_UNINSTALLABLE_MODULE := false
LOCAL_LDFLAGS += -lrt -ldl

LOCAL_STATIC_LIBRARIES := 

LOCAL_SHARED_LIBRARIES := libaicpu_builder_common \
                          libregister \
						  libgraph \
                          libslog \
						  liberror_manager \
						  libruntime \
                          libascend_protobuf \

LOCAL_SRC_FILES := $(local_builder_acl_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

######################################

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := acl/libaicpu_tf_builder

LOCAL_CFLAGS += -DAICPU_PLUGIN -std=c++11 -Wno-deprecated-declarations -Dgoogle=ascend_private
LOCAL_UNINSTALLABLE_MODULE := false
LOCAL_LDFLAGS += -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libaicpu_builder_common \
                          libregister \
						  libgraph \
                          libslog \
						  liberror_manager \
						  libruntime \
                          libascend_protobuf \

LOCAL_SRC_FILES := $(local_builder_acl_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_SHARED_LIBRARY}

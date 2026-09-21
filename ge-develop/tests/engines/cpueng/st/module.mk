LOCAL_PATH := $(call my-dir)
engine_common_src_path = ../../../../air/compiler/engines/cpu_engine/common/
engine_common_src_files := $(engine_common_src_path)aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.cc \
                           $(engine_common_src_path)aicpu_ops_kernel_info_store/kernel_info.cc \
                           $(engine_common_src_path)aicpu_graph_optimizer/aicpu_graph_optimizer.cc \
                           $(engine_common_src_path)aicpu_graph_optimizer/graph_optimizer_utils.cc \
                           $(engine_common_src_path)aicpu_graph_optimizer/optimizer.cc \
                           $(engine_common_src_path)aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.cc \
                           $(engine_common_src_path)aicpu_ops_kernel_builder/kernel_builder.cc \
                           $(engine_common_src_path)config/config_file.cc \
                           $(engine_common_src_path)config/ops_json_file.cc \
                           $(engine_common_src_path)error_code/error_code.cc \
                           $(engine_common_src_path)util/util.cc \
                           $(engine_common_src_path)engine/base_engine.cc \
                           proto/fwk_adapter.proto \
                           proto/task.proto \

engine_common_src_test_files := st_main.cpp \

engine_common_includes := $(TOPDIR)llt/aicpu/aicpu_host/stub/ \
                          $(TOPDIR)air/compiler/engines/cpu_engine/common/ \
                          $(TOPDIR)air/compiler/engines/cpu_engine/ \
                          ${TOPDIR}inc/ \
                          ${TOPDIR}inc/external \
                          ${TOPDIR}metadef/inc \
                          ${TOPDIR}graphengine/inc \
                          ${TOPDIR}metadef/inc/external \
                          ${TOPDIR}graphengine/inc/external \
                          ${TOPDIR}graphengine/inc/framework \
                          $(TOPDIR)libc_sec/include \
                          ${TOPDIR}third_party/json/include \
                          ${TOPDIR}open_source/json/include \
                          ${TOPDIR}cann/ops/built-in/op_proto/inc \

tf_engine_src_files := ../../../../aicpu/aicpu_host/tf_engine/engine/tf_engine.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/tf_kernel_info/tf_kernel_info.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/tf_optimizer/tf_optimizer.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/tf_optimizer/tf_optimizer_utils.cc \
					   ../../../../aicpu/aicpu_host/tf_engine/tf_optimizer/tensorflow_util.cc \
					   ../../../../aicpu/aicpu_host/tf_engine/tf_optimizer/tf_function_builder.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/tf_kernel_builder/tf_kernel_builder.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/tf_kernel_builder/tf_ops_kernel_builder.cc \
					   ../../../../aicpu/aicpu_host/tf_engine/util/tf_util.cc \
					   ../../../../aicpu/aicpu_host/tf_engine/ir2tf/ir2tf_parser_factory.cc \
					   ../../../../aicpu/aicpu_host/tf_engine/ir2tf/ir2tf_base_parser.cc \
                       ../../../../aicpu/aicpu_host/tf_engine/config/ir2tf_json_file.cc \

tf_engine_src_test_files := tf_engine/tf_engine_st.cpp \
                            tf_engine/tf_kernel_info_st.cpp \
                            tf_engine/tf_optimizer_st.cpp \
                            tf_engine/tf_kernel_builder_st.cpp \
                            tf_engine/tf_ops_kernel_builder_st.cpp \
                            tf_engine/ir2tf_st.cpp \

tf_engine_includes := $(TOPDIR)aicpu/aicpu_host/tf_engine/ \
                      proto/tensorflow/types.proto \
                      proto/tensorflow/graph.proto \
                      proto/tensorflow/node_def.proto \
                      proto/tensorflow/attr_value.proto \
                      proto/tensorflow/tensor.proto \
                      proto/tensorflow/resource_handle.proto \
                      proto/tensorflow/tensor_shape.proto \
                      proto/tensorflow/function.proto \
                      proto/tensorflow/op_def.proto \
                      proto/tensorflow/versions.proto \

cpu_engine_src_files := ../../../../aicpu/aicpu_host/aicpu_engine/engine/aicpu_engine.cpp \
                        ../../../../aicpu/aicpu_host/aicpu_engine/kernel_info/aicpu_kernel_info.cpp \
                        ../../../../aicpu/aicpu_host/aicpu_engine/kernel_info/aicpu_cust_kernel_info.cpp \
		                ../../../../aicpu/aicpu_host/aicpu_engine/optimizer/aicpu_optimizer.cpp \
                        ../../../../aicpu/aicpu_host/aicpu_engine/kernel_builder/aicpu_cust_kernel_builder.cpp \
                        ../../../../aicpu/aicpu_host/aicpu_engine/kernel_builder/aicpu_kernel_builder.cpp \

cpu_engine_includes :=  $(TOPDIR)llt/aicpu/aicpu_host/stub/ \
                        $(TOPDIR)air/compiler/engines/cpu_engine/aicpu_engine/ \
                        ${TOPDIR}air/compiler/engines/cpu_engine \
                        ${TOPDIR}air/compiler/engines/cpu_engine/common \
                        ${TOPDIR}inc \
                        ${TOPDIR}inc/external \
                        ${TOPDIR}metadef/inc \
                        ${TOPDIR}graphengine/inc \
                        ${TOPDIR}metadef/inc/external \
                        ${TOPDIR}graphengine/inc/external \
                        ${TOPDIR}graphengine/inc/framework \
                        ${TOPDIR}third_party/json/include \
                        ${TOPDIR}open_source/json/include \
                        proto/task.proto \
                        ${TOPDIR}third_party/protobuf/include \
                        ${TOPDIR}inc/aicpu/cpu_kernels \
                        ${TOPDIR}inc/external/aicpu \

cpu_engine_src_test_files := aicpu_engine/aicpu_engine_st.cpp \
                             aicpu_engine/aicpu_kernel_info_st.cpp \
                             aicpu_engine/aicpu_cust_kernel_builder_st.cpp \
                             aicpu_engine/aicpu_optimizer_st.cpp \
                             aicpu_engine/aicpu_kernel_builder_st.cpp \

include $(CLEAR_VARS)
LOCAL_MODULE := tf_engine_stest

LOCAL_SRC_FILES := $(engine_common_src_files) + $(engine_common_src_test_files) + $(tf_engine_src_files) + \
                   $(tf_engine_src_test_files)

LOCAL_C_INCLUDES := $(engine_common_includes) + $(tf_engine_includes) + $(cpu_engine_includes)

LOCAL_CFLAGS := -DRUN_TEST -std=c++11 -Dgoogle=ascend_private

LOCAL_SHARED_LIBRARIES := libstub_for_aicpu \
                          libgraph_aicpu_stub \
                          libascend_protobuf \
                          libc_sec \
                          libruntime_aicpu_stub \
                          libslog_aicpu_stub \

LOCAL_STATIC_LIBRARIES :=

LOCAL_CLASSFILE_RULE := aicpu
include $(BUILD_UT_TEST)

include $(CLEAR_VARS)
LOCAL_MODULE := aicpu_engine_stest

LOCAL_SRC_FILES := $(engine_common_src_files) + $(engine_common_src_test_files) + $(cpu_engine_src_files) + $(cpu_engine_src_test_files)

LOCAL_C_INCLUDES := $(engine_common_includes) + $(cpu_engine_includes)

LOCAL_CFLAGS := -DRUN_TEST -std=c++11 -Dgoogle=ascend_private

LOCAL_SHARED_LIBRARIES := libstub_for_aicpu \
                          libgraph_aicpu_stub \
                          libascend_protobuf \
                          libc_sec \
                          libruntime_aicpu_stub \
                          libslog_aicpu_stub \


LOCAL_STATIC_LIBRARIES :=

LOCAL_CLASSFILE_RULE := aicpu
include $(BUILD_UT_TEST)

LOCAL_PATH := $(call my-dir)

local_rts_kernel_files := ops_kernel_store/op/op_factory.cc \
                        ops_kernel_store/op/op.cc \
                        ops_kernel_store/op/send_op.cc \
                        ops_kernel_store/op/recv_op.cc \
                        ops_kernel_store/op/stream_switch_op.cc \
                        ops_kernel_store/op/stream_active_op.cc \
                        ops_kernel_store/op/memcpy_async_op.cc \
                        ops_kernel_store/op/model_exit_op.cc \
                        ops_kernel_store/op/stream_merge_op.cc \
                        ops_kernel_store/op/end_graph_op.cc \
                        ops_kernel_store/op/stream_switchN_op.cc \
                        ops_kernel_store/op/memcpy_addr_async_op.cc \
                        ops_kernel_store/op/cmo_addr_op.cc \
                        ops_kernel_store/op/label_set_op.cc \
                        ops_kernel_store/op/label_switch_op.cc \
                        ops_kernel_store/op/label_goto_op.cc \
                        ops_kernel_store/op/label_switch_by_index_op.cc \
                        ops_kernel_store/op/label_goto_ex_op.cc \

local_lib_src_files :=  ${local_rts_kernel_files} \
                        engine/rts_engine.cc \
                        ops_kernel_store/rts_ops_kernel_info.cc \
                        graph_optimizer/RtsGraphOptimizer.cc \

local_lib_inc_path :=   proto/task.proto \
                        ${LOCAL_PATH} \
                        ${TOPDIR}inc \
                        ${TOPDIR}metadef/inc \
                        ${TOPDIR}inc/external \
                        ${TOPDIR}metadef/inc/external \
                        ${TOPDIR}graphengine/inc/external/ge \
                        ${TOPDIR}graphengine/inc/external \
                        ${TOPDIR}graphengine/inc \
                        ${TOPDIR}parser/inc/enternal/parser \
                        ${TOPDIR}metadef/inc/external/graph \
                        $(TOPDIR)abl/libc_sec/include \
                        ${TOPDIR}third_party/protobuf/include \
                        ${TOPDIR}graphengine/inc/framework \
                        $(TOPDIR)framework/domi \
                        $(TOPDIR)runtime/ \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := librts_engine

LOCAL_CFLAGS += -std=c++11 -Dgoogle=ascend_private
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libgraph \
                            libregister \
                            libascend_protobuf \
                            libmmpa \
                            libc_sec \
                            libslog \
                            libruntime

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}


#compiler for omg
include $(CLEAR_VARS)
LOCAL_MODULE := atclib/librts_engine

LOCAL_CFLAGS += -std=c++11 -Dgoogle=ascend_private
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libmmpa \
                            libascend_protobuf \
                            libc_sec \
                            libslog \
                            libgraph \
                            libregister \
                            libruntime

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

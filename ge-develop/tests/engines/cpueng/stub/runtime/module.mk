LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libruntime_aicpu_stub

LOCAL_C_INCLUDES := \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/cce \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)third_party/glog/include \
    $(TOPDIR)third_party/protobuf/include \

LOCAL_SRC_FILES := \
    src/runtime_stub.cpp \

include $(BUILD_HOST_SHARED_LIBRARY)

#compile x86 libruntime_stub
include $(CLEAR_VARS)

LOCAL_MODULE := libruntime_aicpu_stub

LOCAL_C_INCLUDES := \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/cce \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)framework/domi \
    $(TOPDIR)framework/domi/common \
    $(TOPDIR)third_party/glog/include \
    $(TOPDIR)third_party/protobuf/include \

LOCAL_SRC_FILES := \
    src/runtime_stub.cpp \

include $(BUILD_LLT_SHARED_LIBRARY)

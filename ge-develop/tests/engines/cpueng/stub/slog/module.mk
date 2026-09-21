LOCAL_PATH := $(call my-dir)

#compile x86 libslog_aicpu_stub

include $(CLEAR_VARS)

LOCAL_MODULE := libslog_aicpu_stub

LOCAL_C_INCLUDES := \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \

LOCAL_SRC_FILES := \
    src/slog_stub.cpp \

include $(BUILD_LLT_SHARED_LIBRARY)

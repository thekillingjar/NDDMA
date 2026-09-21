LOCAL_PATH := $(call my-dir)

COMMON_LOCAL_SRC_FILES := \
    ./compress.cpp \
    ./data_compressor.cpp \
    ./log.cpp \
    ./mode_a_index_generator.cpp \
    ./mode_b_index_generator.cpp \

COMMON_LOCAL_C_INCLUDES := \
    metadef/inc/common/util/compress \
    common/utils/compress/inc \
    libc_sec/include \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libcompress

LOCAL_CFLAGS += -g

LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)
#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libcompress

LOCAL_CFLAGS += -g

LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_STATIC_LIBRARY)
#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libcompress

LOCAL_CFLAGS +=

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \


LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)


# compile for ut/st
include $(CLEAR_VARS)
LOCAL_MODULE := libcompress

LOCAL_CFLAGS += -D__CCE_ST_TEST__

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \


LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_LLT_SHARED_LIBRARY)


UTIL_COMMON_LOCAL_SRC_FILES := \
    ./compress_weight.cpp \
    ./log.cpp \

UTIL_COMMON_LOCAL_C_INCLUDES := \
    metadef/inc/common/util/compress \
    common/utils/compress/inc \
    libc_sec/include \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libcompressweight

LOCAL_CFLAGS += -g

LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(UTIL_COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(UTIL_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libcompress   \

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)
#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libcompressweight

LOCAL_CFLAGS += -g

LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(UTIL_COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(UTIL_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libcompress   \

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_STATIC_LIBRARY)
#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libcompressweight

LOCAL_CFLAGS +=

LOCAL_C_INCLUDES := $(UTIL_COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(UTIL_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libcompress   \


LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)


# compile for ut/st
include $(CLEAR_VARS)
LOCAL_MODULE := libcompressweight

LOCAL_CFLAGS += -D__CCE_ST_TEST__

LOCAL_C_INCLUDES := $(UTIL_COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES  := $(UTIL_COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libcompress   \


LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_LLT_SHARED_LIBRARY)

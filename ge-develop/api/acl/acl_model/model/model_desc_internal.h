/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_MODEL_DESC_INTERNAL_H
#define ACL_MODEL_DESC_INTERNAL_H

#include <map>
#include <vector>
#include <string>
#include <set>
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "common/dynamic_aipp.h"
#include "mmpa/mmpa_api.h"
#include "common/ge_common/ge_types.h"

struct aclmdlTensorDesc {
    aclmdlTensorDesc() : name(""), size(0U), format(ACL_FORMAT_UNDEFINED), dataType(ACL_DT_UNDEFINED) {}
    ~aclmdlTensorDesc() = default;
    std::string name;
    size_t size;
    aclFormat format;
    aclDataType dataType;
    std::vector<int64_t> dims;
    std::vector<int64_t> dimsV2; // supported for static aipp scene
    std::vector<std::pair<int64_t, int64_t>> shapeRanges;
};

struct aclmdlDesc {
    void Clear()
    {
        inputDesc.clear();
        outputDesc.clear();
        dynamicBatch.clear();
        dynamicHW.clear();
        dynamicDims.clear();
        dynamicOutputShape.clear();
        dataNameOrder.clear();
        modelId = 0U;
    }
    uint32_t modelId = 0U;
    std::vector<aclmdlTensorDesc> inputDesc;
    std::vector<aclmdlTensorDesc> outputDesc;
    std::vector<uint64_t> dynamicBatch;
    std::vector<std::vector<uint64_t>> dynamicHW;
    std::vector<std::vector<uint64_t>> dynamicDims;
    std::vector<std::vector<int64_t>> dynamicOutputShape;
    std::vector<std::string> dataNameOrder;
    std::map<std::string, std::map<std::string, std::string>> opAttrValueMap;
};

namespace acl {
struct AclModelTensor {
    AclModelTensor(aclDataBuffer *const dataBufIn,
        aclTensorDesc *const tensorDescIn) : dataBuf(dataBufIn), tensorDesc(tensorDescIn)
    {
    }

    ~AclModelTensor() = default;
    aclDataBuffer *dataBuf;
    aclTensorDesc *tensorDesc;
};

enum CceAippInputFormat {
    CCE_YUV420SP_U8 = 1,
    CCE_XRGB8888_U8 = 2,
    CCE_NC1HWC0DI_FP16 = 3,
    CCE_NC1HWC0DI_S8 = 4,
    CCE_RGB888_U8 = 5,
    CCE_ARGB8888_U8 = 6,
    CCE_YUYV_U8 = 7,
    CCE_YUV422SP_U8 = 8,
    CCE_AYUV444_U8 = 9,
    CCE_YUV400_U8 = 10,
    CCE_RAW10 = 11,
    CCE_RAW12 = 12,
    CCE_RAW16 = 13,
    // Hardware needs 15 and aipp component needs to reduce 1, so here it needs to be configured as 16.
    CCE_RAW24 = 16,
    CCE_RESERVED = 17
};

enum AippMode : int32_t {
    UNDEFINED = 0,
    STATIC_AIPP = 1,
    DYNAMIC_AIPP = 2
};

struct BundleSubModelInfo {
  size_t workSize = 0U;
  size_t weightSize = 0U;
  size_t offset = 0U;
  size_t modelSize = 0U;
};
}

struct aclmdlDataset {
    aclmdlDataset()
        : seq(0U),
          modelId(0U),
          timestamp(0U),
          timeout(0U),
          requestId(0U),
          dynamicBatchSize(0U),
          dynamicResolutionHeight(0U),
          dynamicResolutionWidth(0U) {}
    ~aclmdlDataset() = default;
    uint32_t seq;
    uint32_t modelId;
    std::vector<acl::AclModelTensor> blobs;
    uint32_t timestamp;
    uint32_t timeout;
    uint64_t requestId;
    uint64_t dynamicBatchSize;
    uint64_t dynamicResolutionHeight;
    uint64_t dynamicResolutionWidth;
    std::vector<uint64_t> dynamicDims;
};

struct aclmdlAIPP {
    uint64_t batchSize = 0U;
    std::vector<kAippDynamicBatchPara> aippBatchPara;
    kAippDynamicPara aippParms{};
};

struct aclAippExtendInfo {
    bool isAippExtend = false;
};

struct aclmdlConfigHandle {
    aclmdlConfigHandle()
        : priority(0),
          mdlLoadType(0U),
          mdlAddr(nullptr),
          mdlSize(0U),
          workPtr(nullptr),
          workSize(0U),
          weightPtr(nullptr),
          weightSize(0U),
          inputQ(nullptr),
          inputQNum(0U),
          outputQ(nullptr),
          outputQNum(0U),
          reuseZeroCopy(0U),
          withoutGraph(false) {}
    int32_t priority;
    size_t mdlLoadType;
    std::string loadPath;
    void *mdlAddr;
    size_t mdlSize;
    void *workPtr;
    size_t workSize;
    void *weightPtr;
    size_t weightSize;
    const uint32_t *inputQ;
    size_t inputQNum;
    const uint32_t *outputQ;
    size_t outputQNum;
    size_t reuseZeroCopy;
    std::string weightPath;
    bool withoutGraph;
    std::set<aclmdlConfigAttr> attrState;
    std::vector<ge::FileConstantMem> fileConstantMem;
};

struct aclmdlExecConfigHandle {
    int32_t streamSyncTimeout = -1; /**< stream synchronize timeout */
    int32_t eventSyncTimeout = -1; /**< event synchronize timeout */
};

struct aclmdlBundleQueryInfo {
  size_t varSize = 0U;
  std::vector<acl::BundleSubModelInfo> subModelInfos;
};
#endif // ACL_MODEL_DESC_INTERNAL_H

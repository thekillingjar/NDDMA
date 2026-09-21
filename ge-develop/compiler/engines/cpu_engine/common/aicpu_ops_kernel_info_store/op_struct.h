/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_OP_STRUCT_H
#define AICPU_OP_STRUCT_H

#include <map>
#include <string>
#include <vector>
#include "fwk_adpt_struct.h"

namespace aicpu {

struct OpFullInfo {
  std::string engine;        // which engine
  std::string opKernelLib;   // which opsKernelStore
  int computeCost;           // compute cost
  bool flagPartial;          // whether to support is related to shape
  bool flagAsync;            // Whether to support asynchronous
  std::string kernelSo;      // kernel so
  std::string functionName;  // function name
  bool userDefined;          // user defined
  std::map<std::string, std::string> inOutFormat;  // input output format
  std::string opsFlag;  // opsFlag[0] indicates constant folding
  int shapeType;
  std::map<std::string, std::string> inOutDataType;  // input output DataType
  std::map<std::string, std::string> inOutRealName;  // input output name
  bool formatAgnostic;                               // set format agnostic
  int workspaceSize;                                 // workspace size
  std::map<std::string, std::string> castSrcType;
  std::map<std::string, std::string> castDstType;
  FWKAdapter::FWKExtTopicType topicType; // support: DEVICE_ONLY/DEVICE_FIRST/HOST_ONLY/HOST_FIRST
  std::string resource;
  bool noTiling;                         // whether to support no tiling
  std::string slicePattern; // STR_SLICE_PATTERN_VEC
  bool flagSupportBlockDim;
  int blockDimByIndex;
  int implementType;        // host-aicpu support: HOST/DEVCE/ALL
};

struct OpInfoDesc {
  std::string opName;  // op name
  OpFullInfo opInfo;   // engine information that the op
};

struct OpInfoDescs {
  std::vector<OpInfoDesc> opInfos;  // op info
  std::string libName;              // kernel lib name
};
}  // namespace aicpu

#endif
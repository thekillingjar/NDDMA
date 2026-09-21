/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_OPS_H_
#define DVPP_ENGINE_COMMON_DVPP_OPS_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dvpp {
struct InOutInfo {
  std::string dataName;
  std::string dataType;
  std::string dataFormat;
};

struct AttrInfo {
  std::string type;
  std::string value;
};

struct DvppOpInfo {
  std::string engine; // ge need to know, current is DNN_VM_DVPP
  std::string opKernelLib; // ge need to know, current is DVPPKernel
  int32_t computeCost; // ge need to know
  bool flagPartial; // ge need to know
  bool flagAsync; // ge need to know
  std::string kernelSo; // use in 310p, device kernel
  std::string functionName; // use in 310p, current is RunCpuKernel
  bool userDefined; // reserve
  int32_t workspaceSize; // reserve
  std::string resource; // only use in 310p, RES VPC/JPEGD/JPEGE CHANNEL
  std::string memoryType; // 310p: RT_MEMORY_DVPP; 910b: RT_MEMORY_HBM
  int64_t widthMin;
  int64_t widthMax;
  int64_t widthAlign;
  int64_t heightMin;
  int64_t heightMax;
  int64_t heightAlign;
  std::map<std::string, struct InOutInfo> inputs;
  std::map<std::string, struct InOutInfo> outputs;
  std::map<std::string, struct AttrInfo> attrs;
};

struct DvppOp {
  std::string name; // op name
  DvppOpInfo info;
};

struct DvppOpsInfoLib {
  std::string name; // kernel lib name
  std::vector<DvppOp> ops; // ops info
};

// 用于GenerateTask定义 和AicpuParamHead保持一致 为了编译解耦
// 这里把原来对齐方式设置压栈 并设新的对齐方式为1byte对齐 否则device解析出错
#pragma pack(push, 1)
struct DvppParamHead {
  uint32_t length; // total length
  uint32_t ioAddrNum; // inputs and outputs address number
  uint32_t extInfoLength;
  uint64_t extInfoAddr;
};
#pragma pack(pop)
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_OPS_H_

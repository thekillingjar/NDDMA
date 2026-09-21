/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_UTIL_DVPP_CONSTEXPR_H_
#define DVPP_ENGINE_UTIL_DVPP_CONSTEXPR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace dvpp {
// 性能优先标签 打在每个算子op_desc上 使能dvpp engine
const std::string ATTR_NAME_PERFORMANCE_PRIOR = "_performance_prior";
// dvpp私有属性 打在算子上 用于避免重复校验
const std::string ATTR_DVPP_CHECK_RESULT = "_dvpp_check_result";
const std::string ATTR_DVPP_CHECK_UNSUPPORT_REASON =
    "_dvpp_check_unsupport_reason";

const std::string kSocVersionAscend910B1 = "Ascend910B1";
const std::string kSocVersionAscend910B2 = "Ascend910B2";
const std::string kSocVersionAscend910B2C = "Ascend910B2C";
const std::string kSocVersionAscend910B3 = "Ascend910B3";
const std::string kSocVersionAscend910B4 = "Ascend910B4";
const std::string kSocVersionAscend910B41 = "Ascend910B4-1";
const std::string kSocVersionAscend9109391 = "Ascend910_9391";
const std::string kSocVersionAscend9109392 = "Ascend910_9392";
const std::string kSocVersionAscend9109381 = "Ascend910_9381";
const std::string kSocVersionAscend9109382 = "Ascend910_9382";
const std::string kSocVersionAscend9109372 = "Ascend910_9372";
const std::string kSocVersionAscend9109362 = "Ascend910_9362";

const std::vector<std::string> kSocVersionMLR1 = {
  kSocVersionAscend910B1,
  kSocVersionAscend910B2,
  kSocVersionAscend910B2C,
  kSocVersionAscend910B3,
  kSocVersionAscend910B4,
  kSocVersionAscend910B41,
  kSocVersionAscend9109391,
  kSocVersionAscend9109392,
  kSocVersionAscend9109381,
  kSocVersionAscend9109382,
  kSocVersionAscend9109372,
  kSocVersionAscend9109362
};

const std::string kJsonFileAscend910B = "dvpp_ops_info_ascend910b.json";

const std::string kDvppEngineName = "DNN_VM_DVPP";
const std::string kDvppKernelSo = "libdvpp_ops.so";
const std::string kDvppFunctionName = "RunCpuKernel";

const std::string kAiCpuEngineName = "DNN_VM_AICPU";
constexpr uint32_t kAiCpuKernelType = 6; // 310p版本和aicpu一致
const std::string kDvppOpDef = "op_def";

const std::string kDvppGraphOptimizer = "dvpp_graph_optimizer";
const std::string kDvppOpsKernel = "dvpp_ops_kernel";
const std::string kAiCpuDecodeOpsKernel = "aicpu_tf_kernel";
const std::string kNodeData = "Data";
const std::string kNodeNetOutput = "NetOutput";
const std::string kNodeConst = "Const";
const std::string kNodeConstant = "Constant";
const std::string kNodeSub = "Sub";
const std::string kNodeMul = "Mul";
constexpr int64_t kDynamicShape = -2;
constexpr int64_t kDynamicDim = -1;

// new 路径为整体整改修改，旧的路径需要保持兼容
const std::string kDvppOpsJsonFilePathBaseOnEnvPath =
    "/op_impl/built-in/dvpp/dvpp_kernel/config/";
const std::string kDvppOpsJsonFilePathBaseOnEnvPathNew =
    "/built-in/op_impl/dvpp/dvpp_kernel/config/";

#ifdef RUN_TEST
const std::string kDvppOpsJsonFilePath =
    "../../../../../tests/engines/dvppeng/stub/";
const std::string kDvppOpsJsonFilePathNew =
    "../../../../../tests/engines/dvppeng/stub/";
#else // ifdef RUN_TEST
const std::string kDvppOpsJsonFilePath =
    "/usr/local/Ascend/opp/op_impl/built-in/dvpp/dvpp_kernel/config/";
const std::string kDvppOpsJsonFilePathNew =
    "/usr/local/Ascend/opp/built-in/op_impl/dvpp/dvpp_kernel/config/";
#endif // ifdef RUN_TEST

const std::string kOutputMemsizeVector = "output_memsize_vector";
const std::string kOutputAlignmentVector = "output_alignment_vector";
constexpr uint64_t kAlignmentValue = 128;

const std::string kWorkspaceReuseFlag = "workspace_reuse_flag";

const std::string kResourceList = "_resource_list";
const std::string kResourceType = "resource_type";
const std::string kResourceVpcChannel = "RES_VPC_CHANNEL";
const std::string kResourceJpegdChannel = "RES_JPEGD_CHANNEL";
const std::string kResourceJpegeChannel = "RES_JPEGE_CHANNEL";

// use for separate data type
const std::string kComma = ",";
const std::string kTilde = "~";

constexpr uint32_t kNum0 = 0;
constexpr uint32_t kNum1 = 1;
constexpr uint32_t kNum2 = 2;
constexpr uint32_t kNum3 = 3;
constexpr uint32_t kNum4 = 4;
constexpr uint32_t kNum6 = 6;
constexpr uint32_t kNum8 = 8;

constexpr uint32_t kSizeOfFloat16 = 2;
// DT_RESOURCE: ge allocate 8 bytes
constexpr int32_t kSizeOfResource = 8;
// DT_STRING_REF: ge allocate 16 bytes
// store mutex pointer and tensor pointer
constexpr int32_t kSizeOfStringRef = 16;
// DT_STRING: ge allocate 8 bytes
// store string rawdata pointer on device ddr memory
constexpr int32_t kSizeOfString = 16;
constexpr int32_t kSizeOfComplex64 = 8;
constexpr int32_t kSizeOfComplex128 = 16;
constexpr int32_t kSizeOfVariant = 8;

// json file item: struct DvppOpsInfoLib
// DvppOpsInfoLib::name
const std::string kJsonDvppOpsInfoLibName = "libName";
// DvppOpsInfoLib::ops
const std::string kJsonDvppOpsInfoLibOps = "libOps";

// json file item: struct DvppOp
// DvppOp::name
const std::string kJsonDvppOpName = "opName";
// DvppOp::info
const std::string kJsonDvppOpInfo = "opInfo";

// json file item: struct DvppOpInfo
// DvppOpInfo::engine
const std::string kJsonDvppOpInfoEngine = "engine";
// DvppOpInfo::opKernelLib
const std::string kJsonDvppOpInfoOpKernelLib = "opKernelLib";
// DvppOpInfo::computeCost
const std::string kJsonDvppOpInfoComputeCost = "computeCost";
// DvppOpInfo::flagPartial
const std::string kJsonDvppOpInfoFlagPartial = "flagPartial";
// DvppOpInfo::flagAsync
const std::string kJsonDvppOpInfoFlagAsync = "flagAsync";
// DvppOpInfo::kernelSo
const std::string kJsonDvppOpInfoKernelSo = "kernelSo";
// DvppOpInfo::functionName
const std::string kJsonDvppOpInfoFunctionName = "functionName";
// DvppOpInfo::userDefined
const std::string kJsonDvppOpInfoUserDefined = "userDefined";
// DvppOpInfo::workspaceSize
const std::string kJsonDvppOpInfoWorkspaceSize = "workspaceSize";
// DvppOpInfo::resource
const std::string kJsonDvppOpInfoResource = "resource";
// DvppOpInfo::memoryType
const std::string kJsonDvppOpInfoMemoryType = "memoryType";
// DvppOpInfo::widthMin
const std::string kJsonDvppOpInfoWidthMin = "widthMin";
// DvppOpInfo::widthMax
const std::string kJsonDvppOpInfoWidthMax = "widthMax";
// DvppOpInfo::widthAlign
const std::string kJsonDvppOpInfoWidthAlign = "widthAlign";
// DvppOpInfo::heightMin
const std::string kJsonDvppOpInfoHeightMin = "heightMin";
// DvppOpInfo::heightMax
const std::string kJsonDvppOpInfoHeightMax = "heightMax";
// DvppOpInfo::heightAlign
const std::string kJsonDvppOpInfoHeightAlign = "heightAlign";
// DvppOpInfo::inputs
const std::string kJsonDvppOpInfoInputs = "inputs";
// DvppOpInfo::outputs
const std::string kJsonDvppOpInfoOutputs = "outputs";
// DvppOpInfo::attrs
const std::string kJsonDvppOpInfoAttrs = "attrs";

// json file item: struct InOutInfo
// InOutInfo::dataName
const std::string kJsonInOutInfoDataName = "dataName";
// InOutInfo::dataType
const std::string kJsonInOutInfoDataType = "dataType";
// InOutInfo::dataFormat
const std::string kJsonInOutInfoDataFormat = "dataFormat";

// json file item: struct AttrInfo
// AttrInfo::type
const std::string kJsonAttrInfoType = "type";
// AttrInfo::value
const std::string kJsonAttrInfoValue = "value";

// json string
const std::string kJsonStrInput = "input";
const std::string kJsonStrOutput = "output";
const std::string kJsonStrAttr = "attr";
const std::string kJsonStrType = "type";
const std::string kJsonStrFormat = "format";
const std::string kJsonStrName = "name";

// dvpp op type
const std::string kDvppOpDecodeAndCropJpeg = "DecodeAndCropJpeg";
const std::string kDvppOpDecodeJpeg = "DecodeJpeg";
const std::string kDvppOpDecodeJpegPre = "DecodeJpegPre";
const std::string kDvppOpNormalizeV2 = "NormalizeV2";
const std::string kDvppOpYUVToRGB = "YUVToRGB";
const std::string kDvppOpIf = "If";
const std::string kDvppOpAdjustContrastWithMean = "AdjustContrastWithMean";
const std::string kDvppOpRgbToGrayscale = "RgbToGrayscale";
const std::string kDvppOpAdjustContrast = "AdjustContrast";
const std::string kAICpuOpReduceMean = "ReduceMean";
} // namespace dvpp

#endif // DVPP_ENGINE_UTIL_DVPP_CONSTEXPR_H_

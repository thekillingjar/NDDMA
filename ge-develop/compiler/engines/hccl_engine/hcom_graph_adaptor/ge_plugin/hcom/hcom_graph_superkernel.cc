/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_graph_superkernel.h"
#include "register/hidden_inputs_func_registry.h"
#include "register/op_tiling_info.h"
#include "hcom/hcom_topo_info.h"
#include "hccl/hcom.h"
#include "hcom_op_utils.h"
#include "hcom_plugin.h"
#include "graph/debug/ge_attr_define.h"
#include "hcom_graph_optimizer.h"

namespace hccl {
/* superkernel temp code start */
bool GetAddrFromDesc(const ge::OpDescPtr &op, int64_t &inputAddr, int64_t &outputAddr, int64_t &workSpaceAddr) {
  bool bRet = false;
  if (ge::AttrUtils::HasAttr(op, "_skn_hcom_input_addr")) {
    bRet = ge::AttrUtils::GetInt(op, "_skn_hcom_input_addr", inputAddr);
    if (!bRet) {
      return false;
    }
  }
  if (ge::AttrUtils::HasAttr(op, "_skn_hcom_output_addr")) {
    bRet = ge::AttrUtils::GetInt(op, "_skn_hcom_output_addr", outputAddr);
    if (!bRet) {
      return false;
    }
  }
  if (ge::AttrUtils::HasAttr(op, "_skn_hcom_ws_addr")) {
    bRet = ge::AttrUtils::GetInt(op, "_skn_hcom_ws_addr", workSpaceAddr);
    if (!bRet) {
      return false;
    }
  }
  return bRet;
}

bool GetGraphIdFromDesc(const ge::OpDescPtr &op, int32_t &graphId) {
  bool bRet = false;
  if (ge::AttrUtils::HasAttr(op, "hcom_graph_id")) {
    bRet = ge::AttrUtils::GetInt(op, "hcom_graph_id", graphId);
    if (!bRet) {
      return false;
    }
  }
  return bRet;
}

ge::graphStatus HcomGetSuperKernelHiddenInputs(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts) {
  HCCL_INFO("[HcomGetSuperKernelHiddenInputs]Hcomm create com resource of op %s.", opdesc->GetName().c_str());

  HcomOpsKernelInfoStorePtr hcomOpsKernelInfoStoreInfoPtr;
  hccl::HcomPlugin::Instance().GetOpsKernelInfoPtr(hcomOpsKernelInfoStoreInfoPtr);

  std::string group;
  ge::AttrUtils::GetStr(opdesc, "group", group);
  // 获取clearEnable
  bool clearEnable = false;
  int graphId = 0;
  GetGraphIdFromDesc(opdesc, graphId);

  if (hcomOpsKernelInfoStoreInfoPtr->UpdateGraphIdGroupMap(group, graphId)) {
    clearEnable = true;
  }

  HcclDataType dataType;
  std::string sCollectiveType = opdesc->GetType();
  int64_t comm = 0;
  char cTag[CCL_OP_TAG_MAX_LEN];
  CHK_RET(HcomGenerateCclOpTag(sCollectiveType.c_str(), comm, group.c_str(), cTag));

  // get datatype
  CHK_RET(HcomOpUtils::ConversionOpDataType(opdesc, sCollectiveType, dataType));
  // get count
  u64 count = 0;
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  CHK_RET(HcomOpUtils::GetCountFromOpDescSuperkernel(opdesc, sCollectiveType, dataType, count, rankSize));

  // get optype
  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  HcclCMDType opType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;
  // get algName
  int64_t input;
  int64_t output;
  int64_t workspace;
  // get inputPtr, outputPtr
  GetAddrFromDesc(opdesc, input, output, workspace);
  std::vector<rtStream_t> hcclStreamList;
  if (opdesc->GetWorkspaceBytes().size() != 0) {
    CHK_RET(SetWorkspaceResource(cTag, group.c_str(), hcclStreamList, reinterpret_cast<void *>(workspace),
                                 opdesc->GetWorkspaceBytes()[0]));
  }

  u32 aivCoreLimit = 0;
  CHK_RET(HcomOpUtils::GetAivCoreLimit(opdesc, sCollectiveType, aivCoreLimit));

  // get algRes, commContext, len
  u64 len;
  void *context = nullptr;
  HCCL_INFO("SPK input, %p, output %p", reinterpret_cast<void *>(input), reinterpret_cast<void *>(output));
  HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;
  if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
    CHK_RET(HcomOpUtils::GetReduction(opdesc, reduction));
  }
  CHK_RET(HcomGetAlgExecParam(cTag, group.c_str(), count, reinterpret_cast<void *>(input),
                              reinterpret_cast<void *>(output), opType, clearEnable, dataType, reduction, &context,
                              &len, aivCoreLimit));
  HCCL_INFO("SPK context %p", context);
  contexts.emplace_back(context);

  return ge::GRAPH_SUCCESS;
}

REG_HIDDEN_INPUTS_FUNC(ge::HiddenInputsType::HCCLSUPERKERNEL, HcomCreateSuperkernelResource);
ge::graphStatus HcomCreateSuperkernelResource(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts) {
  HCCL_INFO("[HcomCreateSuperkernelResource]Hcom create com resource of op %s.", opdesc->GetName().c_str());
  ge::graphStatus gRet = ge::GRAPH_FAILED;
  std::string sCollectiveType = opdesc->GetType();

  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  HcclCMDType opType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;
  if (opType == HcclCMDType::HCCL_CMD_INVALID) {
    HCCL_ERROR(
        "[HcomCreateSuperkernelResource]Try get superkernel hidden input fail, opType is invalid, sCollectiveType[%s].",
        sCollectiveType.c_str());
  } else {
    HCCL_INFO("Select HcomGetSuperKernelHiddenInputs.");
    gRet = HcomGetSuperKernelHiddenInputs(opdesc, contexts);
  }
  return gRet;
}
}  // namespace hccl
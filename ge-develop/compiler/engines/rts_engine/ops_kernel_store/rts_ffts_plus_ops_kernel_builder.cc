/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rts_ffts_plus_ops_kernel_builder.h"
#include <string>
#include "register/ops_kernel_builder_registry.h"

#include "common/constant/constant.h"
#include "common/util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "op/op_ffts_plus_factory.h"
#include "proto/task.pb.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {

constexpr uint64_t RT_MEMCPYASYNC_SPLIT_SIZE = 67108864UL;  // 64*1024*1024

#ifdef WIN32
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

constexpr uint32_t RT_GENERAL_SQE_NUM = 3U;
REGISTER_OPS_KERNEL_BUILDER(RTS_FFTS_PLUS_OP_KERNEL_LIB_NAME, RtsFftsPlusOpsKernelBuilder);

Status RtsFftsPlusOpsKernelBuilder::CalcOpRunningParam(Node &geNode) {
  bool isNeedCalc = false;
  auto ret = IsNeedCalcOpRunningParam(geNode, isNeedCalc);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!isNeedCalc) {
    return SUCCESS;
  }
  OpDescPtr opDesc = geNode.GetOpDesc();
  const string nodeName = geNode.GetName();
  const string nodeType = geNode.GetType();
  size_t outputSize = opDesc->GetOutputsSize();
  RTS_LOGI("[%s: %s] ffts plus output size=%zu.", nodeName.c_str(), nodeType.c_str(), outputSize);

  int64_t nodeSqeNum = 0;
  for (size_t i = 0; i < outputSize; ++i) {
    int64_t outputMemSize = 0;
    GeTensorDesc outputTensor = opDesc->GetOutputDesc(i);
    GeShape outputShape = outputTensor.GetShape();
    Format format = outputTensor.GetFormat();
    DataType dataType = outputTensor.GetDataType();
    graphStatus grhStatus = ge::TensorUtils::GetTensorMemorySizeInBytes(outputTensor, outputMemSize);
    nodeSqeNum += outputMemSize / RT_MEMCPYASYNC_SPLIT_SIZE;
    if (grhStatus != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Get ffts plus op[%s: %s] out[%zu] memory size failed, "
          "format=%s, dataType=%s, retCode=%#x.",
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), grhStatus);
      return FAILED;
    }
    if (outputMemSize < 0) {
      RTS_REPORT_CALL_ERROR(
          "Got ffts plus op[%s:%s] out[%zu] memory size is negative(not support), format=%s, dataType=%s, "
          "outputMemSize=%" PRId64,
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), outputMemSize);
      return FAILED;
    }
    RTS_LOGI("Calc ffts plus op[%s: %s] out[%zu] mem size is %" PRId64 " format=%s, dataType=%s.", nodeName.c_str(),
             nodeType.c_str(), i, outputMemSize, TypeUtils::FormatToAscendString(format).GetString(),
             TypeUtils::DataTypeToAscendString(dataType).GetString());
    TensorUtils::SetSize(outputTensor, outputMemSize);
    grhStatus = opDesc->UpdateOutputDesc(i, outputTensor);
    if (grhStatus != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Update ffts plus op[%s: %s] out[%zu] description failed, format=%s, dataType=%s, retCode=%#x.",
          nodeName.c_str(), nodeType.c_str(), i, TypeUtils::FormatToAscendString(format).GetString(),
          TypeUtils::DataTypeToAscendString(dataType).GetString(), grhStatus);
      return FAILED;
    }
  }

  if (nodeType == "MemcpyAsync") {
    size_t setSqeNumRes = AttrUtils::SetInt(opDesc, ATTR_NAME_NODE_SQE_NUM,
                                            (nodeSqeNum > RT_GENERAL_SQE_NUM) ? nodeSqeNum : RT_GENERAL_SQE_NUM);
    RTS_LOGD("setSqeNumRes is %zu, nodeSqeNum is %lld.", setSqeNumRes, nodeSqeNum);
  }
  RTS_LOGD("Calc ffts plus op[%s: %s] running param success.", nodeName.c_str(), nodeType.c_str());
  return SUCCESS;
}

Status RtsFftsPlusOpsKernelBuilder::Initialize(const map<string, string> &options) {
  (void)options;

  return IsSupportFftsPlus(isSupportFftsPlus_);
}

Status RtsFftsPlusOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status RtsFftsPlusOpsKernelBuilder::GenerateTask(const Node &geNode, RunContext &context, vector<TaskDef> &tasks) {
  if (!isSupportFftsPlus_) {
    RTS_REPORT_CALL_ERROR("op:%s does not support ffts plus, does not need to generate task.",
                          geNode.GetName().c_str());
    return FAILED;
  }

  string name = geNode.GetName();
  string type = geNode.GetType();

  RTS_LOGI("Generate task start, node:%s, node type:%s, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());

  auto op = OpFftsPlusFactory::Instance().CreateOp(geNode, context);
  if (op == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op for node:%s, node type:%s failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  Status ret = op->Init();
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Node:%s, node type:%s op init failed, retCode=%#x", name.c_str(), type.c_str(), ret);
    return ret;
  }

  bool hasSgtSliceInfo = geNode.GetOpDesc()->HasAttr("_ffts_plus");
  if (hasSgtSliceInfo) {
    RTS_LOGI("Generate ffts+ context def start.");
    ret = op->GenerateCtxDef(geNode);
    if (ret != SUCCESS) {
      RTS_LOGI("Generate ffts+ context def failed.");
      return ret;
    }
    return SUCCESS;
  }

  ret = op->Run(tasks);
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Node:%s, node type:%s op run failed, retCode=%#x", name.c_str(), type.c_str(), ret);
    return ret;
  }

  RTS_LOGD("Generate task end, node:%s, node type:%s, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());
  return ret;
}

Status RtsFftsPlusOpsKernelBuilder::UpdateTask(const Node &geNode, vector<TaskDef> &tasks) {
  string name = geNode.GetName();
  string type = geNode.GetType();
  RunContext runCtx;  // not used
  auto op = OpFftsPlusFactory::Instance().CreateOp(geNode, runCtx);
  if (op == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op for node:%s, node type:%s failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  return op->UpdateTaskDef(tasks);
}

}  // namespace runtime
}  // namespace cce

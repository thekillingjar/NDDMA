/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_graph_optimizer.h"
#include <string>
#include "securec.h"
#include "common/util.h"

#include "common/constant/constant.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "external/ge/ge_api_types.h"
#include "ops_kernel_store/op/op_factory.h"
#include "proto/task.pb.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/util/log.h"
#define RT_ERROR_INVALID_VALUE 0x07110001
const uint32_t RT_MEMORY_DEFAULT = 0x0U;
const uint32_t RT_MEMORY_TS = 0x40U;

using namespace ge;
namespace cce {
namespace runtime {
using domi::TaskDef;
using std::string;
using std::vector;

const std::vector<std::string> RTS_SUPPORTED_OP_TYPE = {
    "MemcpyAsync",
    "MemcpyAddrAsync",
    "Cmo",
};

const std::map<std::string, uint32_t> RTS_SUPPORTED_CONDITION_OP_TYPE = {
    {"StreamSwitchN", 1}, {"StreamSwitch", 2}, {"LabelSwitchByIndex", 1}};

const std::vector<std::string> MEM_TYPE_RANGE_NODES = {"MemcpyAddrAsync", "LabelSwitchByIndex", "LabelGoto",
                                                       "LabelGotoEx", "StreamSwitchN"};
RtsGraphOptimizer::RtsGraphOptimizer() {}

RtsGraphOptimizer::~RtsGraphOptimizer() {}

ge::Status RtsGraphOptimizer::Initialize(const map<std::string, std::string> &options,
                                         ge::OptimizeUtility *const optimizeUtility) {
  (void)options;
  (void)optimizeUtility;
  // do nothing
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::Finalize() {
  // do nothing
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::OptimizeGraphPrepare(ComputeGraph &graph) {
  RTS_LOGI("start rts graph optimizer prepare");
  TraceOwnerGuard guard("RTS", "OptPrepare", graph.GetName());
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      RTS_LOGW("OptimizeGraphPrepare: null node exits");
      continue;
    }
    std::string nodeName = nodePtr->GetName();
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      RTS_LOGW("OptimizeGraphPrepare: desc of node[%s] is null", nodeName.c_str());
      continue;
    }
    if (CheckSupportedOP(opDescPtr->GetType()) == RT_ERROR_NONE) {
      const string name = "_format_agnostic";
      bool bRet = AttrUtils::SetInt(opDescPtr, name, RTS_FORMAT_PAIRED_INPUT_OUTPUT);
      RTS_LOGD("graph[%s]: node [%s] op type [%s] format agnostic value is set", graph.GetName().c_str(),
               nodeName.c_str(), opDescPtr->GetType().c_str());
      if (!bRet) {
        RTS_LOGE("graph[%s]: node [%s] op type [%s] set format agnostic failed", graph.GetName().c_str(),
                 nodeName.c_str(), opDescPtr->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  RTS_LOGD("end rts graph optimizer prepare");
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::OptimizeWholeGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::GetAttributes(GraphOptimizerAttribute &attrs) const {
  attrs.scope = ge::UNIT;
  attrs.engineName = RTS_ENGINE_NAME;
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::OptimizeOriginalGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::OptimizeFusedGraph(ComputeGraph &graph) {
  (void)graph;
  // do nothing
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::OptimizeGraphBeforeBuild(ComputeGraph &graph) {
  RTS_LOGD("Enter OptimizeGraphBeforeBuild.");
  return PorcMemtypeRange(graph);
}

rtError_t RtsGraphOptimizer::CheckSupportedOP(const std::string &sCollectiveType) {
  const auto it = std::find(RTS_SUPPORTED_OP_TYPE.begin(), RTS_SUPPORTED_OP_TYPE.end(), sCollectiveType);
  return (it != RTS_SUPPORTED_OP_TYPE.end()) ? RT_ERROR_NONE : RT_ERROR_INVALID_VALUE;
}

ge::Status RtsGraphOptimizer::PorcMemtypeRange(ge::ComputeGraph &graph) {
  const int32_t kSocVersionLen = 50;
  char_t version[kSocVersionLen] = {0};
  auto ret = GetSocVersion(version, kSocVersionLen);
  if (ret != SUCCESS) {
    RTS_LOGE("GetSocVersion failed, graphId=%u.", graph.GetGraphID());
    return FAILED;
  }
  if ((strncmp(version, "Ascend610", strlen("Ascend610")) != 0) &&
      (strncmp(version, "Ascend310P1", strlen("Ascend310P1")) != 0) &&
      (strncmp(version, "Ascend310P3", strlen("Ascend310P3")) != 0) &&
      (strncmp(version, "BS9SX1AA", strlen("BS9SX1AA")) != 0) &&
      (strncmp(version, "BS9SX1AB", strlen("BS9SX1AB")) != 0) &&
      (strncmp(version, "BS9SX1AC", strlen("BS9SX1AC")) != 0)) {
    RTS_LOGI("Soc version is [%s], no need to process the condition node", version);
    return SUCCESS;
  }
  RTS_LOGI("Soc version is [%s]", version);
  TraceOwnerGuard guard1("RTS", "SetMemTypeRange", graph.GetName());
  ret = SetMemTypeRange(graph);
  if (ret != SUCCESS) {
    RTS_LOGE("Set mem type range of root graph:%s failed, retCode=%#x.", graph.GetName().c_str(), ret);
    return FAILED;
  }
  for (auto &subGraph : graph.GetAllSubgraphs()) {
    ret = SetMemTypeRange(*subGraph);
    if (ret != SUCCESS) {
      RTS_LOGE("Set mem type range of graph:%s failed, retCode=%#x", subGraph->GetName().c_str(), ret);
      return FAILED;
    }
  }
  TraceOwnerGuard guard2("RTS", "ProcConditionNode", graph.GetName());
  ret = ProcConditionNode(graph);
  if (ret != SUCCESS) {
    RTS_LOGE("Process condition node of root graph:%s failed, retCode=%#x.", graph.GetName().c_str(), ret);
    return FAILED;
  }
  for (auto &subGraph : graph.GetAllSubgraphs()) {
    ret = ProcConditionNode(*subGraph);
    if (ret != SUCCESS) {
      RTS_LOGE("Process condition node of graph:%s failed, retCode=%#x.", subGraph->GetName().c_str(), ret);
      return FAILED;
    }
  }
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::ProcConditionNode(ge::ComputeGraph &graph) {
  if (graph.GetGraphUnknownFlag()) {
    RTS_LOGD("Graph[%s] is unknown graph, skip.", graph.GetName().c_str());
    return SUCCESS;
  }

  uint32_t inputNodeNum;

  for (auto nodePtr : graph.GetDirectNode()) {
    if (nodePtr == nullptr) {
      RTS_LOGW("Node ptr is null.");
      continue;
    }

    auto opDescPtr = nodePtr->GetOpDesc();
    if (opDescPtr == nullptr) {
      RTS_LOGW("Desc of node[%s] is null", nodePtr->GetName().c_str());
      continue;
    }

    inputNodeNum = 0;
    if (CheckConditionOPAndGetInputNodeNum(opDescPtr->GetType(), &inputNodeNum) != RT_ERROR_NONE) {
      continue;
    }
    auto ret = InsertMemcpyAsyncNodeAndSetMemType(nodePtr, graph, inputNodeNum);
    if (ret != SUCCESS) {
      RTS_LOGE("Insert memcpy node failed, inputNodeNum [%u], retCode=%u", inputNodeNum, ret);
      return FAILED;
    }
  }

  return SUCCESS;
}

rtError_t RtsGraphOptimizer::CheckConditionOPAndGetInputNodeNum(const std::string &sCollectiveType,
                                                                uint32_t *inputNum) {
  auto search = RTS_SUPPORTED_CONDITION_OP_TYPE.find(sCollectiveType);
  if (search != RTS_SUPPORTED_CONDITION_OP_TYPE.end()) {
    *inputNum = search->second;
    return RT_ERROR_NONE;
  }
  RTS_LOGD("Can not find key:%s in op type map.", sCollectiveType.c_str());
  return RT_ERROR_INVALID_VALUE;
}

ge::Status RtsGraphOptimizer::InsertMemcpyAsyncNodeFunc(const ge::NodePtr &nexNode, ge::NodePtr &memcpyAsyncNode,
                                                        ge::OpDescPtr &memcpyAsyncOpDesc, uint32_t index) {
  // only insert memcpyAsync between StreamSwitchN and its 4G input node.
  auto inDataAnchor = nexNode->GetInDataAnchor(index);
  if (inDataAnchor == nullptr) {
    RTS_REPORT_CALL_ERROR("First in data anchor is null.");
    return FAILED;
  }

  auto srcOutAnchor = inDataAnchor->GetPeerOutAnchor();
  if (srcOutAnchor == nullptr) {
    RTS_REPORT_CALL_ERROR("Source out anchor is null.");
    return FAILED;
  }

  auto nexDesc = nexNode->GetOpDesc();
  if (nexDesc == nullptr) {
    RTS_REPORT_CALL_ERROR("Next op desc is null.");
    return FAILED;
  }

  if (nexDesc->HasAttr(ATTR_NAME_STREAM_LABEL)) {
    std::string nextIterationName;
    if (!AttrUtils::GetStr(nexDesc, ATTR_NAME_STREAM_LABEL, nextIterationName)) {
      RTS_REPORT_CALL_ERROR("Get next op[%s] attribute[ATTR_NAME_STREAM_LABEL] failed.", nexDesc->GetName().c_str());
      return FAILED;
    }

    if (!AttrUtils::SetStr(memcpyAsyncOpDesc, ATTR_NAME_STREAM_LABEL, nextIterationName)) {
      RTS_REPORT_CALL_ERROR("Set memcpyAsyncOpDesc[%s] attribute[ATTR_NAME_STREAM_LABEL] failed.",
                            memcpyAsyncOpDesc->GetName().c_str());
      return FAILED;
    }
  }

  ge::Status status = GraphUtils::RemoveEdge(srcOutAnchor, inDataAnchor);
  if (status != SUCCESS) {
    RTS_REPORT_CALL_ERROR(
        "Graph remove edge failed, src index[%d], dst index[%d], dst node [%s],"
        "retCode=%u.",
        srcOutAnchor->GetIdx(), inDataAnchor->GetIdx(), inDataAnchor->GetOwnerNode()->GetName().c_str(), status);
    return FAILED;
  }

  status = GraphUtils::AddEdge(srcOutAnchor, memcpyAsyncNode->GetInDataAnchor(0));
  if (status != SUCCESS) {
    RTS_REPORT_CALL_ERROR(
        "Graph add edge failed, src index[%d], dst index[%d], dst node [%s],"
        "retCode=%u.",
        srcOutAnchor->GetIdx(), inDataAnchor->GetIdx(), inDataAnchor->GetOwnerNode()->GetName().c_str(), status);
    return FAILED;
  }
  status = GraphUtils::AddEdge(memcpyAsyncNode->GetOutDataAnchor(0), inDataAnchor);
  if (status != SUCCESS) {
    RTS_REPORT_CALL_ERROR(
        "Graph add edge failed, src index[%d], dst index[%d], dst node [%s],"
        "retCode=%u.",
        srcOutAnchor->GetIdx(), inDataAnchor->GetIdx(), inDataAnchor->GetOwnerNode()->GetName().c_str(), status);
    return FAILED;
  }

  for (const NodePtr &inNode : nexNode->GetInControlNodes()) {
    status = GraphUtils::RemoveEdge(inNode->GetOutControlAnchor(), nexNode->GetInControlAnchor());
    if (status != SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Graph remove edge failed, src index[%d], dst index[%d], dst node [%s],"
          "retCode=%u.",
          srcOutAnchor->GetIdx(), inDataAnchor->GetIdx(), inDataAnchor->GetOwnerNode()->GetName().c_str(), status);
      return FAILED;
    }

    status = GraphUtils::AddEdge(inNode->GetOutControlAnchor(), memcpyAsyncNode->GetInControlAnchor());
    if (status != SUCCESS) {
      RTS_REPORT_CALL_ERROR(
          "Graph add edge failed, src index[%d], dst index[%d], dst node [%s],"
          "retCode=%u.",
          srcOutAnchor->GetIdx(), inDataAnchor->GetIdx(), inDataAnchor->GetOwnerNode()->GetName().c_str(), status);
      return FAILED;
    }
  }

  RTS_LOGI("Insert memcpyAsync op success, src node: %s, dst node: %s", srcOutAnchor->GetOwnerNode()->GetName().c_str(),
           nexNode->GetName().c_str());

  return SUCCESS;
}

ge::Status RtsGraphOptimizer::InsertMemcpyAsyncNodeAndSetMemType(const ge::NodePtr &nexNode, ge::ComputeGraph &graph,
                                                                 uint32_t inputNodeNum) {
  if (nexNode == nullptr) {
    RTS_LOGW("Next node is null.");
    return FAILED;
  }

  uint32_t index;
  for (index = 0; index < inputNodeNum; index++) {
#ifndef FEATURE_GE_API
    ge::OpDescPtr memcpyAsyncOpDesc = CreateMemcpyAsyncOpByIndex(nexNode, index);
    if (memcpyAsyncOpDesc == nullptr) {
      RTS_LOGE("Create memcpy async op failed.");
      return FAILED;
    }

    ge::NodePtr memcpyAsyncNode = graph.AddNode(memcpyAsyncOpDesc);
    if (memcpyAsyncNode == nullptr) {
      RTS_REPORT_CALL_ERROR("Insert memcpy node failed.");
      return FAILED;
    }

    bool isLabelNode = false;
    if (AttrUtils::GetBool(nexNode->GetOpDesc(), ATTR_NAME_RTS_LABEL_NODE, isLabelNode)) {
      if (!AttrUtils::SetBool(memcpyAsyncOpDesc, ATTR_NAME_RTS_LABEL_NODE, isLabelNode)) {
        RTS_REPORT_CALL_ERROR("Node [%s] set ATTR_NAME_RTS_LABEL_NODE failed", nexNode->GetName().c_str());
        return FAILED;
      }
    }

    ge::Status status = InsertMemcpyAsyncNodeFunc(nexNode, memcpyAsyncNode, memcpyAsyncOpDesc, index);
    if (status != SUCCESS) {
      RTS_LOGE("Graph add memcpyAsync node failed, retCode=%u.", status);
      return FAILED;
    }

    status = SetMemTypeOutputRange(memcpyAsyncNode, 0);
    if (status != SUCCESS) {
      RTS_REPORT_CALL_ERROR("Set mem type output range attr failed, node [%s], retCode=%u.",
                            memcpyAsyncNode->GetName().c_str(), status);
      return FAILED;
    }

    status = SetMemTypeInputRange(nexNode, index);
    if (status != SUCCESS) {
      RTS_REPORT_CALL_ERROR("Set mem type input range attr failed, node [%s], retCode=%u.", nexNode->GetName().c_str(),
                            status);
      return FAILED;
    }
#endif
  }

  return SUCCESS;
}

ge::OpDescPtr RtsGraphOptimizer::CreateMemcpyAsyncOpByIndex(const ge::NodePtr &nexNode, uint32_t index) {
  if (nexNode == nullptr) {
    RTS_REPORT_CALL_ERROR("Node is null.");
    return nullptr;
  }

  string nodeName = nexNode->GetName() + std::to_string(index) + "_MemcpyAsync";
  OpDescPtr opDesc;
  try {
    opDesc = std::make_shared<OpDesc>(nodeName.c_str(), "MemcpyAsync");
  } catch (std::bad_alloc &) {
    RTS_REPORT_INNER_ERROR("Make shared OpDesc failed, bad_alloc raised, node name=%s.", nodeName.c_str());
    return nullptr;
  } catch (...) {
    RTS_REPORT_INNER_ERROR("Make shared OpDesc failed, other exception raised, node name=%s.", nodeName.c_str());
    return nullptr;
  }
  RTS_LOGI("Create memcpyAsync op [%s] success.", opDesc->GetName().c_str());

  ge::OpDescPtr nextNodeOpDesc = nexNode->GetOpDesc();
  if (nextNodeOpDesc == nullptr) {
    RTS_REPORT_CALL_ERROR("OpDesc of next node is invalid, node name=%s.", nodeName.c_str());
    return nullptr;
  }
  size_t inputSize = nextNodeOpDesc->GetInputsSize();
  if (inputSize == 0) {
    RTS_REPORT_CALL_ERROR("The input size of node [%s] is 0.", nexNode->GetName().c_str());
    return nullptr;
  }

  auto ret = opDesc->AddInputDesc(nextNodeOpDesc->GetInputDesc(index));
  // only insert memcpyAsync between StreamSwitchN and its first input node.
  if (ret != GRAPH_SUCCESS) {
    RTS_REPORT_CALL_ERROR("Create memcpyAsync op [%s], add input desc failed, retCode=%u.", nodeName.c_str(), ret);
    return nullptr;
  }

  ret = opDesc->AddOutputDesc(nextNodeOpDesc->GetInputDesc(index));
  if (ret != GRAPH_SUCCESS) {
    RTS_REPORT_CALL_ERROR("Create memcpyAsync op[%s], add output desc failed, retCode=%u.", nodeName.c_str(), ret);
    return nullptr;
  }

  return opDesc;
}

ge::Status RtsGraphOptimizer::SetMemTypeInputRange(const ge::NodePtr &node, uint32_t index) {
  if (node == nullptr) {
    RTS_LOGW("Node is null.");
    return FAILED;
  }

  auto opDesc = node->GetOpDesc();
  if (opDesc == nullptr) {
    RTS_LOGW("Desc of node[%s] is null", node->GetName().c_str());
    return FAILED;
  }

  if (index >= opDesc->GetInputsSize()) {
    RTS_LOGW("Index[%u] large than input size[%zu]", index, opDesc->GetInputsSize());
    return FAILED;
  }

  vector<int64_t> vMemoryType;
  (void)ge::AttrUtils::GetListInt(opDesc, ATTR_NAME_INPUT_MEM_TYPE_LIST, vMemoryType);
  for (size_t i = vMemoryType.size(); i < opDesc->GetInputsSize(); ++i) {
    vMemoryType.push_back(RT_MEMORY_DEFAULT);
  }

  vMemoryType[index] = RT_MEMORY_TS;
  if (!ge::AttrUtils::SetListInt(opDesc, ATTR_NAME_INPUT_MEM_TYPE_LIST, vMemoryType)) {
    RTS_REPORT_CALL_ERROR("Node [%s] set ATTR_NAME_INPUT_MEM_TYPE_LIST failed", node->GetName().c_str());
    return FAILED;
  }

  RTS_LOGD("Node [%s] set ATTR_NAME_INPUT_MEM_TYPE_LIST success.", node->GetName().c_str());
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::SetMemTypeOutputRange(const ge::NodePtr &node, uint32_t index) {
  if (node == nullptr) {
    RTS_LOGW("Node is null.");
    return FAILED;
  }

  auto opDesc = node->GetOpDesc();
  if (opDesc == nullptr) {
    RTS_LOGW("Desc of node[%s] is null", node->GetName().c_str());
    return FAILED;
  }

  if (index >= opDesc->GetOutputsSize()) {
    RTS_LOGW("Index[%u] large than output size[%zu]", index, opDesc->GetOutputsSize());
    return FAILED;
  }

  vector<int64_t> vMemoryType;
  (void)ge::AttrUtils::GetListInt(opDesc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, vMemoryType);
  for (size_t i = vMemoryType.size(); i < opDesc->GetOutputsSize(); ++i) {
    vMemoryType.push_back(RT_MEMORY_DEFAULT);
  }

  vMemoryType[index] = RT_MEMORY_TS;
  if (!ge::AttrUtils::SetListInt(opDesc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, vMemoryType)) {
    ;
    RTS_REPORT_CALL_ERROR("Node [%s] set ATTR_NAME_OUTPUT_MEM_TYPE_LIST failed", node->GetName().c_str());
    return FAILED;
  }

  RTS_LOGD("Node [%s] set ATTR_NAME_OUTPUT_MEM_TYPE_LIST success.", node->GetName().c_str());
  return SUCCESS;
}

ge::Status RtsGraphOptimizer::SetMemTypeRange(const ge::NodePtr &node) {
  if (node == nullptr) {
    RTS_LOGW("Node is null.");
    return FAILED;
  }

  auto opDesc = node->GetOpDesc();
  if (opDesc == nullptr) {
    RTS_LOGW("Desc of node[%s] is null", node->GetName().c_str());
    return FAILED;
  }

  if (!AttrUtils::SetBool(opDesc, ge::ATTR_NAME_MEMORY_TYPE_RANGE, true)) {
    RTS_REPORT_CALL_ERROR("Node [%s] set ATTR_NAME_MEMORY_TYPE_RANGE failed", node->GetName().c_str());
    return FAILED;
  }
  RTS_LOGD("Node [%s] set ATTR_NAME_MEMORY_TYPE_RANGE success.", node->GetName().c_str());

  return SUCCESS;
}

ge::Status RtsGraphOptimizer::SetMemTypeRange(ge::ComputeGraph &graph) {
  if (graph.GetGraphUnknownFlag()) {
    RTS_LOGD("Graph[%s] is unknown graph, skip.", graph.GetName().c_str());
    return SUCCESS;
  }

  for (const auto &nodePtr : graph.GetDirectNode()) {
    if (nodePtr == nullptr) {
      RTS_LOGW("Node is null.");
      continue;
    }

    auto opDescPtr = nodePtr->GetOpDesc();
    if (opDescPtr == nullptr) {
      RTS_LOGW("Desc of node[%s] is null", nodePtr->GetName().c_str());
      continue;
    }

    auto iter = std::find(MEM_TYPE_RANGE_NODES.begin(), MEM_TYPE_RANGE_NODES.end(), opDescPtr->GetType());
    if (iter != MEM_TYPE_RANGE_NODES.end()) {
      auto ret = SetMemTypeRange(nodePtr);
      if (ret != SUCCESS) {
        RTS_REPORT_CALL_ERROR("Set mem type range attr failed, node[%s], retCode=%u.", nodePtr->GetName().c_str(), ret);
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce

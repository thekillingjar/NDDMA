/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"
#include "acl_rt_memcpy_kind.h"

namespace {
constexpr int64_t RUNTIME_TENSOR_DESC_SIZE = 1024;
constexpr int64_t RUNTIME_TENSOR_DESC_DATA_SIZE = 16;
constexpr int64_t RUNTIME_TENSOR_DESC_REMOVE_DATA_SIZE = RUNTIME_TENSOR_DESC_SIZE - RUNTIME_TENSOR_DESC_DATA_SIZE;
}  // namespace

using namespace ge;
namespace cce {
namespace runtime {
Op::Op(const Node &node, RunContext &runContext)
    : run_context_(runContext), node_(node), op_desc_(node.GetOpDesc()), input_num_(0), output_num_(0) {
  name_ = op_desc_->GetName();
  type_ = op_desc_->GetType();
}

ge::Status Op::GetOpMemType(const std::string &str, int64_t &memType) const {
  std::vector<int64_t> typeList;
  bool getAttrSucc = AttrUtils::GetListInt(op_desc_, str, typeList);
  if (!getAttrSucc) {
    RTS_REPORT_CALL_ERROR("get attr MEM_TYPE_LIST fail.");
    return FAILED;
  }
  if (typeList.size() == 0U) {
    RTS_REPORT_CALL_ERROR("mem_type list size must be greater than 0.");
    return FAILED;
  }

  memType = typeList[0];
  return SUCCESS;
}

ge::Status Op::GetOpInputMemData(uint8_t *&memBase, uint64_t &memSize, int64_t &inputOffset,
                                 ge::ConstGeTensorDescPtr tensorDescPtr) {
  int64_t kind = 0;
  (void)AttrUtils::GetInt(op_desc_, "_rt_memcpy_kind", kind);

  if (kind == RT_MEMCPY_HOST_TO_DEVICE) {
    int64_t type;
    auto ret = GetOpMemType(ATTR_NAME_INPUT_MEM_TYPE_LIST, type);
    if (ret != SUCCESS) {
      RTS_REPORT_CALL_ERROR("OpInput GetOpMemType Fail.");
      return FAILED;
    }
    auto hostBase = run_context_.mem_type_data_mem_base.find(type);
    auto hostSize = run_context_.mem_type_data_mem_size.find(type);
    if ((hostBase != run_context_.mem_type_data_mem_base.end()) &&
        (hostSize != run_context_.mem_type_data_mem_size.end())) {
      memBase = hostBase->second;
      memSize = hostSize->second;
      inputOffset -= static_cast<int64_t>(reinterpret_cast<uintptr_t>(memBase));
      return SUCCESS;
    } else {
      RTS_REPORT_CALL_ERROR("hostBase or hostSize not find.");
      return FAILED;
    }
  }

  int64_t tensorMemtype = -1;
  (void)AttrUtils::GetInt(tensorDescPtr, ATTR_NAME_TENSOR_MEM_TYPE, tensorMemtype);
  auto it = run_context_.mem_type_data_mem_base.find(tensorMemtype);
  auto iter = run_context_.mem_type_data_mem_size.find(tensorMemtype);
  if ((it != run_context_.mem_type_data_mem_base.end()) && (iter != run_context_.mem_type_data_mem_size.end())) {
    memBase = it->second;
    memSize = iter->second;
  }
  return SUCCESS;
}

ge::Status Op::GetOpOutputMemData(uint8_t *&memBase, uint64_t &memSize, int64_t &outputOffset,
                                  ge::ConstGeTensorDescPtr tensorDescPtr) {
  int64_t kind = 0;
  (void)AttrUtils::GetInt(op_desc_, "_rt_memcpy_kind", kind);

  if (kind == RT_MEMCPY_DEVICE_TO_HOST) {
    int64_t memType;
    auto ret = GetOpMemType(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memType);
    if (ret != SUCCESS) {
      RTS_REPORT_CALL_ERROR("OpOutput getOpMemType Fail.");
      return FAILED;
    }
    auto hostBase = run_context_.mem_type_data_mem_base.find(memType);
    auto hostSize = run_context_.mem_type_data_mem_size.find(memType);
    if ((hostBase != run_context_.mem_type_data_mem_base.end()) &&
        (hostSize != run_context_.mem_type_data_mem_size.end())) {
      memBase = hostBase->second;
      memSize = hostSize->second;
      outputOffset -= static_cast<int64_t>(reinterpret_cast<uintptr_t>(memBase));
      return SUCCESS;
    } else {
      RTS_REPORT_CALL_ERROR("hostBase or hostSize not find.");
      return FAILED;
    }
  }

  int64_t tensorMemtype = -1;
  (void)AttrUtils::GetInt(tensorDescPtr, ATTR_NAME_TENSOR_MEM_TYPE, tensorMemtype);
  auto it = run_context_.mem_type_data_mem_base.find(tensorMemtype);
  auto iter = run_context_.mem_type_data_mem_size.find(tensorMemtype);
  if ((it != run_context_.mem_type_data_mem_base.end()) && (iter != run_context_.mem_type_data_mem_size.end())) {
    memBase = it->second;
    memSize = iter->second;
  }
  return SUCCESS;
}

Status Op::InitInput() {
  vector<int64_t> input_offsets = op_desc_->GetInputOffset();

  uint32_t inputIndex = 0;
  uint32_t anchorIndex = 0;
  for (auto &anchor : node_.GetAllInDataAnchors()) {
    if (AnchorUtils::GetStatus(anchor) == ANCHOR_SUSPEND) {
      ++anchorIndex;
      continue;
    }
    if (inputIndex >= input_offsets.size()) {
      RTS_REPORT_CALL_ERROR(
          "InputIndex must be less than input offsets size,"
          "input index=%u, input_offsets size=%zu.",
          inputIndex, input_offsets.size());
      return FAILED;
    }

    int64_t input_offset = input_offsets[inputIndex];

    auto tensor_desc = op_desc_->GetInputDescPtr(anchorIndex);
    if (tensor_desc == nullptr) {
      RTS_REPORT_CALL_ERROR("GetInputDescPtr failed, name:%s, anchorIndex:%u.", name_.c_str(), anchorIndex);
      return FAILED;
    }

    int64_t tensorSize = 0;
    if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensorSize) != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR("Get tensor size failed, name:%s, index:%u", name_.c_str(), anchorIndex);
      return FAILED;
    }

    std::vector<bool> is_input_var_vec;
    bool getAttrRes = AttrUtils::GetListBool(op_desc_, "INPUT_IS_VAR", is_input_var_vec);
    // var address no need add base
    if (getAttrRes && (is_input_var_vec.size() > inputIndex) && is_input_var_vec[inputIndex]) {
      v_input_data_addr_.push_back((void *)(uintptr_t)input_offset);
    } else {
      uint8_t *memBase = run_context_.dataMemBase;
      uint64_t memSize = run_context_.dataMemSize;

      if (AnchorUtils::GetStatus(anchor) == ANCHOR_CONST) {
        memBase = run_context_.weightMemBase;
        memSize = run_context_.weightMemSize;
      }
      if (GetOpInputMemData(memBase, memSize, input_offset, tensor_desc) != SUCCESS) {
        RTS_LOGE("get input mem data Failed ");
        return FAILED;
      }

      uintptr_t memAddr = 0;
      Status ret = CalcAddr((uintptr_t)memBase, memSize, input_offset, tensorSize, memAddr);
      if (ret != SUCCESS) {
        RTS_LOGW(
            "Calc input(name:%s, inputIndex:%u) Failed, "
            "offset=%" PRId64 ", size=%" PRId64 ", total size=%" PRIu64 ", retcode=%#x!",
            name_.c_str(), inputIndex, input_offset, tensorSize, memSize, ret);
        return ret;
      }
      v_input_data_addr_.push_back((void *)memAddr);

      ret = InitDesc((uintptr_t)memBase, memSize, tensor_desc, memAddr, inputDescAddrs_);
      if (ret != SUCCESS) {
        RTS_REPORT_CALL_ERROR("Calc input desc(name:%s, inputIndex:%u) Failed, retcode=%#x!", name_.c_str(), inputIndex,
                              ret);
        return ret;
      }
    }

    v_input_size_.push_back(tensorSize);

    inputIndex++;
    anchorIndex++;
  }
  return SUCCESS;
}

ge::Status Op::InitDesc(const uintptr_t memBase, const uint64_t memSize, const ge::ConstGeTensorDescPtr &tensorDesc,
                        const uintptr_t dataAddr, std::unordered_map<uintptr_t, uintptr_t> &descAddrs) {
  int64_t tensorDescOffset = 0;
  const bool hasOffsetAttr = AttrUtils::GetInt(tensorDesc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, tensorDescOffset);
  if (hasOffsetAttr) {
    uintptr_t sinkTensorDescAddr = 0U;
    auto ret = CalcAddr(memBase, memSize, tensorDescOffset, RUNTIME_TENSOR_DESC_SIZE, sinkTensorDescAddr);
    if (ret != SUCCESS) {
      return ret;
    }
    (void)descAddrs.emplace(dataAddr, sinkTensorDescAddr);
    RTS_LOGI("Add dataAddr=%#" PRIx64 ", descAddr=%#" PRIx64 ",", dataAddr, sinkTensorDescAddr);
  }
  return SUCCESS;
}

Status Op::CalcAddr(uintptr_t mem_base, uint64_t mem_size, int64_t offset, int64_t data_size, uintptr_t &memAddr) {
  // when offset < 0, it's p2p mem, no need check.
  if (offset >= 0) {
    auto offsetU64 = static_cast<uint64_t>(offset);
    Status ret = CheckOffsetAndSize(offsetU64, data_size, mem_size);
    if (ret != SUCCESS) {
      RTS_LOGW(
          "Check (name:%s) offset and size failed, "
          "offset=%" PRId64 ", size=%" PRId64 ", total size=%" PRIu64,
          name_.c_str(), offset, data_size, mem_size);
      return ret;
    }
    if (CheckUint64Overflow(mem_base, offsetU64)) {
      RTS_LOGI("Calc (name:%s) addr overflow, mem base=%" PRIu64 ", offset=%" PRId64, name_.c_str(), mem_base, offset);
      return FAILED;
    }
  }
  memAddr = mem_base + offset;
  return SUCCESS;
}

Status Op::InitOutput() {
  vector<int64_t> output_offsets = op_desc_->GetOutputOffset();
  if (output_num_ != output_offsets.size()) {
    RTS_REPORT_CALL_ERROR(
        "Init output failed, output num is not equal to output offsets size,"
        " output num=%zu, output offsets size=%zu.",
        output_num_, output_offsets.size());
    return FAILED;
  }
  for (size_t outputIndex = 0; outputIndex < output_num_; ++outputIndex) {
    ConstGeTensorDescPtr tensorDescPtr = op_desc_->GetOutputDescPtr(static_cast<uint32_t>(outputIndex));
    if (tensorDescPtr == nullptr) {
      RTS_REPORT_CALL_ERROR("Get output desctription handle failed, name:%s, outputIndex:%zu.", name_.c_str(),
                            outputIndex);
      return FAILED;
    }
    int64_t outputOffset = output_offsets[outputIndex];

    int64_t tensorSize = 0;
    if (TensorUtils::GetSize(*tensorDescPtr, tensorSize) != GRAPH_SUCCESS) {
      RTS_REPORT_CALL_ERROR("Get tensor size failed, name:%s, output index:%zu", name_.c_str(), outputIndex);
      return FAILED;
    }
    std::vector<bool> isOutputVarVec;
    bool getAttrRes = AttrUtils::GetListBool(op_desc_, "OUTPUT_IS_VAR", isOutputVarVec);
    // var address no need add base
    if (getAttrRes && (isOutputVarVec.size() > outputIndex) && isOutputVarVec[outputIndex]) {
      v_output_data_addr_.push_back((void *)(uintptr_t)outputOffset);
    } else {
      uint8_t *memBase = run_context_.dataMemBase;
      uint64_t memSize = run_context_.dataMemSize;

      if (GetOpOutputMemData(memBase, memSize, outputOffset, tensorDescPtr) != SUCCESS) {
        RTS_LOGE("get output mem data Failed ");
        return FAILED;
      }
      uintptr_t memAddr = 0;
      Status ret = CalcAddr((uintptr_t)memBase, memSize, outputOffset, tensorSize, memAddr);
      if (ret != SUCCESS) {
        RTS_LOGW(
            "Calc output(name:%s, outputIndex:%zu) Failed, "
            "offset=%" PRId64 ", size=%" PRId64 ", total size=%" PRIu64,
            name_.c_str(), outputIndex, outputOffset, tensorSize, run_context_.dataMemSize);
        return ret;
      }
      v_output_data_addr_.push_back((void *)memAddr);
      ret = InitDesc((uintptr_t)run_context_.dataMemBase, run_context_.dataMemSize, tensorDescPtr, memAddr,
                     outputDescAddrs_);
      if (ret != SUCCESS) {
        RTS_REPORT_CALL_ERROR("Calc output desc(name:%s, outputIndex:%zu) Failed, retcode=%#x!", name_.c_str(),
                              outputIndex, ret);
        return ret;
      }
    }
    v_output_size_.push_back(tensorSize);
  }
  return SUCCESS;
}

Status Op::CheckOffsetAndSize(uint64_t offset, uint64_t space_size, uint64_t total_size) {
  RTS_LOGI("check offset start, Offset=%" PRIu64 ", space_size=%" PRIu64 ", total_size=%" PRIu64, offset, space_size,
           total_size);

  if (CheckUint64Overflow(offset, space_size)) {
    RTS_REPORT_CALL_ERROR(
        "Check offset and size failed, %s offset add space_size overflow,"
        " offset=%" PRId64 ", space_size=%" PRIu64,
        name_.c_str(), offset, space_size);
    return FAILED;
  }

  if (offset + space_size > total_size) {
    RTS_REPORT_CALL_ERROR("Check offset and size failed, %s offset add space_size %" PRIu64 " + %" PRIu64
                          " should "
                          "not bigger than total memory size %" PRIu64,
                          name_.c_str(), offset, space_size, total_size);
    return FAILED;
  }
  return SUCCESS;
}

// init the member
Status Op::Init() {
  auto parent_node = node_.GetOwnerComputeGraphBarePtr()->GetParentNode();
  if (parent_node != nullptr && parent_node->GetOpDesc() != nullptr &&
      parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
    dynamic_flag_ = parent_node->GetOwnerComputeGraphBarePtr()->GetGraphUnknownFlag();
  } else {
    dynamic_flag_ = node_.GetOwnerComputeGraphBarePtr()->GetGraphUnknownFlag();
  }
  if (dynamic_flag_) {
    RTS_LOGI("op:%s is in unknownShape graph, does not need to init io.", node_.GetName().c_str());
    return SUCCESS;
  }
  RTS_LOGD("op:%s start init input.", node_.GetName().c_str());
  Status ret = InitInput();
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Name:%s InitInput failed,  retCode=%#x!", name_.c_str(), ret);
    return ret;
  }

  ret = InitOutput();
  if (ret != SUCCESS) {
    RTS_REPORT_CALL_ERROR("Name:%s InitOutput failed, retCode=%#x!", name_.c_str(), ret);
    return ret;
  }
  return SUCCESS;
}

Status Op::UpdateOutDescFromInDesc(const void *inputAddr, const void *outputAddr, vector<TaskDef> &tasks) {
  auto inputIter = inputDescAddrs_.find(reinterpret_cast<uintptr_t>(inputAddr));
  if (inputIter != inputDescAddrs_.end()) {
    auto outputIter = outputDescAddrs_.find(reinterpret_cast<uintptr_t>(outputAddr));
    if (outputIter == outputDescAddrs_.end()) {
      RTS_REPORT_CALL_ERROR("Run opName[%s] opType[%s] failed, input has desc, but output don't has desc.",
                            name_.c_str(), type_.c_str());
      return FAILED;
    }
    domi::TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_MEMCPY_ADDR_ASYNC);
    taskDef.set_stream_id(op_desc_->GetStreamId());
    domi::MemcpyAsyncDef *memcpyDef = taskDef.mutable_memcpy_async();
    memcpyDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
    memcpyDef->set_dst(outputIter->second + RUNTIME_TENSOR_DESC_DATA_SIZE);
    memcpyDef->set_dst_max(RUNTIME_TENSOR_DESC_REMOVE_DATA_SIZE);
    memcpyDef->set_src(inputIter->second + RUNTIME_TENSOR_DESC_DATA_SIZE);
    memcpyDef->set_count(RUNTIME_TENSOR_DESC_REMOVE_DATA_SIZE);
    memcpyDef->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
    tasks.push_back(taskDef);
    RTS_LOGI("Memcpy desc outAddr:%" PRIu64 ", inAddr:%" PRIu64, outputIter->second, inputIter->second);
    return SUCCESS;
  }
  RTS_LOGI("Not need updata desc.");
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce

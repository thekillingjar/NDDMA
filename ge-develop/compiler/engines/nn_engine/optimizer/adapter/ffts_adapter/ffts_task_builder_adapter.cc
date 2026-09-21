/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/ffts_adapter/ffts_task_builder_adapter.h"
#include "securec.h"
#include "common/fe_log.h"
#include "common/configuration.h"
#include "common/graph_comm.h"
#include "common/aicore_util_types.h"
#include "common/op_tensor_utils.h"
#include "common/fe_type_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_graph_common.h"
#include "graph/tuning_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "rt_error_codes.h"
#include "framework/common/ge_types.h"

namespace fe {
namespace {
const uint32_t DEFAULT_KERNEL_BLOCK_DIM = 1;
const uint32_t DEFAULT_KERNEL_SIZE = 0;
const int64_t UNKNOWN_WORK_SPACE_SIZE = -1;
};

FftsTaskBuilderAdapter::FftsTaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context)
    : TaskBuilderAdapter(node, context), block_dim_(DEFAULT_KERNEL_BLOCK_DIM),
      tbe_kernel_size_(DEFAULT_KERNEL_SIZE), thread_dim_(0) {}

FftsTaskBuilderAdapter::~FftsTaskBuilderAdapter() {}

Status FftsTaskBuilderAdapter::ThreadInitInput(vector<vector<vector<ffts::DimRange>>> &tensor_slice) {
  uint32_t anchor_index = 0;
  size_t valid_tensor_count = 0;
  vector<int64_t> tensor_size_vec;
  vector<ge::DataType> data_type_vec;
  std::vector<uint32_t> input_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node_.GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes);
  for (auto const &anchor : node_.GetAllInDataAnchors()) {
    if (ge::AnchorUtils::GetStatus(anchor) == ge::ANCHOR_SUSPEND) {
      FE_LOGD("Node[type=%s,name=%s]: the anchor %u status is suspend.",
              node_.GetType().c_str(), node_.GetName().c_str(), anchor_index);
      anchor_index++;
      continue;
    }
    FE_CHECK_NOTNULL(node_.GetOpDesc()->MutableInputDesc(anchor->GetIdx()));
    FE_CHECK_NOTNULL(node_.GetOpDesc()->MutableInputDesc(anchor_index));
    FE_LOGD("Node[type=%s,name=%s]: the in anchor %u name is %s, status is %d.",
            node_.GetType().c_str(), node_.GetName().c_str(), anchor_index,
            node_.GetOpDesc()->MutableInputDesc(anchor_index)->GetName().c_str(),
            ge::AnchorUtils::GetStatus(anchor));
    valid_tensor_count++;

    int64_t tensor_size;
    ge::GeTensorDescPtr tensor_desc_ptr = node_.GetOpDesc()->MutableInputDesc(anchor->GetIdx());
    if (tensor_desc_ptr == nullptr) {
      anchor_index++;
      continue;
    }
    if (ge::TensorUtils::GetSize(*tensor_desc_ptr, tensor_size) != SUCCESS) {
        FE_LOGD("Task Builder get size unsuccessful.");
        return FAILED;
    }
    input_tensor_sizes_.push_back(tensor_size);

    const ge::DataType data_type = tensor_desc_ptr->GetDataType();
    FE_LOGD("input node %s data_type: %d.", node_.GetName().c_str(), data_type);
    data_type_vec.push_back(data_type);
    anchor_index++;
  }

  if (tensor_slice.size() == 0) {
      FE_LOGE("tensor slice size is zero.");
      return FAILED;
  }
  if ((input_addrs_.size() != tensor_slice[0].size()) ||
      (tensor_slice[0].size() != valid_tensor_count)) {
    FE_LOGE("Num of node input, sgt input and shape dim are not equal, node input %zu, sgt input %lu, shape %zu.",
            input_addrs_.size(), tensor_slice[0].size(), valid_tensor_count);
    return FAILED;
  }

  uint32_t input_index = 0;
  for (auto &tensor_range : tensor_slice[0]) {
    if (!std::count(input_tensor_indexes.begin(), input_tensor_indexes.end(), input_index)) {
      FE_LOGD("Node[%s] in index:%u not slice.", node_.GetName().c_str(), input_index);
      thread_addr_offset_.push_back(0);
      input_index++;
      continue;
    }
    int64_t thread_offset = 1;
    for (auto &dim_range : tensor_range) {
      FE_LOGD("input node %s dim_range.higher: %ld, dim_range.lower: %ld",
              node_.GetName().c_str(), dim_range.higher, dim_range.lower);
      thread_offset *= (dim_range.higher - dim_range.lower);
    }
    thread_offset = ge::GetSizeInBytes(thread_offset, data_type_vec[input_index]);
    FE_LOGD("thread_offset: %ld, tensor size: %ld", thread_offset, input_tensor_sizes_[input_index]);
    thread_offset = thread_offset == input_tensor_sizes_[input_index] ? 0 : thread_offset;
    thread_addr_offset_.push_back(thread_offset);
    input_index++;
  }

  first_thread_input_addrs_ = input_addrs_;
  thread_dim_ = tensor_slice.size();
  return SUCCESS;
}

Status FftsTaskBuilderAdapter::ThreadInitOutput(vector<vector<vector<ffts::DimRange>>> &tensor_slice) {
  uint32_t anchor_index = 0;
  vector<vector<int64_t>> tensor_shape_vec;
  vector<ge::DataType> data_type_vec;
  std::vector<uint32_t> output_tensor_indexes;
  (void)ge::AttrUtils::GetListInt(node_.GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes);
  for (auto const &anchor : node_.GetAllOutDataAnchors()) {
    FE_CHECK_NOTNULL(node_.GetOpDesc()->MutableOutputDesc(anchor->GetIdx()));
    FE_CHECK_NOTNULL(node_.GetOpDesc()->MutableOutputDesc(anchor_index));
    FE_LOGD("Node[type=%s,name=%s]: the out anchor %u name is %s, status is %d.",
            node_.GetType().c_str(), node_.GetName().c_str(), anchor_index,
            node_.GetOpDesc()->MutableOutputDesc(anchor_index)->GetName().c_str(), ge::AnchorUtils::GetStatus(anchor));
    tensor_shape_vec.push_back(node_.GetOpDesc()->MutableOutputDesc(anchor->GetIdx())->MutableShape().GetDims());

    int64_t tensor_size;
    ge::GeTensorDescPtr tensor_desc_ptr = node_.GetOpDesc()->MutableOutputDesc(anchor->GetIdx());
    if (ge::TensorUtils::GetSize(*tensor_desc_ptr, tensor_size) != SUCCESS) {
        FE_LOGE("Task Builder get size failed.");
        return FAILED;
    }
    output_tensor_sizes_.push_back(tensor_size);
    const ge::DataType data_type = tensor_desc_ptr->GetDataType();
    FE_LOGD("output node %s data_type: %d.", node_.GetName().c_str(), data_type);
    data_type_vec.push_back(data_type);
    anchor_index++;
  }

  if (tensor_slice.size() == 0) {
      FE_LOGE("tensor slice size is zero.");
      return FAILED;
  }
  if ((output_addrs_.size() != tensor_slice[0].size()) ||
      (tensor_slice[0].size() != tensor_shape_vec.size())) {
    FE_LOGE("Num of node output, sgt output and shape dim are not equal, node output %zu, sgt output %lu, shape %zu.",
            output_addrs_.size(), tensor_slice[0].size(), tensor_shape_vec.size());
    return FAILED;
  }

  uint32_t output_index = 0;
  for (auto &tensor_range : tensor_slice[0]) {
    if (!std::count(output_tensor_indexes.begin(), output_tensor_indexes.end(), output_index)) {
      FE_LOGD("Node[%s] out index:%u not slice.", node_.GetName().c_str(), output_index);
      thread_addr_offset_.push_back(0);
      output_index++;
      continue;
    }
    int64_t thread_offset = 1;
    for (auto &dim_range : tensor_range) {
      FE_LOGD("output node %s dim_range.higher: %ld, dim_range.lower: %ld",
              node_.GetName().c_str(), dim_range.higher, dim_range.lower);
      thread_offset *= (dim_range.higher - dim_range.lower);
    }
    thread_offset = ge::GetSizeInBytes(thread_offset, data_type_vec[output_index]);
    FE_LOGD("thread_offset: %ld, tensor size: %ld", thread_offset, output_tensor_sizes_[output_index]);
    thread_offset = thread_offset == output_tensor_sizes_[output_index] ? 0 : thread_offset;
    thread_addr_offset_.push_back(thread_offset);
    output_index++;
  }

  first_thread_output_addrs_ = output_addrs_;
  return SUCCESS;
}

void FftsTaskBuilderAdapter::DebugThreadArgs() const {
  FE_LOGD("Node %s ==== DebugThreadArgs Begin ====", node_.GetName().c_str());
  size_t index = 0;
  for (const vector<uint32_t> &workspace_sizes : thread_workspace_sizes_) {
    size_t index_w = 0;
    for (const uint32_t &workspace_size : workspace_sizes) {
      FE_LOGD("Thread %zu workspace %zu size %u.", index, index_w, workspace_size);
      index_w++;
    }
    index++;
  }

  index = 0;
  for (const int64_t &addr_offset : thread_addr_offset_) {
    FE_LOGD("Addr offset %zu args %ld.", index, addr_offset);
    index++;
  }

  index = 0;
  for (const int64_t &input_tensor_size : input_tensor_sizes_) {
    FE_LOGD("Input %zu tensor size %ld.", index, input_tensor_size);
    index++;
  }

  index = 0;
  for (const int64_t &output_tensor_size : output_tensor_sizes_) {
    FE_LOGD("Output %zu tensor size %ld.", index, output_tensor_size);
    index++;
  }
}
Status FftsTaskBuilderAdapter::ThreadInit() {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node_.GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  FE_LOGI("ThreadInit of Op name: %s, Op id: %ld.",
          node_.GetOpDesc()->GetName().c_str(), node_.GetOpDesc()->GetId());
  FE_CHECK_NOTNULL(slice_info_ptr);

  if (ThreadInitInput(slice_info_ptr->input_tensor_slice) != SUCCESS) {
    FE_LOGE("thread init input failed.");
    return FAILED;
  }

  if (ThreadInitOutput(slice_info_ptr->output_tensor_slice) != SUCCESS) {
    FE_LOGE("thread init output failed.");
    return FAILED;
  }
  (void)ge::AttrUtils::SetListInt(node_.GetOpDesc(), kThreadAddrOffset, thread_addr_offset_);
  return SUCCESS;
}

Status FftsTaskBuilderAdapter::HandleAnchorData(size_t &input_index, const size_t &anchor_index) {
  auto in_desc_ptr = op_desc_->MutableInputDesc(static_cast<uint32_t>(anchor_index));
  FE_CHECK_NOTNULL(in_desc_ptr);
  int64_t weight_offset = 0;
  if (ge::TensorUtils::GetDataOffset(*in_desc_ptr, weight_offset) != ge::GRAPH_SUCCESS) {
    FE_LOGE("Get weight offset failed. op[%s]", op_desc_->GetName().c_str());
    return FAILED;
  }

  FE_LOGI("Print weight_offset and weight_mem_base of op_name[%s], op_type[%s]: [%ld:%lu].",
          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), weight_offset,
          context_.weightMemBase);

  // max value of weight offset : 30 * 1024 * 1024 * 1024L.
  if (weight_offset < kMaxWeightOffset) {
    input_addrs_.push_back(context_.weightMemBase + weight_offset);
  } else {
    input_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(weight_offset)));
  }

  input_index++;
  return SUCCESS;
}

Status FftsTaskBuilderAdapter::InitInput() {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  vector<bool> input_is_addr_var;
  (void)ge::AttrUtils::GetListBool(op_desc_, ATTR_NAME_INPUT_IS_VAR, input_is_addr_var);

  // if input_offsets is empty, set 0 to vector
  vector<int64_t> input_offsets = op_desc_->GetInputOffset();
  if (input_offsets.empty()) {
    vector<int64_t> input_offset_zero(op_desc_->GetInputsSize(), 0);
    input_offsets.swap(input_offset_zero);
    FE_LOGD("Node[%s, %s]: input_offset_size is %zu.", op_type.c_str(), op_name.c_str(), input_offsets.size());
  }

  size_t input_index = 0;
  size_t anchor_index = 0;
  for (auto const &anchor : node_.GetAllInDataAnchors()) {
    if (ge::AnchorUtils::GetStatus(anchor) == ge::ANCHOR_SUSPEND) {
      FE_LOGD("Node[%s, %s]: the status of anchor %zu is suspend.", op_type.c_str(), op_name.c_str(),
              anchor_index);
      anchor_index++;
      continue;
    }

    if (ge::AnchorUtils::GetStatus(anchor) != ge::ANCHOR_DATA) {
      Status ret = HandleAnchorData(input_index, static_cast<size_t>(anchor->GetIdx()));
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }

    if (input_index >= input_offsets.size()) {
      FE_LOGE("Node[%s, %s]: input index %zu must be less than input offsets size %zu.",
              op_type.c_str(), op_name.c_str(), input_index, input_offsets.size());
      return FAILED;
    }

    int64_t input_offset = input_offsets[input_index];
    FE_LOGD("Node[%s, %s]: input_index=%zu, input_offset=%ld, anchor_status=%d.",
            op_type.c_str(), op_name.c_str(), input_index, input_offset, ge::AnchorUtils::GetStatus(anchor));

    bool is_addr_var = input_index < input_is_addr_var.size() && input_is_addr_var[input_index];
    if (is_addr_var) {
      input_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(input_offset)));
    } else {
      input_addrs_.push_back(context_.dataMemBase + input_offset);
    }

    FE_LOGD("Node[%s, %s]: end to init input.", op_type.c_str(), op_name.c_str());
    input_index++;
    anchor_index++;
  }

  return SUCCESS;
}

Status FftsTaskBuilderAdapter::InitOutput() {
  // Verify output number.
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  size_t output_num = op_desc_->GetOutputsSize();
  vector<int64_t> output_offsets = op_desc_->GetOutputOffset();
  // if output_offsets is empty, set 0 to vector
  if (output_offsets.empty()) {
    vector<int64_t> output_offset_zero(output_num, 0);
    output_offsets.swap(output_offset_zero);
    FE_LOGD("Node[type=%s,name=%s]: output_offset_size=%zu.", op_desc_->GetType().c_str(), op_desc_->GetName().c_str(),
            output_offsets.size());
  }

  if (output_num != output_offsets.size()) {
    FE_LOGE(
        "Node[type=%s,name=%s]: output_desc_size != output_offset_size, output_desc_size:%zu, output_offset_size:%zu.",
        op_desc_->GetType().c_str(), op_desc_->GetName().c_str(), output_num, output_offsets.size());
    return FAILED;
  }
  vector<bool> output_is_addr_var;
  (void)ge::AttrUtils::GetListBool(op_desc_, ATTR_NAME_OUTPUT_IS_VAR, output_is_addr_var);
  for (size_t i = 0; i < output_num; ++i) {
    auto output_desc = op_desc_->GetOutputDesc(i);
    if (IsMemoryEmpty(output_desc)) {
      FE_LOGI("Node[type=%s,name=%s]: the output %s is memory empty.", op_desc_->GetType().c_str(),
              op_desc_->GetName().c_str(), op_desc_->GetOutputNameByIndex(i).c_str());
      continue;
    }

    int64_t output_offset = output_offsets[i];
    if (output_is_addr_var.size() > i && output_is_addr_var[i]) {
      output_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(output_offset)));
    } else {
      output_addrs_.push_back(context_.dataMemBase + output_offset);
    }
    FE_LOGD("Node[type=%s,name=%s]: output_index=%zu, output_offset=%ld.", op_type.c_str(), op_name.c_str(), i,
            output_offset);
  }

  FE_LOGD("Node[type=%s,name=%s]: init output.", op_desc_->GetType().c_str(), op_desc_->GetName().c_str());
  return SUCCESS;
}

Status FftsTaskBuilderAdapter::InitWorkspace() {
  vector<int64_t> workspace_sizes = op_desc_->GetWorkspaceBytes();
  vector<int64_t> workspace_offsets = op_desc_->GetWorkspace();
  int64_t non_tail_workspace_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc_, fe::NON_TAIL_WORKSPACE_SIZE, non_tail_workspace_size);

  if (!is_unknown_graph_) {
    if (workspace_offsets.size() != workspace_sizes.size()) {
      FE_LOGE("workspaceOffsets.size()[%zu] != workspace_sizes.size()[%zu]", workspace_offsets.size(),
              workspace_sizes.size());
      return FAILED;
    }
  }

  size_t workspace_num = workspace_sizes.size();
  for (size_t i = 0; i < workspace_num; i++) {
    auto workspace_size = workspace_sizes[i];
    int64_t workspace_offset = 0;
    if (i < workspace_offsets.size()) {
      workspace_offset = workspace_offsets[i];
    }
    if (!is_unknown_graph_) {
      FE_LOGD("Op[name=%s,type=%s]: is_unknown_graph_value flag is false.", op_desc_->GetName().c_str(),
              op_desc_->GetType().c_str());
      Status status = CheckOffsetAndSize(workspace_offset, static_cast<int64_t>(workspace_size), context_.dataMemSize);
      if (status != SUCCESS) {
        FE_LOGE("Check offset and size of workspace (op: %s, workspace index: %zu) Failed!",
                op_desc_->GetName().c_str(), i);
        return status;
      }
    }

    workspace_sizes_.push_back(workspace_size);
    workspace_addrs_.push_back(workspace_size == 0 ? nullptr : context_.dataMemBase + workspace_offset);
  }

  // get thread_workspace_sizes_ and thread_workspace_addrs_
  ffts::ThreadSliceMapPtr slice_info_ptr_ = nullptr;
  slice_info_ptr_ = op_desc_->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr_);
  FE_CHECK_NOTNULL(slice_info_ptr_);
  uint32_t thread_num = slice_info_ptr_->slice_instance_num;
  vector<uint32_t> one_thread_workspace_sizes;
  vector<void *> one_thread_workspace_addrs;
  if (static_cast<size_t>(non_tail_workspace_size) > workspace_sizes_.size()) {
    FE_LOGE("non_tail_workspace_size[%ld] > workspace_sizes_.size()[%zu].",
            non_tail_workspace_size, workspace_sizes.size());
    return FAILED;
  }
  for (size_t i = 0; i < static_cast<size_t>(non_tail_workspace_size); i++) {
      one_thread_workspace_sizes.push_back(
          workspace_sizes_[i] == UNKNOWN_WORK_SPACE_SIZE ? workspace_sizes_[i] : (workspace_sizes_[i] / thread_num));
      one_thread_workspace_addrs.push_back(workspace_addrs_[i]);
  }
  thread_workspace_sizes_.push_back(one_thread_workspace_sizes);
  thread_workspace_addrs_.push_back(one_thread_workspace_addrs);
  one_thread_workspace_sizes.clear();
  one_thread_workspace_addrs.clear();
  for (size_t i = static_cast<size_t>(non_tail_workspace_size); i < workspace_sizes_.size(); i++) {
      one_thread_workspace_sizes.push_back(workspace_sizes_[i]);
      one_thread_workspace_addrs.push_back(workspace_addrs_[i]);
  }
  thread_workspace_sizes_.push_back(one_thread_workspace_sizes);
  thread_workspace_addrs_.push_back(one_thread_workspace_addrs);

  for (size_t j = 0; j < static_cast<size_t>(non_tail_workspace_size); j++) {
    thread_addr_offset_.push_back(
        workspace_sizes_[j] == UNKNOWN_WORK_SPACE_SIZE ? workspace_sizes_[j] : (workspace_sizes_[j] / thread_num));
  }

  return SUCCESS;
}

Status FftsTaskBuilderAdapter::VerifyWeights() const {
  // Verify weight offset.
  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(node_);
  for (const ge::ConstGeTensorPtr& weight : weights) {
    int64_t weight_offset = 0;
    if (ge::TensorUtils::GetDataOffset(weight->GetTensorDesc(), weight_offset) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Get weight offset failed. opname[%s]", op_desc_->GetName().c_str());
      return FAILED;
    }

    uint64_t weight_size = ge::TensorUtils::GetWeightSize(weight);
    Status status = CheckOffsetAndSize(weight_offset, weight_size, context_.weightMemSize);
    if (status != SUCCESS) {
      FE_LOGE("Check offset and size of weight failed. opname[%s]", op_desc_->GetName().c_str());
      return status;
    }
  }

  return SUCCESS;
}
 
Status FftsTaskBuilderAdapter::Init() {
  FE_LOGD("Init begin, node name:%s, type:%s.", node_.GetName().c_str(), node_.GetType().c_str());
  FE_CHECK_NOTNULL(op_desc_);
  is_unknown_graph_ = FeGraphCommon::IsNodeOfUnknownRootGraph(node_);
  FE_LOGD("Op[name=%s,type=%s]: is_unknown_graph flag is %ld.", op_desc_->GetName().c_str(),
          op_desc_->GetType().c_str(), is_unknown_graph_);

  // Init input
  Status status = InitInput();
  if (status != SUCCESS) {
    FE_LOGE("InitInput failed. op name:%s", op_desc_->GetName().c_str());
    return status;
  }

  // Init output
  status = InitOutput();
  if (status != SUCCESS) {
    FE_LOGE("InitOutput failed. op name:%s", op_desc_->GetName().c_str());
    return status;
  }

  if (!is_unknown_graph_) {
    // Verify weight offset.
    status = VerifyWeights();
    if (status != SUCCESS) {
      FE_LOGE("VerifyWeights failed. op name:%s", op_desc_->GetName().c_str());
      return status;
    }
  }

  status = ThreadInit();
  if (status != SUCCESS) {
    FE_LOGE("ThreadInit failed, op[%s].", op_desc_->GetName().c_str());
    return status;
  }

  // Init workspace
  status = InitWorkspace();
  if (status != SUCCESS) {
    FE_LOGE("InitWorkspace failed. op name:%s.", op_desc_->GetName().c_str());
    return status;
  }
  
  DebugThreadArgs();
  return SUCCESS;
}

void FftsTaskBuilderAdapter::GetThreadParamOffset(ThreadParamOffset &param_offset) const {
  param_offset.thread_input_addrs = thread_input_addrs_;
  param_offset.thread_output_addrs = thread_output_addrs_;
  param_offset.first_thread_input_addrs = first_thread_input_addrs_;
  param_offset.first_thread_output_addrs = first_thread_output_addrs_;
  param_offset.thread_workspace_addrs = thread_workspace_addrs_;
  param_offset.thread_workspace_sizes = thread_workspace_sizes_;
  param_offset.thread_addr_offset = thread_addr_offset_;
  param_offset.input_tensor_sizes = input_tensor_sizes_;
  param_offset.output_tensor_sizes = output_tensor_sizes_;
  param_offset.thread_dim = thread_dim_;
}
}  // namespace fe

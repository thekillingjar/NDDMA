/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdatarnn_generator.h"
#include <memory>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "framework/common/types.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"

namespace fe {

TransNodeTransdataRNNGenerator::TransNodeTransdataRNNGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr,
                                                               TransInfoPtr trans_info_ptr)
    : TransNodeTransdataGenerator(fe_ops_store_ptr, trans_info_ptr) {}
TransNodeTransdataRNNGenerator::~TransNodeTransdataRNNGenerator() {}

Status TransNodeTransdataRNNGenerator::AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) {
  trans_info_ptr_ = trans_info_ptr;

  ge::Format out_format_new_node = trans_info_ptr->dst_in_primary_format;
  auto out_sub_format = trans_info_ptr->dst_in_sub_format;

  return AddOpAndNode(fused_graph, ge::GeShape(), out_format_new_node, out_sub_format,
                      trans_info_ptr->src_out_data_type);
}

Status TransNodeTransdataRNNGenerator::AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape,
                                                    const ge::Format &primary_format, const int32_t &sub_format,
                                                    const ge::DataType &dtype) {
  TransInfoPtr trans_info_ptr = trans_info_ptr_;
  ge::OpDescPtr op_desc_ptr = CreateBasicOpDescForTransNode(TRANSDATARNN);
  FE_CHECK(op_desc_ptr == nullptr, , return PARAM_INVALID);

  FE_LOGD("Create [%s] node between [%s] and [%s] successfully!", TRANSDATARNN.c_str(),
          trans_info_ptr->src_op_desc->GetName().c_str(), trans_info_ptr->dst_op_desc->GetName().c_str());

  SetAttr(trans_info_ptr, primary_format, sub_format, op_desc_ptr);

  if (AddAndSetTensor(shape, primary_format, sub_format, dtype, op_desc_ptr) != SUCCESS) {
    return FAILED;
  }
  /* set attr to transdata RNN op */
  (void)ge::AttrUtils::SetInt(op_desc_ptr, "hidden_size", trans_info_ptr->hidden_size);
  (void)ge::AttrUtils::SetInt(op_desc_ptr, "input_size", trans_info_ptr->input_size);
  if (trans_info_ptr->state_size != RNN_STATE_SIZE_DEFAULT_VALUE) {
    (void)ge::AttrUtils::SetInt(op_desc_ptr, "state_size", trans_info_ptr->state_size);
  }
  if (!TransNodeCheckSupportedByFormatTune(fused_graph, op_desc_ptr)) {
    FE_LOGW("[GraphOpt][Trans][TransDataRNN] Format conversion from %u to %u for op %s to %s is not supported by %s!",
            trans_info_ptr_->src_out_primary_format, trans_info_ptr_->dst_in_primary_format,
            trans_info_ptr_->src_op_desc->GetName().c_str(), trans_info_ptr_->dst_op_desc->GetName().c_str(),
            op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  // insert new op need add attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
  // for data dump
  std::vector<std::string> original_names;
  if (!ge::AttrUtils::SetListStr(op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names)) {
    FE_LOGE("Set op[%s] attr ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES failed.", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  ge::GeShape new_out_shape;
  ge::GeShape new_in_shape;
  vector<std::pair<int64_t, int64_t>> new_in_range;
  vector<std::pair<int64_t, int64_t>> new_out_range;
  GetShapeOfTransdata(op_desc_ptr, new_in_shape, new_out_shape, new_in_range, new_out_range, primary_format, dtype);
  SetTensorRealDimCountAndNewShape(op_desc_ptr, {new_in_shape}, new_out_shape);
  SetNewShapeRange(op_desc_ptr, new_in_range, new_out_range);
  SetNodeParallelGroupId(op_desc_ptr, trans_info_ptr_->src_op_desc, trans_info_ptr_->dst_op_desc);
  if (AddEdgesAndFreshTransInfo(fused_graph, op_desc_ptr) != SUCCESS) {
    FE_LOGD("Add edge unsuccessful!");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe

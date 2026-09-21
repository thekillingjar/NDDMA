/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace fe {
OpFormatDtypeUpdateDesc::OpFormatDtypeUpdateDesc(FormatDtypeQuerierPtr format_dtype_querier_ptr)
    : op_format_dtype_update_tensor_desc_ptr_(nullptr),
      format_dtype_querier_ptr_(format_dtype_querier_ptr) {}
OpFormatDtypeUpdateDesc::~OpFormatDtypeUpdateDesc() {}

Status OpFormatDtypeUpdateDesc::Initialize() {
  FE_MAKE_SHARED(op_format_dtype_update_tensor_desc_ptr_ =
      std::make_shared<OpFormatDtypeUpdateDescBase>(format_dtype_querier_ptr_), return FAILED);
  FE_CHECK_NOTNULL(op_format_dtype_update_tensor_desc_ptr_);
  return SUCCESS;
}

Status OpFormatDtypeUpdateDesc::UpdateTensorDescInfo(const OpKernelInfoPtr& op_kernel_info_ptr,
                                                    const uint32_t& matched_index,
                                                    const IndexNameMap& tensor_index_name_map, const bool& is_input,
                                                    ge::NodePtr& node_ptr, const bool is_dtype) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  FE_CHECK_NOTNULL(op_kernel_info_ptr);

  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();

  uint32_t tensor_size = is_input ? op_desc_ptr->GetAllInputsSize() : op_desc_ptr->GetAllOutputsDescSize();
  for (uint32_t i = 0; i < tensor_size; ++i) {
    if (is_input && op_desc_ptr->MutableInputDesc(i) == nullptr) {
      continue;
    } else if (!is_input && op_desc_ptr->MutableOutputDesc(i) == nullptr) {
      continue;
    }
    // query input format from op kernel info store
    auto tensor_iter = tensor_index_name_map.find(i);
    if (tensor_iter == tensor_index_name_map.end()) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Node[%s, %s]: %s %u is not found in op store.",
                      op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(is_input), i);
      return OP_JUDGE_MAP_KEY_FIND_FAILED;
    }
    FE_LOGD("[UpdTensorDesc] Update %s Node[%s, %s]: %s map is [%u, %s].", DtypeFormatToString(is_dtype).c_str(),
            op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(is_input), i, tensor_iter->second.c_str());
    InputOrOutputInfoPtr tensor_info_ptr = nullptr;
    Status ret = op_kernel_info_ptr->GetTensorInfoByName(is_input, tensor_iter->second, tensor_info_ptr);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Node[%s, %s]: %s %u is not found in op store.",
                      op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(is_input), i);
      return ret;
    }

    FE_CHECK_NOTNULL(tensor_info_ptr);
    ge::GeTensorDescPtr tensor_desc = is_input ? op_desc_ptr->MutableInputDesc(i) : op_desc_ptr->MutableOutputDesc(i);
    UpdateInfo update_info = {op_kernel_info_ptr, tensor_info_ptr, matched_index, node_ptr, i, *tensor_desc, is_input};
    if (is_dtype) {
      ret = op_format_dtype_update_tensor_desc_ptr_->UpdateDtype(update_info);
    } else {
      ret = op_format_dtype_update_tensor_desc_ptr_->UpdateFormat(update_info);
    }
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][UpdTensorDesc] Failed to update %s for node [%s, %s].",
                     DtypeFormatToString(is_dtype).c_str(), node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status OpFormatDtypeUpdateDesc::UpdateTensorDtypeInfo(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     const uint32_t &matched_index,
                                                     const IndexNameMap &tensor_index_name_map, const bool &is_input,
                                                     ge::NodePtr &node_ptr) {
  return UpdateTensorDescInfo(op_kernel_info_ptr, matched_index, tensor_index_name_map,
                              is_input, node_ptr);
}

Status OpFormatDtypeUpdateDesc::UpdateTensorFormatInfo(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const uint32_t &matched_index,
                                                      const IndexNameMap &tensor_index_name_map, const bool &is_input,
                                                      ge::NodePtr &node_ptr) {
  return UpdateTensorDescInfo(op_kernel_info_ptr, matched_index, tensor_index_name_map,
                              is_input, node_ptr, false);
}

const std::string OpFormatDtypeUpdateDesc::DtypeFormatToString(bool is_dtype) {
  if (is_dtype) {
    return "dtype";
  } else {
    return "format";
  }
}
}  // namespace fe

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/get_original_format_pass.h"

#include <vector>

#include "framework/omg/omg_inner_types.h"
#include "graph/utils/op_desc_utils.h"
#include "common/context/local_context.h"
#include "graph/utils/op_type_utils.h"

using domi::DOMI_TENSOR_NCHW;
using domi::DOMI_TENSOR_NHWC;
using domi::DOMI_TENSOR_RESERVED;

namespace ge {
Status GetOriginalFormatPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GE_RETURN_WITH_LOG_IF_ERROR(SetOriginalFormat(graph),
                              "[Set][OriginalFormat] for graph:%s failed", graph->GetName().c_str());

  return SUCCESS;
}

Status GetOriginalFormatPass::SetOriginalFormat(const ge::ComputeGraphPtr &graph) const {
  int64_t ori_format = 0;
  int64_t tmp_format = 0;

  for (auto &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);

    GE_IF_BOOL_EXEC(!AttrUtils::SetInt(node_ptr->GetOpDesc(), ATTR_NAME_INFERRED_FORMAT, DOMI_TENSOR_RESERVED),
                    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                      ATTR_NAME_INFERRED_FORMAT.c_str(),
                                      node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
                    GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INFERRED_FORMAT.c_str(),
                           node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
                    return FAILED);
  }

  for (auto &node_ptr : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    OpDescPtr desc_ptr = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(desc_ptr);
    auto is_data = OpTypeUtils::IsDataNode(desc_ptr->GetType());
    if (is_data) {
      GELOGI("Data node: %s, format: %d", node_ptr->GetName().c_str(), GetLocalOmgContext().format);
      ori_format = static_cast<int64_t>(GetLocalOmgContext().format);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(desc_ptr, ATTR_NAME_FORMAT, ori_format),
                      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                        ATTR_NAME_FORMAT.c_str(),
                                        desc_ptr->GetName().c_str(), desc_ptr->GetType().c_str());
                      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_FORMAT.c_str(),
                             desc_ptr->GetName().c_str(), desc_ptr->GetType().c_str());
                      return FAILED);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(desc_ptr, ATTR_NAME_INFERRED_FORMAT, ori_format),
                      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                        ATTR_NAME_INFERRED_FORMAT.c_str(),
                                        desc_ptr->GetName().c_str(), desc_ptr->GetType().c_str());
                      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INFERRED_FORMAT.c_str(),
                             desc_ptr->GetName().c_str(), desc_ptr->GetType().c_str());
                      return FAILED);
      continue;
    }
    int32_t i = 0;
    bool continue_flag = false;
    bool ignore_pred_format = false;
    for (auto &bias_node_ptr : node_ptr->GetInDataNodes()) {
      GE_CHECK_NOTNULL(bias_node_ptr);

      OpDescPtr bias_op_ptr = bias_node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(bias_op_ptr);

      if (bias_op_ptr->GetType() == BIASADD) {
        ignore_pred_format = true;
        std::size_t tmp_size = ge::OpDescUtils::GetNonConstInputsSize(bias_node_ptr);
        GE_IF_BOOL_EXEC(tmp_size > 2 || tmp_size == 0,
                        GELOGW("bias_node is node followed by %zu nodes, should be 1 or 2", tmp_size);
                        continue_flag = true; break);
        OpDescPtr tmp_first_op_ptr = bias_node_ptr->GetInDataNodes().at(0)->GetOpDesc();
        GE_CHECK_NOTNULL(tmp_first_op_ptr);
        bias_op_ptr = tmp_first_op_ptr;

        // if biasadd have 2 input edges, format should be same
        if (tmp_size == 2) {
          int64_t first_input_format = 0;
          int64_t second_input_format = 0;
          OpDescPtr tmpSecondOpPtr = bias_node_ptr->GetInDataNodes().at(1)->GetOpDesc();
          GE_CHECK_NOTNULL(tmpSecondOpPtr);
          GE_IF_BOOL_EXEC(
              !AttrUtils::GetInt(tmp_first_op_ptr, ATTR_NAME_FORMAT, first_input_format), continue_flag = true; break);
          GE_IF_BOOL_EXEC(
              !AttrUtils::GetInt(tmpSecondOpPtr, ATTR_NAME_FORMAT, second_input_format), continue_flag = true; break);

          if (first_input_format != second_input_format) {
            GELOGW("biasadd node is followed two nodes with different format, get original format failed");
            continue_flag = true;
            break;
          }
        }
      }
      GE_IF_BOOL_EXEC(!AttrUtils::GetInt(bias_op_ptr, ATTR_NAME_FORMAT, tmp_format), continue_flag = true; break);
      if (i == 0) {
        ori_format = tmp_format;
      }

      GE_IF_BOOL_EXEC(tmp_format != ori_format,
                      GELOGW("node: %s , original format of src nodes must be same!", bias_node_ptr->GetName().c_str());
                      continue_flag = true; break);

      i++;
    }

    GE_IF_BOOL_EXEC(continue_flag, continue);
    OpDescPtr tmp_op_ptr = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(tmp_op_ptr);

    if (IsFormatTranspose(tmp_op_ptr, static_cast<int32_t>(ori_format))) {
      ori_format = (ori_format == DOMI_TENSOR_NCHW) ? DOMI_TENSOR_NHWC : DOMI_TENSOR_NCHW;
    }

    if (ignore_pred_format) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetBool(tmp_op_ptr, ATTR_NAME_IGNORE_PRED_FORMAT, true),
                      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                        ATTR_NAME_IGNORE_PRED_FORMAT.c_str(),
                                        tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_IGNORE_PRED_FORMAT.c_str(),
                             tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      return FAILED);
    }

    // Do not reset ATTR_NAME_FORMAT if it is set in the OpParser.
    if (!tmp_op_ptr->HasAttr(ATTR_NAME_FORMAT)) {
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_FORMAT, ori_format),
                      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                        ATTR_NAME_FORMAT.c_str(),
                                        tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_FORMAT.c_str(),
                             tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      return FAILED);
      GE_IF_BOOL_EXEC(!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_INFERRED_FORMAT, ori_format),
                      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                                        ATTR_NAME_INFERRED_FORMAT.c_str(),
                                        tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INFERRED_FORMAT.c_str(),
                             tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
                      return FAILED);
    } else {
      int64_t existingFormat = 0;
      GE_RETURN_WITH_LOG_IF_FALSE(AttrUtils::GetInt(tmp_op_ptr, ATTR_NAME_FORMAT, existingFormat),
                                  "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_FORMAT.c_str(),
                                  tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
      if (!AttrUtils::SetInt(tmp_op_ptr, ATTR_NAME_INFERRED_FORMAT, existingFormat)) {
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed",
                          ATTR_NAME_INFERRED_FORMAT.c_str(),
                          tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
        GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_INFERRED_FORMAT.c_str(),
               tmp_op_ptr->GetName().c_str(), tmp_op_ptr->GetType().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

bool GetOriginalFormatPass::IsFormatTranspose(const ge::OpDescPtr op_ptr, int32_t ori_format) const {
  GE_CHK_BOOL_EXEC(op_ptr != nullptr, return false, "[Check][Param] op_ptr is nullptr");
  if (op_ptr->GetType() == PERMUTE) {
    std::vector<int32_t> index_list;
    GE_IF_BOOL_EXEC(!AttrUtils::GetListInt(op_ptr, PERMUTE_ATTR_ORDER, index_list), return false);

    auto index_size = index_list.size();

    GE_IF_BOOL_EXEC(static_cast<int32_t>(index_size) != PERMUTE_ORDER_NUM, return false);

    int32_t perm_nchw[4] = {0, 2, 3, 1};  // 4 format nums, {0,2,3,1} NCHW -> NHWC
    int32_t perm_nhwc[4] = {0, 3, 1, 2};  // 4 format nums, {0,3,1,2} NHWC -> NCHW
    bool is_nchw = true;
    bool is_nhwc = true;
    for (size_t i = 0; i < index_size; ++i) {
      is_nchw = (perm_nchw[i] != index_list[i]) ? false : is_nchw;
      is_nhwc = (perm_nhwc[i] != index_list[i]) ? false : is_nhwc;
    }
    bool ret = (is_nchw && ori_format == DOMI_TENSOR_NCHW && !is_nhwc) ||
               (is_nhwc && ori_format == DOMI_TENSOR_NHWC && !is_nchw);

    return ret;
  }
  return false;
}
}  // namespace ge

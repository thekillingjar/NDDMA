/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump_util.h"
#include <sstream>
#include "common/fe_log.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "common/string_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace fe {
using ToOpStructPtr = std::shared_ptr<ToOpStruct_t>;

void DumpOpInfoTensor(const std::vector<te::TbeOpParam> &op_params, std::string &debug_str) {
  std::ostringstream debug_oss;
  for (size_t i = 0; i < op_params.size(); i++) {
    debug_oss << std::endl << "tensor[" << std::to_string(i) << "]";
    debug_oss << std::endl << "tensor type : " << std::to_string(op_params.at(i).GetType());
    const std::vector<te::TbeOpTensor> &tensors = op_params.at(i).GetTensors();
    for (const te::TbeOpTensor &tensor : tensors) {
      debug_oss << std::endl << "name : " << tensor.GetName();
      debug_oss << std::endl << "shape : [" << StringUtils::IntegerVecToString(tensor.GetShape()) << "]";
      debug_oss << std::endl << "origin shape : [" << StringUtils::IntegerVecToString(tensor.GetOriginShape()) << "]";
      debug_oss << std::endl << "shape type : " << std::to_string(tensor.GetShapeType());
      debug_oss << std::endl << "format : " << tensor.GetFormat();
      debug_oss << std::endl << "sub_format : " << tensor.GetSubFormat();
      debug_oss << std::endl << "origin format : " << tensor.GetOriginFormat();
      debug_oss << std::endl << "dtype : " << tensor.GetType();
      debug_oss << std::endl << "addr type : " << std::to_string(tensor.GetAddrType());
      debug_oss << std::endl << "L1WorkspaceFlag : " << std::to_string(tensor.GetL1WorkspaceFlag());
      debug_oss << std::endl << "slice offset : " << StringUtils::IntegerVecToString(tensor.GetSliceOffset());
      debug_oss << std::endl << "valid shape : " << StringUtils::IntegerVecToString(tensor.GetValidShape());
      debug_oss << std::endl << "sgt slice shape : " << StringUtils::IntegerVecToString(tensor.GetSgtSliceShape());
    }
  }
  debug_oss << std::endl;
  debug_str = debug_oss.str();
}

void DumpOpInfo(const te::TbeOpInfo &op_info) {
  std::ostringstream debug_oss;
  debug_oss << std::endl << "name : " << op_info.GetName();
  debug_oss << std::endl << "OpFileName : " << op_info.GetOpFileName();
  debug_oss << std::endl << "OpFuncName : " << op_info.GetOpFuncName();
  debug_oss << std::endl << "CheckSupportedFunc : " << op_info.GetCheckSupportedFunc();
  debug_oss << std::endl << "Pattern : " << op_info.GetPattern();
  debug_oss << std::endl << "KernelName : " << op_info.GetKernelName();
  debug_oss << std::endl << "L1Space : " << std::to_string(op_info.GetL1Space());
  debug_oss << std::endl;
  FE_LOGD("Dump Op info. %s.", debug_oss.str().c_str());

  string debug_str;
  DumpOpInfoTensor(op_info.GetInputs(), debug_str);
  FE_LOGD("Dump Op info inputs, %s.", debug_str.c_str());

  debug_str.clear();
  DumpOpInfoTensor(op_info.GetOutputs(), debug_str);
  FE_LOGD("Dump Op info outputs, %s.", debug_str.c_str());
}

void DumpL1Attr(const ge::Node *node) {
  auto op_desc = node->GetOpDesc();
  vector<int32_t> in_memery_type_vec;
  vector<int32_t> out_memery_type_vec;
  ToOpStructPtr l1_info = std::shared_ptr<ToOpStruct_t>();
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memery_type_vec);
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memery_type_vec);
  l1_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, l1_info);
  string in_mem_type_str;
  string out_mem_type_str;
  string op_l1_fusion_type_str;
  string slice_input_offset_str;
  string slice_output_offset_str;
  string slice_input_shape_str;
  string slice_output_shape_str;
  FE_CHECK(l1_info == nullptr, FE_LOGD("l1Info is null."), return);
  for (unsigned int i = 0; i < in_memery_type_vec.size(); i++) {
    in_mem_type_str += std::to_string(in_memery_type_vec[i]) + " ";
  }

  for (unsigned int i = 0; i < out_memery_type_vec.size(); i++) {
    out_mem_type_str += std::to_string(out_memery_type_vec[i]) + " ";
  }

  for (unsigned int i = 0; i < l1_info->op_l1_fusion_type.size(); i++) {
    op_l1_fusion_type_str += " L1 fusion type" + std::to_string(i) + ": ";
    for (auto &type : l1_info->op_l1_fusion_type) {
      op_l1_fusion_type_str += std::to_string(type) + " ";
    }
  }

  for (unsigned int i = 0; i < l1_info->slice_input_shape.size(); i++) {
    slice_input_shape_str += " valid shape input" + std::to_string(i) + ": ";
    for (auto &valid_input_shape : l1_info->slice_input_shape.at(i)) {
      slice_input_shape_str += std::to_string(valid_input_shape) + " ";
    }
  }

  for (unsigned int i = 0; i < l1_info->slice_output_shape.size(); i++) {
    slice_output_shape_str += " valid shape output" + std::to_string(i) + ": ";
    for (auto &valid_output_shape : l1_info->slice_output_shape.at(i)) {
      slice_output_shape_str += std::to_string(valid_output_shape) + " ";
    }
  }

  for (unsigned int i = 0; i < l1_info->slice_input_offset.size(); i++) {
    slice_input_offset_str += " slice offset input" + std::to_string(i) + ": ";
    for (auto &slice_offset : l1_info->slice_input_offset.at(i)) {
      slice_input_offset_str += std::to_string(slice_offset) + " ";
    }
  }

  for (unsigned int i = 0; i < l1_info->slice_output_offset.size(); i++) {
    slice_output_offset_str += " slice offset output" + std::to_string(i) + ": ";
    for (auto &slice_offset : l1_info->slice_output_offset.at(i)) {
      slice_output_offset_str += std::to_string(slice_offset) + " ";
    }
  }

  FE_LOGD("Dump L1 Op info[%s, %s], input size [%zu], type [%s], output size [%zu], type [%s].",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), in_memery_type_vec.size(), in_mem_type_str.c_str(),
          out_memery_type_vec.size(), out_mem_type_str.c_str());

  FE_LOGD("Dump L1 Op info. op_l1_fusion_type size [%zu], type [%s],"
          " slice_input_shape size [%zu], type [%s]. slice_output_shape size [%zu], type [%s],"
          " slice_input_offset size [%zu], type [%s]. slice_output_offset size [%zu], type [%s], l1space [%ld].",
          l1_info->op_l1_fusion_type.size(), op_l1_fusion_type_str.c_str(), l1_info->slice_input_shape.size(),
          slice_input_shape_str.c_str(), l1_info->slice_output_shape.size(), slice_output_shape_str.c_str(),
          l1_info->slice_input_offset.size(), slice_input_offset_str.c_str(), l1_info->slice_output_offset.size(),
          slice_output_offset_str.c_str(), l1_info->op_l1_space);
}

void DumpL2Attr(const ge::Node *node) {
  auto op_desc = node->GetOpDesc();
  vector<int32_t> in_memery_type_vec;
  vector<int32_t> out_memery_type_vec;
  ToOpStructPtr l2_info = std::shared_ptr<ToOpStruct_t>();
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, in_memery_type_vec);
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, out_memery_type_vec);
  l2_info = op_desc->TryGetExtAttr(ATTR_NAME_L2_FUSION_EXTEND_PTR, l2_info);
  string in_mem_type_str;
  string out_mem_type_str;
  string slice_input_offset_str;
  string slice_output_offset_str;
  string slice_input_shape_str;
  string slice_output_shape_str;
  FE_CHECK(l2_info == nullptr, FE_LOGD("l2Info is null."), return);
  for (unsigned int i = 0; i < in_memery_type_vec.size(); i++) {
    in_mem_type_str += std::to_string(in_memery_type_vec[i]) + " ";
  }

  for (unsigned int i = 0; i < out_memery_type_vec.size(); i++) {
    out_mem_type_str += std::to_string(out_memery_type_vec[i]) + " ";
  }

  for (unsigned int i = 0; i < l2_info->slice_input_shape.size(); i++) {
    slice_input_shape_str += " valid shape input" + std::to_string(i) + ": ";
    for (auto &valid_input_shape : l2_info->slice_input_shape.at(i)) {
      slice_input_shape_str += std::to_string(valid_input_shape) + " ";
    }
  }

  for (unsigned int i = 0; i < l2_info->slice_output_shape.size(); i++) {
    slice_output_shape_str += " valid shape output" + std::to_string(i) + ": ";
    for (auto &valid_output_shape : l2_info->slice_output_shape.at(i)) {
      slice_output_shape_str += std::to_string(valid_output_shape) + " ";
    }
  }

  for (unsigned int i = 0; i < l2_info->slice_input_offset.size(); i++) {
    slice_input_offset_str += " slice offset input" + std::to_string(i) + ": ";
    for (auto &slice_offset : l2_info->slice_input_offset.at(i)) {
      slice_input_offset_str += std::to_string(slice_offset) + " ";
    }
  }

  for (unsigned int i = 0; i < l2_info->slice_output_offset.size(); i++) {
    slice_output_offset_str += " slice offset output" + std::to_string(i) + ": ";
    for (auto &slice_offset : l2_info->slice_output_offset.at(i)) {
      slice_output_offset_str += std::to_string(slice_offset) + " ";
    }
  }

  FE_LOGD("Dump L2 Op info[%s, %s], input size [%zu], type [%s], output size [%zu], type [%s].",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), in_memery_type_vec.size(), in_mem_type_str.c_str(),
          out_memery_type_vec.size(), out_mem_type_str.c_str());

  FE_LOGD("Dump L2 Op info."
          " slice_input_shape size [%zu], type [%s]. slice_output_shape size [%zu], type [%s],"
          " slice_input_offset size [%zu], type [%s]. slice_output_offset size [%zu], type [%s].",
          l2_info->slice_input_shape.size(), slice_input_shape_str.c_str(), l2_info->slice_output_shape.size(),
          slice_output_shape_str.c_str(), l2_info->slice_input_offset.size(), slice_input_offset_str.c_str(),
          l2_info->slice_output_offset.size(), slice_output_offset_str.c_str());
}
}  // namespace fe

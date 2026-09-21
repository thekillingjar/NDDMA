/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_def/ir_pb_converter.h"
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "google/protobuf/map.h"
#include "graph/ge_tensor.h"
#include "graph/buffer.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_types.h"
#include "common/util.h"
#include "base/err_msg.h"

namespace ge {
static void ConvertList(const std::pair<std::string, OpAttribute> &op_attr_pair, ge::OpDescPtr op_def) {
  domi::AttrDef_ListValue a_list = op_attr_pair.second.value_.list();

  vector<int64_t> v_i;
  for (int32_t i = 0; i < a_list.i_size(); i++) {
    v_i.push_back(static_cast<int64_t>(a_list.i(i)));
  }
  if (v_i.size() > 0) {
    (void)ge::AttrUtils::SetListInt(op_def, op_attr_pair.first, v_i);
    return;
  }
  vector<float> v_f;
  for (int32_t i = 0; i < a_list.f_size(); i++) {
    v_f.push_back(a_list.f(i));
  }
  if (v_f.size() > 0) {
    (void)ge::AttrUtils::SetListFloat(op_def, op_attr_pair.first, v_f);
    return;
  }
  vector<bool> v_b;
  for (int32_t i = 0; i < a_list.b_size(); i++) {
    v_b.push_back(a_list.b(i));
  }
  if (v_b.size() > 0) {
    (void)ge::AttrUtils::SetListBool(op_def, op_attr_pair.first, v_b);
    return;
  }
  vector<int32_t> v_u;
  for (int32_t i = 0; i < a_list.u_size(); i++) {
    v_u.push_back(static_cast<int32_t>(a_list.u(i)));
  }
  if (v_u.size() > 0) {
    (void)ge::AttrUtils::SetListInt(op_def, op_attr_pair.first, v_u);
    return;
  }
  // set for empty list
  (void)ge::AttrUtils::SetListInt(op_def, op_attr_pair.first, v_i);
  GELOGI("set empty list for node %s attr %s", op_def->GetName().c_str(), op_attr_pair.first.c_str());
}

static void UpdateTensorForOpDesc(const ParserOperator &op, ge::OpDescPtr op_def) {
  if (op_def == nullptr) {
    return;
  }
  uint32_t in_index = 0;
  for (const ge::GeTensorDesc &input_desc : op.GetInputTensorDesc()) {
    if (in_index < op_def->GetInputsSize()) {
      (void)op_def->UpdateInputDesc(in_index++, input_desc);
    } else {
      (void)op_def->AddInputDesc(input_desc);
      in_index++;
    }
  }

  uint32_t out_index = 0;
  for (const ge::GeTensorDesc &output_desc : op.GetOutputTensorDesc()) {
    if (out_index < op_def->GetOutputsSize()) {
      op_def->UpdateOutputDesc(out_index++, output_desc);
    } else {
      op_def->AddOutputDesc(output_desc);
      out_index++;
    }
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status ConvertToOpDesc(const ParserOperator &op,
                                                                              ge::OpDescPtr &op_def) {
  if (op_def == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param op_def is nullptr, check invalid.");
    GELOGE(ge::FAILED, "[Check][Param] param op_def is nullptr, check invalid.");
    return ge::FAILED;
  }
  op_def->SetName(op.GetName());
  ge::OpDescUtilsEx::SetType(op_def, op.GetType());
  GE_IF_BOOL_EXEC(op.GetType() == ge::parser::YOLO, ge::OpDescUtilsEx::SetType(op_def, ge::parser::REGION));

  UpdateTensorForOpDesc(op, op_def);
  GELOGD("Convert to op desc: name:%s, input size: %zu, output size:%zu", op_def->GetName().c_str(),
         op_def->GetInputsSize(), op_def->GetOutputsSize());

  for (const auto &op_attr_pair : op.GetOpAttrs()) {
    if (op_attr_pair.second.value_.has_list()) {
      ConvertList(op_attr_pair, op_def);
    } else {
      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kBt) {
        auto &buffer = op_attr_pair.second.value_.bt();
        (void)ge::AttrUtils::SetZeroCopyBytes(op_def, op_attr_pair.first,
            ge::Buffer::CopyFrom(PtrToPtr<void, uint8_t>(const_cast<char *>(buffer.data())), buffer.size()));
      }

      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kS) {
        (void)ge::AttrUtils::SetStr(op_def, op_attr_pair.first, op_attr_pair.second.value_.s());
      }
      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kI) {
        (void)ge::AttrUtils::SetInt(op_def, op_attr_pair.first, op_attr_pair.second.value_.i());
      }
      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kF) {
        (void)ge::AttrUtils::SetFloat(op_def, op_attr_pair.first, op_attr_pair.second.value_.f());
      }
      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kB) {
        (void)ge::AttrUtils::SetBool(op_def, op_attr_pair.first, op_attr_pair.second.value_.b());
      }
      if (op_attr_pair.second.value_.value_case() == domi::AttrDef::kU) {
        (void)ge::AttrUtils::SetInt(op_def, op_attr_pair.first, op_attr_pair.second.value_.u());
      }
    }
  }
  return domi::SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status ConvertFromOpDesc(const ge::OpDescPtr op_def,
                                                                                ParserOperator &op) {
  if (op_def == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param op_def is nullptr, check invalid.");
    GELOGE(ge::FAILED, "[Check][Param] param op_def is nullptr, check invalid.");
    return ge::FAILED;
  }
  op.Name(op_def->GetName());

  map<string, ge::GeAttrValue> allattrs = op_def->GetAllAttrs();

  for (const auto &attr : allattrs) {
    ge::GeAttrValue::ValueType v_t = attr.second.GetValueType();
    switch (v_t) {
      case ge::GeAttrValue::ValueType::VT_LIST_STRING: {
        std::vector<string> vec;
        (void)ge::AttrUtils::GetListStr(op_def, attr.first, vec);
        op.Attr(attr.first, vec);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_LIST_FLOAT: {
        std::vector<float> vec;
        (void)ge::AttrUtils::GetListFloat(op_def, attr.first, vec);
        op.Attr(attr.first, vec);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_LIST_BOOL: {
        std::vector<bool> vec;
        (void)ge::AttrUtils::GetListBool(op_def, attr.first, vec);
        op.Attr(attr.first, vec);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_LIST_INT: {
        std::vector<int64_t> vec;
        (void)ge::AttrUtils::GetListInt(op_def, attr.first, vec);
        op.Attr(attr.first, vec);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_STRING: {
        string s = "";
        (void)ge::AttrUtils::GetStr(op_def, attr.first, s);
        op.Attr(attr.first, s);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_FLOAT: {
        float f = 0.0;
        (void)ge::AttrUtils::GetFloat(op_def, attr.first, f);
        op.Attr(attr.first, f);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_BOOL: {
        bool b = false;
        (void)ge::AttrUtils::GetBool(op_def, attr.first, b);
        op.Attr(attr.first, b);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_INT: {
        int64_t i = 0;
        (void)ge::AttrUtils::GetInt(op_def, attr.first, i);
        op.Attr(attr.first, i);
        break;
      }
      default:
        break;
    }
  }

  return domi::SUCCESS;
}
}  // namespace ge

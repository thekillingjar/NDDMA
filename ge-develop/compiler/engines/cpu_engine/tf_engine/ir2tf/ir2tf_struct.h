/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_IR2TF_STRUCT_H_
#define AICPU_IR2TF_STRUCT_H_

#include <string>
#include <vector>

namespace aicpu {
struct RefTransDesc {
  // Source input/output name.
  std::string src_inout_name;
  // Target input/output name.
  std::string dst_inout_name;
  // target field is ref type or not, setted for mapping between ref and variable.
  // when IR2TF,it can be true.when TF2IR,it always be false.
  bool is_ref;
};

struct ParserExpDesc {
  // Source field name.
  std::string src_field_name;
  // Target field name.
  std::string dst_field_name;
  // Mapping between src and dst should follow this rule.
  // If empty, means srcFieldName=dstFieldName.
  std::string parser_express;
};

struct DynamicExpDesc {
  // dynamic index.
  std::string index;
  // dynamic type.
  std::string type;
  // dynamic name
  std::string name;
};

struct ExtendFieldDesc {
  // field name that need be added.
  std::string field_name;
  // enum value belongs to tensorflow's dtype definition
  std::string data_type;
  // default value, need convert string to other dtype; if empty, mean no default value
  std::string default_value;
};

struct OpMapInfo {
  // Source op type, to be converted op.
  std::string src_op_type;
  // Target op type, converted op.
  std::string dst_op_type;
  // op attrs blacklist
  std::string attrs_blacklist;
  // The mapping between src op.attrs and target op.attrs.
  std::vector<ParserExpDesc> attrs_map_desc;
  // If empty, mean is not dynamic type
  std::vector<DynamicExpDesc> attrs_dynamic_desc;
  // The mapping between src op.attrs and target op.input.
  std::vector<ParserExpDesc> attrs_input_map_desc;
  // The mapping between src op.input and target op.attrs.
  std::vector<ParserExpDesc> input_attr_map_desc;
  // The mapping between src op.input and target op.input.
  std::vector<RefTransDesc> input_ref_map_desc;
  // For ref convert.
  // The mapping between src op.attrs and target op.output.
  std::vector<ParserExpDesc> attrs_output_map_desc;
  // The mapping between src op.output and target op.attrs.
  std::vector<ParserExpDesc> output_attr_map_desc;
  // The mapping between src op.output and target op.output.
  std::vector<RefTransDesc> output_ref_desc;
  // For ref convert.
  // generated attr fileds
  std::vector<ExtendFieldDesc> attr_ext_desc;
};

struct IRFMKOpMapLib {
  // version tag.Defined for future use.
  std::string version;
  // Source OP is IR op,target OP is TF op.
  std::vector<OpMapInfo> ir2tf;
  // Source OP is TF op,target OP is IR op.
  std::vector<OpMapInfo> tf2ir;
};
} // namespace aicpu
#endif // AICPU_IR2TF_STRUCT_H_

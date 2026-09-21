/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir2tf_json_file.h"
#include <fstream>
#include "util/log.h"
#include "util/tf_util.h"

namespace aicpu {
// ir mapping json file configuration item: Source input/output name
const std::string kIrMappingConfigSrcInoutName = "srcInOutputName";

// ir mapping json file configuration item: Target input/output name
const std::string kIrMappingConfigDstInoutName = "dstInOutputName";

// ir mapping json file configuration item: field is ref type or not
const std::string kIrMappingConfigIsRef = "isRef";

// ir mapping json file configuration item: output Ref convert
const std::string kIrMappingConfigOutputRefDesc = "outputRefDesc";

// ir mapping json file configuration item: attr extend convert
const std::string kIrMappingConfigAttrExtDesc = "attrExtDesc";

// ir mapping json file configuration item: field name
const std::string kIrMappingConfigFieldName = "fieldName";

// ir mapping json file configuration item: type of field
const std::string kIrMappingConfigDataType = "dataType";

// ir mapping json file configuration item: default value of field
const std::string kIrMappingConfigDefaultValue = "defaultValue";

// ir mapping json file configuration item: all attrs map
const std::string kIrMappingConfigAttrsDesc = "attrsMapDesc";

// ir mapping json file configuration item: source field name
const std::string kIrMappingConfigSrcFieldName = "srcFieldName";

// ir mapping json file configuration item: target field name
const std::string kIrMappingConfigDstFieldName = "dstFieldName";

// ir mapping json file configuration item: parserExpress
const std::string kIrMappingConfigParseExpress = "parserExpress";

// ir mapping json file configuration item: all dynamic desc
const std::string kIrMappingConfigDynamicDesc = "attrsDynamicDesc";

// ir mapping json file configuration item:  index
const std::string kIrMappingConfigIndex = "index";

// ir mapping json file configuration item: type
const std::string kIrMappingConfigType = "type";

// ir mapping json file configuration item: name
const std::string kIrMappingConfigName = "name";

// ir mapping json file configuration item: source op type
const std::string kIrMappingConfigSrcOpType = "srcOpType";

// ir mapping json file configuration item: target op type
const std::string kIrMappingConfigDstOpType = "dstOpType";

// ir mapping json file configuration item: version tag
const std::string kIrMappingConfigVersion = "version";

// ir mapping json file configuration item: source is ir op,target is tf op
const std::string kIrMappingConfigIr2Tf = "IR2TF";

// ir mapping json file configuration item: source is tf op,target is ir op
const std::string kIrMappingConfigTf2Ir = "TF2IR";

// ir mapping json file configuration item: src atrrs to dest input attr mapping
const std::string kIrMappingConfigAttrsInputMapDesc = "attrsInputMapDesc";

// ir mapping json file configuration item: src input attr to dst attr mapping
const std::string kIrMappingConfigInputAttrMapDesc = "inputAttrMapDesc";

// ir mapping json file configuration item: src atrrs to dest output attr mapping
const std::string kIrMappingConfigAttrsOutputMapDesc = "outputAttrMapDesc";

// ir mapping json file configuration item: src output attr to dst attr mapping
const std::string kIrMappingConfigOutputAttrMapDesc = "outputAttrMapDesc";

// ir mapping json file configuration item: src input attr to dst input attr mapping
const std::string kIrMappingConfigInputRefMapDesc = "inputRefMapDesc";

// ir mapping json file configuration item: op attr blacklist
const std::string kIrMappingConfigAttrsBlacklist = "attrsBlacklist";

void from_json(const nlohmann::json &json_read, RefTransDesc &ref_desc) {
  auto iter = json_read.find(kIrMappingConfigSrcInoutName);
  if (iter != json_read.end()) {
    ref_desc.src_inout_name = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigDstInoutName);
  if (iter != json_read.end()) {
    ref_desc.dst_inout_name = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigIsRef);
  if (iter != json_read.end()) {
    ref_desc.is_ref = iter.value().get<bool>();
  }
}

void from_json(const nlohmann::json &json_read, ParserExpDesc &parse_desc) {
  auto iter = json_read.find(kIrMappingConfigSrcFieldName);
  if (iter != json_read.end()) {
    parse_desc.src_field_name = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigDstFieldName);
  if (iter != json_read.end()) {
    parse_desc.dst_field_name = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigParseExpress);
  if (iter != json_read.end()) {
    parse_desc.parser_express = iter.value().get<std::string>();
  }
}

void from_json(const nlohmann::json &json_read, OpMapInfo &op_map_info) {
  auto iter = json_read.find(kIrMappingConfigSrcOpType);
  if (iter != json_read.end()) {
    op_map_info.src_op_type = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigDstOpType);
  if (iter != json_read.end()) {
    op_map_info.dst_op_type = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigAttrsBlacklist);
  if (iter != json_read.end()) {
    op_map_info.attrs_blacklist = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigAttrsDesc);
  if (iter != json_read.end()) {
    op_map_info.attrs_map_desc = iter.value().get<std::vector<ParserExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigAttrsInputMapDesc);
  if (iter != json_read.end()) {
    op_map_info.attrs_input_map_desc = iter.value().get<std::vector<ParserExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigInputAttrMapDesc);
  if (iter != json_read.end()) {
    op_map_info.input_attr_map_desc = iter.value().get<std::vector<ParserExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigInputRefMapDesc);
  if (iter != json_read.end()) {
    op_map_info.input_ref_map_desc = iter.value().get<std::vector<RefTransDesc>>();
  }

  iter = json_read.find(kIrMappingConfigAttrsOutputMapDesc);
  if (iter != json_read.end()) {
    op_map_info.attrs_output_map_desc = iter.value().get<std::vector<ParserExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigDynamicDesc);
  if (iter != json_read.end()) {
    op_map_info.attrs_dynamic_desc = iter.value().get<std::vector<DynamicExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigOutputAttrMapDesc);
  if (iter != json_read.end()) {
    op_map_info.output_attr_map_desc = iter.value().get<std::vector<ParserExpDesc>>();
  }

  iter = json_read.find(kIrMappingConfigOutputRefDesc);
  if (iter != json_read.end()) {
    op_map_info.output_ref_desc = iter.value().get<std::vector<RefTransDesc>>();
  }

  iter = json_read.find(kIrMappingConfigAttrExtDesc);
  if (iter != json_read.end()) {
    op_map_info.attr_ext_desc = iter.value().get<std::vector<ExtendFieldDesc>>();
  }
}

void from_json(const nlohmann::json &json_read, ExtendFieldDesc &ext_desc) {
  auto iter = json_read.find(kIrMappingConfigFieldName);
  if (iter != json_read.end()) {
    ext_desc.field_name = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigDataType);
  if (iter != json_read.end()) {
    ext_desc.data_type = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigDefaultValue);
  if (iter != json_read.end()) {
    ext_desc.default_value = iter.value().get<std::string>();
  }
}

void from_json(const nlohmann::json &json_read, DynamicExpDesc &dynamic_desc) {
  auto iter = json_read.find(kIrMappingConfigIndex);
  if (iter != json_read.end()) {
    dynamic_desc.index = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigType);
  if (iter != json_read.end()) {
    dynamic_desc.type = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigName);
  if (iter != json_read.end()) {
    dynamic_desc.name = iter.value().get<std::string>();
  }
}


void from_json(const nlohmann::json &json_read, IRFMKOpMapLib &ir_map) {
  auto iter = json_read.find(kIrMappingConfigVersion);
  if (iter != json_read.end()) {
    ir_map.version = iter.value().get<std::string>();
  }

  iter = json_read.find(kIrMappingConfigIr2Tf);
  if (iter != json_read.end()) {
    ir_map.ir2tf = iter.value().get<std::vector<OpMapInfo>>();
  }

  iter = json_read.find(kIrMappingConfigTf2Ir);
  if (iter != json_read.end()) {
    ir_map.tf2ir = iter.value().get<std::vector<OpMapInfo>>();
  }
}
} // namespace aicpu

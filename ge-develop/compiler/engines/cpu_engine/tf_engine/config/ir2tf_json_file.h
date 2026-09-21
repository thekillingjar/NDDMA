/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_IR2TF_JSON_FILE_H_
#define AICPU_IR2TF_JSON_FILE_H_

#include <nlohmann/json.hpp>
#include "ir2tf/ir2tf_struct.h"

namespace aicpu {
// ir mapping json file configuration item: Source input/output name
extern const std::string kIrMappingConfigSrcInoutName;

// ir mapping json file configuration item: Target input/output name
extern const std::string kIrMappingConfigDstInoutName;

// ir mapping json file configuration item: field is ref type or not
extern const std::string kIrMappingConfigIsRef;

// ir mapping json file configuration item: output Ref convert
extern const std::string kIrMappingConfigOutputRefDesc;

// ir mapping json file configuration item: attr extend convert
extern const std::string kIrMappingConfigAttrExtDesc;

// ir mapping json file configuration item: field name
extern const std::string kIrMappingConfigFieldName;

// ir mapping json file configuration item: type of field
extern const std::string kIrMappingConfigDataType;

// ir mapping json file configuration item: default value of field
extern const std::string kIrMappingConfigDefaultValue;

// ir mapping json file configuration item: all attrs map
extern const std::string kIrMappingConfigAttrsDesc;

// ir mapping json file configuration item: source field name
extern const std::string kIrMappingConfigSrcFieldName;

// ir mapping json file configuration item: target field name
extern const std::string kIrMappingConfigDstFieldName;

// ir mapping json file configuration item: parserExpress
extern const std::string kIrMappingConfigParseExpress;

// ir mapping json file configuration item: all dynamic desc
extern const std::string kIrMappingConfigDynamicDesc;

// ir mapping json file configuration item:  index
extern const std::string kIrMappingConfigIndex;

// ir mapping json file configuration item: type
extern const std::string kIrMappingConfigType;

// ir mapping json file configuration item: name
extern const std::string kIrMappingConfigName;

// ir mapping json file configuration item: source op type
extern const std::string kIrMappingConfigSrcOpType;

// ir mapping json file configuration item: target op type
extern const std::string kIrMappingConfigDstOpType;

// ir mapping json file configuration item: version tag
extern const std::string kIrMappingConfigVersion;

// ir mapping json file configuration item: source is ir op,target is tf op
extern const std::string kIrMappingConfigIr2Tf;

// ir mapping json file configuration item: source is tf op,target is ir op
extern const std::string kIrMappingConfigTf2Ir;

// ir mapping json file configuration item: src atrrs to dest input attr mapping
extern const std::string kIrMappingConfigAttrsInputMapDesc;

// ir mapping json file configuration item: src input attr to dst attr mapping
extern const std::string kIrMappingConfigInputAttrMapDesc;

// ir mapping json file configuration item: src atrrs to dest output attr mapping
extern const std::string kIrMappingConfigAttrsOutputMapDesc;

// ir mapping json file configuration item: src output attr to dst attr mapping
extern const std::string kIrMappingConfigOutputAttrMapDesc;

// ir mapping json file configuration item: src input attr to dst input attr mapping
extern const std::string kIrMappingConfigInputRefMapDesc;

// ir mapping json file configuration item: op attr blacklist
extern const std::string kIrMappingConfigAttrsBlacklist;

/**
 * RefTransDesc json to struct object function
 * @param json_read read json handle
 * @param ref_desc for ref convert
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, RefTransDesc &ref_desc);

/**
 * ParserExpDesc json to struct object function
 * @param json_read read json handle
 * @param parse_desc for op attrs convert
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, ParserExpDesc &parse_desc);

/**
 * OpMapInfo json to struct object function
 * @param json_read read json handle
 * @param op_map_info all op attrs config
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, OpMapInfo &op_map_info);

/**
 * IRFMKOpMapLib json to struct object function
 * @param json_read read json handle
 * @param ir_map ir to tf config
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, IRFMKOpMapLib &ir_map);

/**
 * DynamicExpDesc json to struct object function
 * @param json_read read json handle
 * @param dynamic_desc ir to tf config
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, DynamicExpDesc &dynamic_desc);

/**
 * ExtendFieldDesc json to struct object function
 * @param json_read read json handle
 * @param ext_desc extend attr config
 * @return whether read file successfully
 */
void from_json(const nlohmann::json &json_read, ExtendFieldDesc &ext_desc);
} // namespace aicpu
#endif // AICPU_IR2TF_JSON_FILE_H_
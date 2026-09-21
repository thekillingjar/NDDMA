/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow_util.h"
#include "util/log.h"
#include "error_code/error_code.h"
#include "util/util.h"

using domi::tensorflow::AttrValue;
using domi::tensorflow::NodeDef;

using AttrValueMap = google::protobuf::Map<std::string, AttrValue>;

namespace aicpu {
bool TensorFlowUtil::AddNodeAttr(const std::string &attr_name, const AttrValue &attr_value, NodeDef *node_def) {
  AICPU_CHECK_NOTNULL_ERRCODE(node_def, false);
  auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_name, attr_value));
  if (!pair.second) {
    AICPUE_LOGD("Insert attr[%s] to op[%s] failed.",
                attr_name.c_str(), node_def->name().c_str());
    return false;
  }
  return true;
}

bool TensorFlowUtil::FindAttrValue(const NodeDef *node_def, const std::string attr_name, AttrValue &attr_value) {
  AICPU_CHECK_NOTNULL_ERRCODE(node_def, false);
  const AttrValueMap &attr = node_def->attr();
  auto iter = attr.find(attr_name);
  if (iter != attr.end()) {
    attr_value = iter->second;
    return true;
  }
  return false;
}
}

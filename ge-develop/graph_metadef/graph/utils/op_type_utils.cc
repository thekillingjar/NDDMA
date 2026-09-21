/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/op_type_utils.h"
#include <unordered_set>
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
namespace {
const std::unordered_set<std::string> kDataOpSet = {DATA, REFDATA, AIPPDATA, ANN_DATA};
const std::unordered_set<std::string> kVariableOpSet = {VARIABLE, VARIABLEV2};
const std::unordered_set<std::string> kAssignOpSet = {
    ASSIGNADD, ASSIGN, ASSIGNSUB, ASSIGNADDVARIABLEOP, ASSIGNSUBVARIABLEOP, ASSIGNVARIABLEOP};
const std::unordered_set<std::string> kIdentityOpSet = {IDENTITY, READVARIABLEOP};
const std::unordered_set<std::string> kConstPlaceHolderOpSet = {CONSTPLACEHOLDER};
const std::unordered_set<std::string> kConstOpSet = {CONSTANT, CONSTANTOP, CONSTPLACEHOLDER};
const std::unordered_set<std::string> kGraphOutputOpSet = {NETOUTPUT};
const std::unordered_set<std::string> kAutofuseNodeSet = {ASC_BC, FUSE_ASC_BC, EMPTY_ASC_BC};
}  // namespace

/**
 * @brief 判断类型是否为Autofuse
 * @param type
 * @return true
 * @return false
 */
bool OpTypeUtils::IsAutofuseNode(const std::string &type) {
  return (kAutofuseNodeSet.count(type) > 0);
}

/**
 * @brief 判断类型是否为Autofuse
 *
 * @param op_desc
 * @return true
 * @return false
 */
bool OpTypeUtils::IsAutofuseNode(const ge::OpDescPtr &op_desc) {
  return IsAutofuseNode(op_desc->GetType());
}

/**
 * @brief 判断类型是否为空tensor的Autofuse算子
 *
 * @param type
 * @return true
 * @return false
 */
bool OpTypeUtils::IsEmptyAutofuseNode(const std::string &type) {
  return (type == EMPTY_ASC_BC);
}

/**
 * @brief 判断类型是否为DATA
 *        其中不包含QueueData, 该算子原型与其他Data不同，只有输出没有输出
 *        且在编译过程中有自由逻辑，不宜一起判断。
 *
 * @param type
 * @return true
 * @return false
 */
bool OpTypeUtils::IsDataNode(const std::string &type) {
  return (kDataOpSet.count(type) > 0);
}
/**
 * @brief 判断类型是否为RefDATA并且为输入节点
 * @param node
 * @return true
 * @return false
 */
bool OpTypeUtils::IsInputRefData(const ge::OpDescPtr &op_desc) {
  if ((op_desc == nullptr) || (op_desc->GetType() != REFDATA)) {
      return false;
  }
  return !AttrUtils::HasAttr(op_desc, REF_VAR_SRC_VAR_NAME);
}

bool OpTypeUtils::IsVariableNode(const std::string &type) {
  return (kVariableOpSet.count(type) > 0);
}

bool OpTypeUtils::IsVarLikeNode(const std::string &type) {
  return IsVariableNode(type) || (type == REFDATA);
}

bool OpTypeUtils::IsAssignLikeNode(const std::string &type) {
  return kAssignOpSet.count(type) > 0U;
}

bool OpTypeUtils::IsIdentityLikeNode(const std::string &type) {
  return kIdentityOpSet.count(type) > 0U;
}

bool OpTypeUtils::IsConstPlaceHolderNode(const std::string &type) {
  return kConstPlaceHolderOpSet.count(type) > 0U;
}

// CONST/CONSTANT/CONSTPLACEHOLDER
bool OpTypeUtils::IsConstNode(const std::string &type) {
  return kConstOpSet.count(type) > 0U;
}

// IsDataNode/IsVariableNode/IsVarLikeNode/IsConstNode
bool OpTypeUtils::IsGraphInputNode(const std::string &type) {
  return (IsDataNode(type)) || (IsVariableNode(type)) || IsVarLikeNode(type) || IsConstNode(type);
}

// NETOUTPUT
bool OpTypeUtils::IsGraphOutputNode(const std::string &type) {
  return (kGraphOutputOpSet.count(type) > 0);
}

/**
 * @brief get the Original Type of FrameworkOp
 *        其中不包含QueueData, 该算子原型与其他Data不同，只有输出没有输出
 *        且在编译过程中有自由逻辑，不宜一起判断。
 *
 * @param [in] node
 * @param [out] type
 * @return graphStatus
 */
graphStatus OpTypeUtils::GetOriginalType(const ge::OpDescPtr &op_desc, std::string &type) {
  GE_CHECK_NOTNULL(op_desc);
  type = op_desc->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return GRAPH_SUCCESS);
  const std::string *orig_type_ptr = ge::AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE);
  if (orig_type_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  type = *orig_type_ptr;
  GELOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return GRAPH_SUCCESS;
}

bool OpTypeUtils::IsSubgraphInnerData(const ge::OpDescPtr &op_desc) {
  return ((op_desc->GetType() == DATA) && op_desc->HasAttr(ATTR_NAME_PARENT_NODE_INDEX));
}
}  // namespace ge

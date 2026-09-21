/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_ADAPTER_COMMON_GET_ATTR_BY_TYPE_H_
#define FUSION_ENGINE_ADAPTER_COMMON_GET_ATTR_BY_TYPE_H_

#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_op_info_common.h"
#include "tensor_engine/tbe_op_info.h"

namespace fe{
using GetAttrValueByType =
    std::function<Status(const ge::OpDesc &, const string &, te::TbeAttrValue &, AttrInfoPtr)>;

using GetPrivateAttrValueByType =
    std::function<void(const ge::OpDesc &, const ge::AnyValue &,
                        te::TbeAttrValue &, const string &)>;

template <typename T>
Status GetAttrValue(const AttrInfoPtr& attrs_info, T &value);

Status GetStrAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                       te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                       te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetFloatAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                         te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetBoolAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                        te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListStrAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                           te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                           te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListListIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                               te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListFloatAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                             te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListBoolAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                            te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetTensorAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                          const te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

Status GetListTensorAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                              const te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info);

void GetStrPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetFloatPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetBoolPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetListStrPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetListIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetListFloatPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetListListIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

void GetListBoolPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name);

const std::map<ge::GeAttrValue::ValueType, GetAttrValueByType> k_attr_get_funcs = {
    {ge::GeAttrValue::VT_STRING, GetStrAttrValue},
    {ge::GeAttrValue::VT_INT, GetIntAttrValue},
    {ge::GeAttrValue::VT_FLOAT, GetFloatAttrValue},
    {ge::GeAttrValue::VT_BOOL, GetBoolAttrValue},
    {ge::GeAttrValue::VT_LIST_STRING, GetListStrAttrValue},
    {ge::GeAttrValue::VT_LIST_INT, GetListIntAttrValue},
    {ge::GeAttrValue::VT_LIST_FLOAT, GetListFloatAttrValue},
    {ge::GeAttrValue::VT_LIST_BOOL, GetListBoolAttrValue},
    {ge::GeAttrValue::VT_LIST_LIST_INT, GetListListIntAttrValue},
    {ge::GeAttrValue::VT_TENSOR, GetTensorAttrValue},
    {ge::GeAttrValue::VT_LIST_TENSOR, GetListTensorAttrValue},
};

const std::map<ge::GeAttrValue::ValueType, GetPrivateAttrValueByType> k_private_attr_get_funcs = {
    {ge::GeAttrValue::VT_STRING, GetStrPrivateAttrValue},
    {ge::GeAttrValue::VT_INT, GetIntPrivateAttrValue},
    {ge::GeAttrValue::VT_FLOAT, GetFloatPrivateAttrValue},
    {ge::GeAttrValue::VT_BOOL, GetBoolPrivateAttrValue},
    {ge::GeAttrValue::VT_LIST_STRING, GetListStrPrivateAttrValue},
    {ge::GeAttrValue::VT_LIST_INT, GetListIntPrivateAttrValue},
    {ge::GeAttrValue::VT_LIST_FLOAT, GetListFloatPrivateAttrValue},
    {ge::GeAttrValue::VT_LIST_BOOL, GetListBoolPrivateAttrValue},
    {ge::GeAttrValue::VT_LIST_LIST_INT, GetListListIntPrivateAttrValue}
};

} // namespace fe
#endif // FUSION_ENGINE_ADAPTER_COMMON_GET_ATTR_BY_TYPE_H_

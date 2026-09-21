/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_ATTR_GROUP_ATTR_GROUP_BASE_H
#define INC_GRAPH_ATTR_GROUP_ATTR_GROUP_BASE_H

#include <memory>
#include "graph/ge_error_codes.h"
#include "graph/type_utils.h"

namespace ge {
namespace proto {
class AttrGroupDef;
}
class AttrHolder;
class AttrGroupsBase {
 public:
  AttrGroupsBase() = default;

  virtual ~AttrGroupsBase() = default;
  virtual graphStatus Serialize(proto::AttrGroupDef &attr_group_def) {
    (void) attr_group_def;
    return GRAPH_SUCCESS;
  }
  virtual graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
    (void) attr_holder;
    (void) attr_group_def;
    return GRAPH_SUCCESS;
  }
  virtual std::unique_ptr<AttrGroupsBase> Clone() = 0;
};

// typeid()方法存在bug，不同的类的typeid()可能相同，此处用模板特化一下，新增AttrGroupsBase子类属性组，需要实现这个方法，返回子类的typeid
// 前置声明子类，同时注意typeid不能与anchor冲突
class AscTensorAttr;
class AscNodeAttr;
class AscGraphAttr;
class SymbolicDescAttr;
class ShapeEnvAttr;
class AutoFuseAttrs;
class AutoFuseGraphAttrs;
template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AttrGroupsBase>();

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AscTensorAttr>();

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AscNodeAttr>();

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AscGraphAttr>();

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<SymbolicDescAttr>();

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<ShapeEnvAttr>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AutoFuseAttrs>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<AutoFuseGraphAttrs>();

} // namespace ge

#endif // INC_GRAPH_ATTR_GROUP_ATTR_GROUP_BASE_H

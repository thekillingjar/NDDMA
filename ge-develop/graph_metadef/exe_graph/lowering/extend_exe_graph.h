/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_
#define AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
namespace gert {
template<typename K, typename V>
inline bool FindValFromMapExtAttr(const ge::ExecuteGraph *exe_graph, const char *attr_name, const K &key, V &val) {
  auto ext_attr = exe_graph->GetExtAttr<std::unordered_map<K, V>>(attr_name);
  if (ext_attr == nullptr) {
    return false;
  }
  const auto iter = ext_attr->find(key);
  if (iter != ext_attr->cend()) {
    val = iter->second;
    return true;
  }
  return false;
}

template<typename K, typename V>
inline void AddKVToMapExtAttr(ge::ExecuteGraph *exe_graph, const char *attr_name, const K &key, const V &val) {
  auto ext_attr = exe_graph->GetExtAttr<std::unordered_map<K, V>>(attr_name);
  if (ext_attr == nullptr) {
    std::unordered_map<K, V> temp_ext_attr {};
    temp_ext_attr[key] = val;
    exe_graph->SetExtAttr(attr_name, temp_ext_attr);
  } else {
    (*ext_attr)[key] = val;
  }
}
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_

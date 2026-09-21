/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "list_list_int_serializer.h"
#include <vector>
#include "graph_metadef/graph/debug/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
graphStatus ListListIntSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  std::vector<std::vector<int64_t>> list_list_value;
  const graphStatus ret = av.GetValue(list_list_value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_list_int attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list_list = def.mutable_list_list_int();
  GE_CHECK_NOTNULL(mutable_list_list);
  mutable_list_list->clear_list_list_i();
  for (const auto &list_value : list_list_value) {
    const auto list_i = mutable_list_list->add_list_list_i();
    GE_CHECK_NOTNULL(list_i);
    for (const int64_t val : list_value) {
      list_i->add_list_i(val);
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus ListListIntSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  std::vector<std::vector<int64_t>> values;
  for (auto idx = 0; idx < def.list_list_int().list_list_i_size(); ++idx) {
    std::vector<int64_t> vec;
    for (auto i = 0; i < def.list_list_int().list_list_i(idx).list_i_size(); ++i) {
      vec.push_back(def.list_list_int().list_list_i(idx).list_i(i));
    }
    values.push_back(vec);
  }

  return av.SetValue(std::move(values));
}

REG_GEIR_SERIALIZER(list_list_int_serializer, ListListIntSerializer,
                    GetTypeId<std::vector<std::vector<int64_t>>>(), proto::AttrDef::kListListInt);
}  // namespace ge

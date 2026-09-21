/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_MODEL_SERIALIZE_H_
#define INC_GRAPH_MODEL_SERIALIZE_H_

#include <map>
#include <string>
#include "graph/buffer.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "external/ge_common/ge_api_types.h"

namespace ge {
class ModelSerialize {
 public:
  Buffer SerializeModel(const Model &model, const bool not_dump_all = false) const;
  Buffer SerializeSeparateModel(const Model &model, const std::string &path, const bool not_dump_all = false) const;
  Buffer SerializeModel(const Model &model, const std::string &path,
                        const bool is_need_separate, const bool not_dump_all = false) const;
  Status SerializeModel(const Model &model, const bool not_dump_all, proto::ModelDef &model_def) const;

  bool UnserializeModel(const uint8_t *const data, const size_t len,
                        Model &model, const bool is_enable_multi_thread = false) const;
  bool UnserializeModel(ge::proto::ModelDef &model_def, Model &model, const std::string &path) const;
  bool UnserializeModel(ge::proto::ModelDef &model_def, Model &model) const;
 private:
  friend class ModelSerializeImp;
  friend class GraphDebugImp;
};
}  // namespace ge
#endif  // INC_GRAPH_MODEL_SERIALIZE_H_

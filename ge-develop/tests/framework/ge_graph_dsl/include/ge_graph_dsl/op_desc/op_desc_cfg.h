/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H77F0BD09_6C00_4E45_8DED_38A676D6B20A
#define H77F0BD09_6C00_4E45_8DED_38A676D6B20A

#include <string>
#include "ge_graph_dsl/ge.h"
#include "graph/types.h"
#include "ge_graph_dsl/op_desc/op_type.h"

GE_NS_BEGIN

struct OpDescCfg {
  struct TensorCfg {
    TensorCfg(Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT, std::vector<int64_t> shape = {1, 1, 224, 224})
        : format_(format), data_type_(data_type), shape_(shape) {}
    Format format_;
    DataType data_type_;
    std::vector<int64_t> shape_;
  };

  OpDescCfg(const OpType &type, int in_cnt = 0, int out_cnt = 0, Format format = FORMAT_NCHW,
            DataType data_type = DT_FLOAT, std::vector<int64_t> shape = {1, 1, 224, 224})
      : type_(type), in_cnt_(in_cnt), out_cnt_(out_cnt), default_tensor_(format, data_type, shape) {}

 protected:
  OpType GetType() const { return type_; }
  OpType type_;
  int in_cnt_;
  int out_cnt_;
  int stream_id_ = -1;
  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;
  TensorCfg default_tensor_;
};

GE_NS_END

#endif /* H77F0BD09_6C00_4E45_8DED_38A676D6B20A */

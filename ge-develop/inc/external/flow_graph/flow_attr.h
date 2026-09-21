/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_
#define INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_

#include <cstdint>

namespace ge {
namespace dflow {
struct CountBatch {
  int64_t batch_size = 0L;
  int64_t slide_stride = 0L;
  int64_t timeout = 0L;
  int64_t batch_dim = 0L;
  int32_t flag = 0; // eg: eos/seg
  bool padding = false;
  bool drop_remainder = false;
  int8_t rsv[90] = {};
};

struct TimeBatch {
  int64_t time_window = 0L;
  int64_t time_interval = 0L;
  int64_t timeout = 0L;
  int64_t batch_dim = -1;
  int32_t flag = 0; // eg: eos/seg
  bool padding = false;
  bool drop_remainder = false;
  int8_t rsv[90] = {};
};

enum class DataFlowAttrType {
  COUNT_BATCH = 0,
  TIME_BATCH = 1,
  INVALID = 2,
};

struct DataFlowInputAttr {
  DataFlowAttrType attr_type;
  void *attr_value;
};
} // namespace dflow
} // namespace ge
#endif // INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_

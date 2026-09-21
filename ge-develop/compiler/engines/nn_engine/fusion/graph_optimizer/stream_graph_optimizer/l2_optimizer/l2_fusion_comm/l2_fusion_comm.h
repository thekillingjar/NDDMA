/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_COMM_L2_FUSION_COMM_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_COMM_L2_FUSION_COMM_H_

#include <map>
#include <string>
#include <vector>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/node.h"
#include "runtime/kernel.h"

namespace fe {
const uint32_t L2_CTRL_REMAP_SIZE = 64;
const uint32_t L2_CTRL_DATA_SIZE = 8;
const uint32_t INPUT_DATA = 0;
const uint32_t OUTPUT_DATA = 1;

// data_type: input-0, output-1, filter-2
#define DATA_OVERALL_ID(node_id, datatype, data_id) (((node_id) << 16) | ((datatype) << 8) | (data_id))

struct L2BufferInfo {
  uint64_t l2_buffer_size;
  uint64_t page_num;
  uint64_t max_data_num;  // runtime allows at most 8 data segments in l2 buffer
};

// data info for l2 buffer
struct TensorL2DataInfo {
  uint64_t id;
  uint64_t ddr_addr;
  int64_t data_size;
  uint64_t refer_data_id = 0;
  std::vector<uint32_t> occupy_data_ids;
};

using TensorL2DataPair = std::pair<uint64_t, TensorL2DataInfo>;
using TensorL2DataMap = std::map<uint64_t, TensorL2DataInfo>;

struct OpL2DataInfo {
  uint32_t node_id;
  string node_name;
  uint32_t node_task_num;

  TensorL2DataMap input;
  TensorL2DataMap output;
  TensorL2DataMap filter;
};

// data allocation info in l2 buffer
struct TensorL2AllocInfo {
  TensorL2DataInfo data;    // init data info
  uint64_t l2PageNum;  // available l2 buffer size
  int32_t data_in_l2_id;
  uint64_t data_in_l2_addr;  // l2 addr

  int8_t pre_L2_page_offset_base;  // input
  uint8_t L2_page_offset_base;
  uint8_t modified;       // 1 - data will be modified by kernel, 0 - no modified
  uint8_t L2_preload;     // 1 - preload from mirror_dd_r, 0-no preload
  uint8_t L2_load_to_ddr; /**< 1 - need load out, 0 - no need */
  uint8_t priority;       // data priority

  int32_t occupy_data_id;
};

using TensorL2AllocPair = std::pair<uint64_t, TensorL2AllocInfo>;
using TensorL2AllocMap = std::map<uint64_t, TensorL2AllocInfo>;

struct OpL2AllocInfo {
  ~OpL2AllocInfo() { clear(); }
  void clear() {
    standing_data.clear();
    input.clear();
    output.clear();
    filter_preload.clear();
    input_preload.clear();
  }
  uint32_t node_id;
  string node_name;

  rtL2Ctrl_t l2ctrl;
  TensorL2AllocMap standing_data;
  TensorL2AllocMap input;
  TensorL2AllocMap output;
  TensorL2AllocMap converge;
  TensorL2AllocMap filter_preload;
  TensorL2AllocMap input_preload;
};

using OpL2AllocMap = std::map<string, OpL2AllocInfo>;  // the key is node_name;

// node_id, <output_index, input_index>
using OpReferTensorIndexMap = std::map<uint64_t, uint32_t>;

class L2FusionComm {
 public:
  static void GetL2HardwareSet(L2BufferInfo &l2);
  static Status GetGraphDataSize(const ge::NodePtr &node, const size_t &tensor_index, const uint32_t &data_type,
                                 int64_t &data_size);
  static void DisplayOpL2DataInfo(const std::vector<OpL2DataInfo> &l2_data_vec);
  static void DisplayOpL2AllocInfo(const OpL2AllocMap &l2_alloc_map);
  static bool IsNonValidOp(const ge::NodePtr &node);
  static bool IsNonValidOpOrData(const ge::NodePtr &node);
  static bool IsConstInput(const ge::ConstNodePtr &node, const size_t &index);
  static bool IsConstInput(const ge::Node &node, const size_t &index);

 private:
  static Status GetGraphDataSize(const ge::OpDescPtr &op_desc, const ge::GeTensorDesc &tensor_desc,
                                 const uint32_t &data_type, int64_t &data_size);
  static Status CalcTensorSize(const ge::GeTensorDesc &tensor_desc, int64_t &tensor_size);
  static bool CheckData(const ge::OpDescPtr &op_desc);
  static void DisplayTensorL2AllocInfo(const TensorL2AllocMap &input, rtL2Ctrl_t l2ctrl,
                                       const TensorL2AllocMap &tensor_alloc_map);
  static void DisplayTensorL2DataInfo(const TensorL2DataMap &tensor_l2_map);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_COMM_L2_FUSION_COMM_H_

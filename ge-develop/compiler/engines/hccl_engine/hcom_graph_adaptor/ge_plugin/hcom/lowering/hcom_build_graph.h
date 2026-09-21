/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_BUILD_GRAPH_H
#define HCOM_BUILD_GRAPH_H
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "graph/node.h"
#include "register/node_converter_registry.h"
#include "op_hcom_comm.h"

namespace hccl {

constexpr u32 LAUNCH_RECV_KERNEL_OUTPUT_SIZE = 2; /* recv kerner的输出 */

enum class HcomOpType {
  HCOM_ALL_GATHER,
  HCOM_ALL_GATHER_V,
  HCOM_ALL_REDUCE,
  HCOM_BROADCAST,
  HCOM_REDUCE_SCATTER,
  HCOM_REDUCE_SCATTER_V,
  HCOM_ALL_TO_ALL_V,
  HCOM_ALL_TO_ALL_VC,
  HCOM_ALL_TO_ALL,
  HCOM_SEND,
  HCOM_REDUCE,
  HCOM_RECEIVE,
};

struct HcomAllReduceOpAttr {
  HcclReduceOp reduction;
};

struct HcomAllGatherOpAttr {
  int32_t rankSize;
};

struct HcomAllGatherVOpAttr {
  HcclDataType recvType;
};

struct HcomBroadcastOpAttr {
  int32_t root;
};

struct HcomReduceScatterOpAttr {
  HcclReduceOp reduction;
  int32_t rankSize;
};

struct HcomReduceScatterVOpAttr {
  HcclReduceOp reduction;
  int32_t rankSize;
};

struct HcomAllToAllVOpAttr {
  HcclDataType recvType;
};

struct HcomRecvOpAttr {
  int32_t srcRank;
  int32_t srTag;
};

struct HcomSendOpAttr {
  int32_t destRank;
  int32_t srTag;
};

struct HcomReduceOpAttr {
  int32_t root;
  HcclReduceOp reduction;
};

struct HcomOpAttr {
  HcomOpType opType = HcomOpType::HCOM_RECEIVE;
  HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
  u32 aivCoreLimit = 0;
  char group[GROUP_NAME_MAX_LEN + 1];
  union {
    HcomAllGatherOpAttr allgather;
    HcomAllGatherVOpAttr allgatherv;
    HcomAllReduceOpAttr allreduce;
    HcomBroadcastOpAttr broadcast;
    HcomReduceScatterOpAttr reducescatter;
    HcomReduceScatterVOpAttr reducescatterv;
    HcomAllToAllVOpAttr alltoallv;
    HcomSendOpAttr send;
    HcomRecvOpAttr recv;
    HcomReduceOpAttr reduce;
  } op;

  HcomOpAttr() : opType(HcomOpType::HCOM_RECEIVE), dataType(HcclDataType::HCCL_DATA_TYPE_RESERVED) {
    memset_s(group, sizeof(group), 0, sizeof(group));
    new (&op.allgather) HcomAllGatherOpAttr();
    new (&op.allgatherv) HcomAllGatherVOpAttr();
    new (&op.allreduce) HcomAllReduceOpAttr();
    new (&op.broadcast) HcomBroadcastOpAttr();
    new (&op.reducescatter) HcomReduceScatterOpAttr();
    new (&op.reducescatterv) HcomReduceScatterVOpAttr();
    new (&op.alltoallv) HcomAllToAllVOpAttr();
    new (&op.send) HcomSendOpAttr();
    new (&op.recv) HcomRecvOpAttr();
    new (&op.reduce) HcomReduceOpAttr();
  }
};

struct HcomLaunchArg {
  gert::bg::ValueHolderPtr opArgs;
  gert::bg::ValueHolderPtr stream;
};

gert::bg::ValueHolderPtr GenerateHcomOpArgs(const ge::NodePtr &node);

std::vector<gert::bg::ValueHolderPtr> LaunchHcomOpKernel(const HcomLaunchArg &opAttr,
                                                         const std::vector<gert::bg::DevMemValueHolderPtr> &inputAddrs,
                                                         const std::vector<gert::bg::DevMemValueHolderPtr> &outputAddrs,
                                                         const std::vector<gert::bg::ValueHolderPtr> &inputShapes,
                                                         const std::vector<gert::bg::ValueHolderPtr> &outputShapes);

std::vector<gert::bg::DevMemValueHolderPtr> LaunchRecvOpKernel(const HcomLaunchArg &args, const ge::NodePtr &node,
                                                               const gert::LowerInput &lowerInput);
HcclResult HcomAllToAllGetOpAttr(const ge::NodePtr &node, struct HcomOpAttr &opAttr);
}  // namespace hccl
#endif  // HCOM_BUILD_GRAPH_H
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/host_resource_center/host_resource_serializer.h"
#include "framework/common/host_resource_center/tiling_resource_manager.h"
#include "common/util/mem_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "register/op_tiling_info.h"
#include "common/checker.h"

namespace ge {
constexpr size_t kTLLen = 1UL + sizeof(uint32_t);
constexpr uint32_t kTilingDataMagic = 0xDD776EFEU;
constexpr uint32_t kTilingSerializeVersionOld = 0x1U; // deprecated
constexpr uint32_t kTilingSerializeVersion = 0x2U;

enum class TilingSerializeType : uint8_t { kOpRunInfoStruct = 0U, kTilingData = 1U, kTilingIds = 2U };

struct TilingHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t data_len;
};

struct OpRunInfoMsg {
  uint64_t tiling_key;
  uint64_t tiling_size;
  uint32_t block_dim;
  bool clear_atomic;
  int32_t tiling_cond;
  uint32_t schedule_mode;
  uint32_t workspace_num;
  uint32_t local_memory_size;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  int64_t workspace[0];
};

static size_t CalcTilingSerializeSize(
    const std::unordered_map<const optiling::utils::OpRunInfo *, std::vector<TilingId>> &resource_to_keys) {
  size_t total_size{sizeof(TilingHeader)};
  for (const auto &iter : resource_to_keys) {
    if (iter.first == nullptr) {
      continue;
    }
    // calc tiling struct
    total_size += kTLLen + sizeof(OpRunInfoMsg) + iter.first->GetWorkspaceNum() * sizeof(int64_t);
    // calc tiling data
    total_size += kTLLen + static_cast<uint64_t>(iter.first->GetAllTilingData().str().size());
    // calc tiling ids
    total_size += kTLLen + iter.second.size() * sizeof(int64_t);
  }

  return total_size;
}

graphStatus HostResourceSerializer::RecoverOpRunInfoToExtAttrs(const TilingResourceManager *tiling_resource_manager,
                                                               OpDescPtr op_desc) {
  auto aicore_resource =
      tiling_resource_manager->GetResource(op_desc, static_cast<int64_t>(TilingResourceType::kAiCore));
  auto aicore_tiling = dynamic_cast<const TilingResource *>(aicore_resource);
  if (aicore_tiling != nullptr) {
    GELOGI("recover %s aicore info", op_desc->GetNamePtr());
    GE_ASSERT_TRUE(op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, aicore_tiling->GetOpRunInfo()));
  }

  auto atomic_resource =
      tiling_resource_manager->GetResource(op_desc, static_cast<int64_t>(TilingResourceType::kAtomic));
  auto atomic_tiling = dynamic_cast<const TilingResource *>(atomic_resource);
  if (atomic_tiling != nullptr) {
    GELOGI("recover %s atomic_tiling info", op_desc->GetNamePtr());
    GE_ASSERT_TRUE(op_desc->SetExtAttr(ge::ATTR_NAME_ATOMIC_OP_RUN_INFO, atomic_tiling->GetOpRunInfo()));
  }
  return GRAPH_SUCCESS;
}

graphStatus HostResourceSerializer::RecoverOpRunInfoToExtAttrs(const HostResourceCenter &host_resource_center,
                                                               const ComputeGraphPtr &graph) {
  const HostResourceManager *manager = host_resource_center.GetHostResourceMgr(HostResourceType::kTilingData);
  const TilingResourceManager *tiling_resource_manager = dynamic_cast<const TilingResourceManager *>(manager);
  GE_ASSERT_NOTNULL(tiling_resource_manager);
  GE_ASSERT_NOTNULL(graph, "Root graph cannot be nullptr");
  for (const Node *node : graph->GetAllNodesPtr()) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetType() == "SuperKernel") {
      GELOGI("find sk node %s", op_desc->GetNamePtr());
      ComputeGraphPtr sub_graph = nullptr;
      GE_ASSERT_TRUE(AttrUtils::GetGraph(op_desc, "_sk_sub_graph", sub_graph));
      if (sub_graph != nullptr) {
        GE_ASSERT_TRUE(op_desc->SetExtAttr("_sk_sub_graph", sub_graph));
        GELOGI("recover op_run_info in sk %s sub_graph %s", op_desc->GetNamePtr(), sub_graph->GetName().c_str());
        for (auto &sub_node : sub_graph->GetDirectNode()) {
          GE_ASSERT_NOTNULL(sub_node);
          auto sub_op_desc = sub_node->GetOpDesc();
          GE_ASSERT_NOTNULL(sub_op_desc);
          GE_ASSERT_SUCCESS(RecoverOpRunInfoToExtAttrs(tiling_resource_manager, sub_op_desc));
        }
      }
    } else {
      GE_ASSERT_SUCCESS(RecoverOpRunInfoToExtAttrs(tiling_resource_manager, op_desc));
    }
  }
  GELOGI("Recover oprun_info to ext attrs successfully.");
  return GRAPH_SUCCESS;
}

graphStatus HostResourceSerializer::SerializeTilingData(const HostResourceCenter &host_resource_center, uint8_t *&data,
                                                        size_t &len) {
  HostResourcePartition partition;
  partition.total_size_ = 0UL;
  partitions_.push_back(partition);
  GE_ASSERT_SUCCESS(SerializeTilingData(host_resource_center, partitions_.back()));
  data = partitions_.back().buffer_.data();
  len = partitions_.back().total_size_;
  return GRAPH_SUCCESS;
}

static graphStatus SerializeTilingStruct(const optiling::utils::OpRunInfo *const op_run_info,
                                         const uint32_t tiling_data_len, uint8_t *&head, size_t &remaining_size) {
  // serialize struct
  const uint32_t len_run_info_msg =
      static_cast<uint32_t>(sizeof(OpRunInfoMsg) + op_run_info->GetWorkspaceNum() * sizeof(int64_t));
  GE_ASSERT_TRUE(remaining_size >= kTLLen + len_run_info_msg);
  *head = static_cast<uint8_t>(TilingSerializeType::kOpRunInfoStruct);
  head++;
  remaining_size--;
  GE_ASSERT_EOK(memcpy_s(head, remaining_size, &len_run_info_msg, sizeof(uint32_t)));

  head += sizeof(uint32_t);
  remaining_size -= sizeof(uint32_t);
  OpRunInfoMsg *msg = PtrToPtr<uint8_t, OpRunInfoMsg>(head);
  GE_ASSERT_NOTNULL(msg);
  msg->block_dim = op_run_info->GetBlockDim();
  msg->tiling_key = op_run_info->GetTilingKey();
  msg->tiling_cond = op_run_info->GetTilingCond();
  msg->tiling_size = tiling_data_len;
  msg->clear_atomic = op_run_info->GetClearAtomic();
  msg->workspace_num = static_cast<uint32_t>(op_run_info->GetWorkspaceNum());
  msg->schedule_mode = op_run_info->GetScheduleMode();
  msg->local_memory_size = op_run_info->GetLocalMemorySize();
  msg->reserved0 = 0U;
  msg->reserved1 = 0U;
  msg->reserved2 = 0U;

  for (size_t i = 0UL; i < msg->workspace_num; ++i) {
    GE_ASSERT_GRAPH_SUCCESS(op_run_info->GetWorkspace(i, msg->workspace[i]));
  }
  head += len_run_info_msg;
  remaining_size -= len_run_info_msg;
  return SUCCESS;
}

static graphStatus SerializeTilingDataToIds(const std::stringstream &tiling_stream, const uint32_t tiling_data_len,
                                            const std::vector<TilingId> &tiling_ids, uint8_t *&head,
                                            size_t &remaining_size) {
  // serialize tiling data
  GE_ASSERT_TRUE(remaining_size >= kTLLen + tiling_data_len);
  *head = static_cast<uint8_t>(TilingSerializeType::kTilingData);
  head++;
  remaining_size--;
  GE_ASSERT_EOK(memcpy_s(head, remaining_size, &tiling_data_len, sizeof(uint32_t)));
  head += sizeof(uint32_t);
  remaining_size -= sizeof(uint32_t);

  if (tiling_data_len > 0U) {
    (void)tiling_stream.rdbuf()->pubseekoff(0U, std::ios_base::beg);  // rewind
    const auto rd_len =
        tiling_stream.rdbuf()->sgetn(PtrToPtr<uint8_t, char>(head), static_cast<int64_t>(tiling_data_len));
    GE_ASSERT_TRUE(static_cast<size_t>(rd_len) == tiling_data_len,
                   "Copy tiling data failed, tiling data size: %zu, copied len: %ld.", tiling_data_len, rd_len);
    head += tiling_data_len;
    remaining_size -= tiling_data_len;
  }

  // serialize tiling ids
  const uint32_t len_ids = static_cast<uint32_t>(tiling_ids.size() * sizeof(int64_t));
  GE_ASSERT_TRUE(remaining_size >= kTLLen + len_ids);
  *head = static_cast<uint8_t>(TilingSerializeType::kTilingIds);
  head++;
  remaining_size--;
  GE_ASSERT_EOK(memcpy_s(head, remaining_size, &len_ids, sizeof(uint32_t)));
  head += sizeof(uint32_t);
  remaining_size -= sizeof(uint32_t);
  GE_ASSERT_EOK(memcpy_s(head, remaining_size, tiling_ids.data(), len_ids));
  head += len_ids;
  remaining_size -= len_ids;
  return SUCCESS;
}

graphStatus HostResourceSerializer::SerializeTilingData(const HostResourceCenter &host_resource_center,
                                                        HostResourcePartition &partition) {
  const HostResourceManager *manager = host_resource_center.GetHostResourceMgr(HostResourceType::kTilingData);
  const TilingResourceManager *tiling_resource_manager = dynamic_cast<const TilingResourceManager *>(manager);
  GE_ASSERT_NOTNULL(tiling_resource_manager);
  const std::unordered_map<const optiling::utils::OpRunInfo *, std::vector<TilingId>> &resource_to_keys =
      tiling_resource_manager->GetResourceToKeys();
  if (resource_to_keys.empty()) {
    GELOGI("Tiling data resource is empty.");
    return GRAPH_SUCCESS;
  }
  const size_t total_size = CalcTilingSerializeSize(resource_to_keys);
  GE_ASSERT_TRUE(total_size <= UINT32_MAX, "The total size of tiling data [%zu] exceeds uint32_max.", total_size);
  partition.total_size_ = total_size;
  partition.buffer_.resize(total_size);
  // header
  uint8_t *head = partition.buffer_.data();
  GE_ASSERT_NOTNULL(head);
  size_t remaining_size = total_size;
  constexpr size_t head_len = sizeof(TilingHeader);
  GE_ASSERT_TRUE(remaining_size > head_len);
  TilingHeader *tiling_head = PtrToPtr<uint8_t, TilingHeader>(head);
  tiling_head->version = kTilingSerializeVersion;
  tiling_head->magic = kTilingDataMagic;
  head += head_len;
  remaining_size -= head_len;
  tiling_head->data_len = static_cast<uint32_t>(remaining_size);
  for (const auto &iter : resource_to_keys) {
    const optiling::utils::OpRunInfo *const op_run_info = PtrToPtr<void, optiling::utils::OpRunInfo>(iter.first);
    GE_ASSERT_NOTNULL(op_run_info);

    const std::stringstream &tiling_stream = op_run_info->GetAllTilingData();
    const uint32_t tiling_data_len = static_cast<uint32_t>(tiling_stream.str().size());
    GE_ASSERT_SUCCESS(SerializeTilingStruct(op_run_info, tiling_data_len, head, remaining_size));
    GE_ASSERT_SUCCESS(SerializeTilingDataToIds(tiling_stream, tiling_data_len, iter.second, head, remaining_size));
  }

  tiling_head->data_len -= static_cast<uint32_t>(remaining_size);
  GELOGI("SerializeTilingData successfully, total size:[%zu], data_len:[%u].", total_size, tiling_head->data_len);
  return GRAPH_SUCCESS;
}

static graphStatus InitNewOpRunInfo(std::shared_ptr<optiling::utils::OpRunInfo> &op_run_info,
                                    const OpRunInfoMsg *const run_info_msg, const uint32_t part_len) {
  op_run_info = ge::MakeShared<optiling::utils::OpRunInfo>();  // create new
  GE_ASSERT_NOTNULL(op_run_info);
  op_run_info->SetTilingKey(run_info_msg->tiling_key);
  op_run_info->SetTilingCond(run_info_msg->tiling_cond);
  op_run_info->SetBlockDim(run_info_msg->block_dim);
  op_run_info->SetClearAtomic(run_info_msg->clear_atomic);
  op_run_info->SetScheduleMode(run_info_msg->schedule_mode);
  op_run_info->SetLocalMemorySize(run_info_msg->local_memory_size);
  const size_t ws_num = run_info_msg->workspace_num;
  const size_t check_size = sizeof(OpRunInfoMsg) + ws_num * sizeof(int64_t);
  GE_ASSERT_TRUE(check_size <= part_len, "Tiling struct length mismatch, need size:[%zu], value_len:[%u]", check_size,
                 part_len);
  std::vector<int64_t> workspaces(ws_num, 0);
  for (uint32_t i = 0U; i < ws_num; ++i) {
    workspaces[i] = run_info_msg->workspace[i];
  }
  op_run_info->SetWorkspaces(workspaces);
  return SUCCESS;
}

graphStatus HostResourceSerializer::DeSerializeTilingData(const HostResourceCenter &host_resource_center,
                                                          const uint8_t *const data, size_t length) {
  const HostResourceManager *manager = host_resource_center.GetHostResourceMgr(HostResourceType::kTilingData);
  TilingResourceManager *tiling_resource_manager =
      const_cast<TilingResourceManager *>(dynamic_cast<const TilingResourceManager *>(manager));
  GE_ASSERT_NOTNULL(tiling_resource_manager);

  // checking tiling header.
  const uint8_t *head = data;
  GE_ASSERT_NOTNULL(head);
  GE_ASSERT(length > sizeof(TilingHeader));
  auto tiling_header = PtrToPtr<uint8_t, TilingHeader>(head);
  GE_ASSERT_TRUE(length >= sizeof(TilingHeader) + tiling_header->data_len,
                 "Tiling header check failed, data_size:[%u] is invalid.", tiling_header->data_len);
  GE_ASSERT_TRUE(tiling_header->magic == kTilingDataMagic,
                 "The magic check of tiling header has failed,indicating the om bin file may have been modified.");

  if (tiling_header->version == kTilingSerializeVersionOld) {
    tiling_resource_manager->UseOpIdAsTilingId();
  }

  head += sizeof(TilingHeader);
  size_t pos{0UL};
  shared_ptr<optiling::utils::OpRunInfo> op_run_info;
  while (pos < tiling_header->data_len) {
    GE_ASSERT_TRUE(pos + kTLLen <= tiling_header->data_len, "Tiling data bin is incomplete.");
    const uint8_t type_val = head[pos++];
    GE_ASSERT_TRUE(type_val <= static_cast<uint8_t>(TilingSerializeType::kTilingIds), "Type val [%u] is invalid.",
                   type_val);
    const auto serialize_type = static_cast<TilingSerializeType>(type_val);
    const uint32_t len = *PtrToPtr<uint8_t, uint32_t>(&head[pos]);
    pos += sizeof(uint32_t);
    GE_ASSERT_TRUE(pos + len <= tiling_header->data_len);
    switch (serialize_type) {
      case TilingSerializeType::kOpRunInfoStruct: {
        const OpRunInfoMsg *run_info_msg = PtrToPtr<uint8_t, OpRunInfoMsg>(&head[pos]);
        GE_ASSERT_SUCCESS(InitNewOpRunInfo(op_run_info, run_info_msg, len));
        break;
      }
      case TilingSerializeType::kTilingData: {
        if (len > 0U) {
          GE_ASSERT_NOTNULL(op_run_info);
          const char_t *tiling_data = PtrToPtr<uint8_t, char_t>(&head[pos]);
          op_run_info->AddTilingData(tiling_data, len);
        }
        break;
      }
      case TilingSerializeType::kTilingIds: {
        GE_ASSERT_NOTNULL(op_run_info);
        const uint32_t cnt = static_cast<uint32_t>(len / sizeof(TilingId));
        const TilingId *tiling_id = PtrToPtr<uint8_t, TilingId>(&head[pos]);
        uint32_t i = 0U;
        while (i++ < cnt) {
          std::shared_ptr<HostResource> host_resource = MakeShared<TilingResource>(op_run_info);
          GE_ASSERT_NOTNULL(host_resource);
          GE_ASSERT_SUCCESS(tiling_resource_manager->RecoverResource(*tiling_id, host_resource),
                            "Failed to recover tiling resource.");
          tiling_id++;
        }
        op_run_info = nullptr;
        break;
      }
      default:
        GELOGE(FAILED, "Serialized tiling data_type [%d] is invalid.", static_cast<int32_t>(serialize_type));
        return FAILED;
    }
    pos += len;
  }

  GELOGI("DeSerializeTilingData successfully.");
  return GRAPH_SUCCESS;
}
}  // namespace ge

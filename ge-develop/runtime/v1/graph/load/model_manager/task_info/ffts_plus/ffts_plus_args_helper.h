/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_ARGS_HELPER_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_ARGS_HELPER_H_
#include <set>
#include <map>
#include <vector>

#include "graph/args_format_desc.h"
#include "ge/ge_api_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "runtime/rt.h"
#include "runtime/rt_ffts_plus.h"
#include "graph/compute_graph.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"

namespace ge {
struct ArgsFormatHolder {
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  std::vector<ArgDesc> arg_descs;
  std::vector<std::vector<int64_t>> shape_infos;
};

class FftsPlusProtoTransfer;
enum class Level1AddrType : int16_t { CMO_ADDR, SDMA_SRC, SDMA_DST, DSA_INPUT, DSA_OUTPUT, DSA_WORKSPACE };
struct CtxAddrInfo {
  Level1AddrType level_1_addr_type;
  domi::FftsPlusCtxDef ctx_def;
  rtFftsPlusComCtx_t *rts_ctx;
  OpDescPtr op;
};
using AddrType2CtxAddrInfo = std::vector<CtxAddrInfo>;
struct CtxInfo {
  int32_t ctx_id;
  OpDescPtr ctx_op;
};
class FftsPlusArgsHelper {
 public:
  explicit FftsPlusArgsHelper(const RuntimeParam &runtime_aram) : runtime_param_(runtime_aram){};

  void AppendAbsoluteAddrs(const uint64_t rt_addr, const std::string &addr_type);
  void AppendIoAddrs(const uint64_t logic_addr);
  void AppendRtIoAddrs(const uint64_t rt_addr, const uint64_t mem_type);
  void AppendCtxLevel1Addrs(const uint64_t logic_addr, const uint64_t iow_mem_type, const CtxAddrInfo& ctx_addr_info);

  Status AppendAicpuAddrs(const uint64_t rt_addr, const uint64_t io_mem_type, const size_t relevant_offset);
  Status AppendBinArgs(const uint8_t *const args_addr, const size_t args_size);

  Status UpdateIoAddrByIndex(const size_t index, const uint64_t rt_addr);

  Status InitArgsBase(uint8_t *const pis_args_host_base, uint8_t *const args_dev,
                      uint8_t *const args_host, const size_t args_size, const size_t bin_args_size);

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos,
                                 FftsPlusProtoTransfer &ffts_plus_proto_transfer);
  Status UpdateAddrsWithIOZcpy(const std::vector<uint64_t> &active_mem_base_addr,
                               FftsPlusProtoTransfer &ffts_plus_proto_transfer);

  Status InitRuntimeAddr(DavinciModel *davinci_model);
  Status PlanUpdaterArgslayOut(DavinciModel *davinci_model);
  void UpdateCtxInfo(const CtxInfo &ctx_info) {
    ctx_info_ = ctx_info;
  }
  void GetCtxInfo(CtxInfo &ctx_info) const {
    ctx_info = ctx_info_;
  }
  Status AssembleTilingData() const;
  void SetOpDesc(const OpDescPtr &op_desc) {
    op_desc_ = op_desc;
  }
  const std::vector<uint64_t> &GetIoAddr() const {
    return io_addrs_;
  }

  const uint8_t *GetTilingDataDev() const {
    return tiling_data_dev_;
  }

  size_t GetTilingDataLen() const {
    return tiling_data_len_;
  }

  void SetTilingDataDev(void *const tiling_data_dev) {
    tiling_data_dev_ = reinterpret_cast<uint8_t *>(tiling_data_dev);
  }

  void SetTilingDataHost(uint8_t *const tiling_data_host) {
    tiling_data_host_ = tiling_data_host;
  }

  void SetTilingDataLen(const size_t tiling_data_len) {
    tiling_data_len_ = tiling_data_len;
  }

  const std::set<size_t> &GetModeAddrIdx() const {
    return mode_addr_idx_;
  }

  size_t GetIoAddrSize() const {
    return io_addrs_.size();
  }

  void MarkModelIOAddrIndex() {
    (void)mode_addr_idx_.emplace(io_addrs_.size());
  }

  void GetBinArsDevAddr(void *&addr) const {
    addr = args_dev_ + GetBinArsDevOffset();
  }

  size_t GetBinArsDevOffset() const {
    return args_size_ - bin_args_size_ + used_bin_args_size_;
  }

  void SaveLevel2Offset(const uint32_t ctx_id, const int64_t sub_offset) {
    data_ctx_to_level2_offset_[ctx_id] = sub_offset;
  }

  bool CheckAndGetLevel2Offset(const uint32_t ctx_id, int64_t &sub_offset);

  void SaveArgsFormats(int64_t op_id, const ArgsFormatHolder &holder) {
    op_to_format_holder_[op_id] = holder;
  }

  void SaveMemCheckInfo(const std::string &op_name, const std::string &memcheck_info) {
    memcheck_infos_[op_name] = memcheck_info;
  }

  const std::string GetMemCheckInfo(const std::string &op_name) const;

  bool CheckAndGetArgsFormats(int64_t op_id, ArgsFormatHolder &holder);

  void SaveOffsetPairs(const uint64_t cust_offset, const uint64_t relevant_offset);

  const std::map<uint64_t, uint64_t> &GetCustToRelevantOffset() const {
    return cust_to_relevant_offset_;
  }

  size_t GetCtxArgsSize (const int32_t ctx_id) const;
 private:
  Status GenAicpuRefreshInfos(const std::vector<TaskArgsRefreshInfo> &args_fresh_info,
                             const size_t start_idx,
                             std::vector<TaskArgsRefreshInfo> &aicpu_args_fresh_info);
  Status UpdateAicpuAddrs(const std::vector<uint64_t> &io_addrs, const size_t start_idx);

  const RuntimeParam &runtime_param_;
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_mem_types_;
  size_t aicaiv_addr_size_{0UL};
  size_t level1_addr_size_{0UL};
  std::set<size_t> mode_addr_idx_;  // mode addr at pos of io_addrs_

  // separate update for context, will be merge with level1 addrs
  std::vector<uint64_t> level1_logic_heads_;
  std::vector<uint64_t> level1_mem_types_;
  AddrType2CtxAddrInfo level1_ctx_addr_infos_;
  uint8_t *args_dev_{nullptr};
  uint8_t *args_host_{nullptr};
  uint8_t *pis_args_host_base_{args_host_};
  size_t args_size_{0UL};

  uint8_t *tiling_data_dev_{nullptr};
  uint8_t *tiling_data_host_{nullptr};
  size_t tiling_data_len_{0UL};

  // for ascend aicpu
  std::vector<uint64_t> aicpu_logic_heads_;
  std::vector<uint64_t> aicpu_logic_heads_mem_types_;
  std::vector<size_t> args_relevent_offsets_;  // 自研aicpu算子args不连续，表示每个地址相对位置用于更新
  uint8_t *bin_args_host_{nullptr};
  size_t bin_args_size_{0UL};
  size_t used_bin_args_size_{0UL};
  // save data_ctx offset
  std::map<uint32_t, int64_t> data_ctx_to_level2_offset_;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  OpDescPtr op_desc_;
  // args format res
  std::map<int64_t, ArgsFormatHolder> op_to_format_holder_;
  CtxInfo ctx_info_{-1, nullptr};
  // temp solution for dump with args_format
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
  // record for exception dump
  std::map<int32_t, size_t> ctx_args_size_;
  std::map<std::string, std::string> memcheck_infos_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_ARGS_HELPER_H_

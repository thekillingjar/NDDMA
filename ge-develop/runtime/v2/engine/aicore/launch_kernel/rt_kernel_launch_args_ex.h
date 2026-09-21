/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_RT_KERNEL_LAUNCH_ARGS_EX_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_RT_KERNEL_LAUNCH_ARGS_EX_H_
#include <memory>
#include <cstdint>
#include "graph/utils/node_utils.h"
#include "graph/utils/math_util.h"
#include "runtime/kernel.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tiling_data.h"
#include "common/checker.h"
#include "engine/aicore/fe_rt2_common.h"

namespace gert {
constexpr size_t kMaxHostInputDataLen = 128U;  // 64 aligned
constexpr size_t kHostInputDataAlginSize = 32U;  // 32 alignd
struct DynDesc {
  bool is_group_first{false};
  uint32_t io_group_id{0};
  uint32_t io_index{0};
  uint32_t dyn_num{0};
  uint32_t group_size{0};
  uint32_t group_offset{0};
  uint32_t ptr_offset{0};
};
struct DynDescInfo {
  std::vector<DynDesc> dyn_in_desc;
  std::vector<DynDesc> dyn_out_desc;
};
class ArgsInfosDesc {
 public:
  struct ArgsInfo {
    enum class ArgsInfoType { INPUT = 0, OUTPUT = 1, TYPE_END };
    enum class ArgsInfoFormat { DIRECT_ADDR = 0, FOLDED_DESC_ADDR = 1, FORMAT_END };
    void Init(ArgsInfoType cur_arg_type, ArgsInfoFormat cur_arg_fmt, int32_t start_idx, uint32_t cur_arg_size) {
      arg_type = cur_arg_type;
      arg_format = cur_arg_fmt;
      start_index = start_idx;
      arg_size = cur_arg_size;
    }
    ArgsInfoType arg_type;
    ArgsInfoFormat arg_format;
    int32_t start_index;
    uint32_t arg_size;
    bool folded_first{false};
    bool be_folded{false};
    DynDesc dyn_desc;
    size_t data_size{0U};
  };

  static std::unique_ptr<uint8_t[]> Create(size_t args_infos_size, size_t &total_size);
  void Init(size_t input_args_info_num, size_t output_args_info_num, size_t input_valid_num, size_t output_valid_num) {
    input_args_info_num_ = input_args_info_num;
    output_args_info_num_ = output_args_info_num;
    input_valid_num_ = input_valid_num;
    output_valid_num_ = output_valid_num;
    args_infos_size_ = (input_args_info_num + output_args_info_num) * sizeof(ArgsInfo);
  }
  size_t GetInputArgsInfoNum() const {
    return input_args_info_num_;
  }
  size_t GetOutputArgsInfoNum() const {
    return output_args_info_num_;
  }
  size_t GetValidInputArgsNum() const {
    return input_valid_num_;
  }
  size_t GetValidOutputArgsNum() const {
    return output_valid_num_;
  }
  size_t GetArgsInfoNum() const {
    return input_args_info_num_ + output_args_info_num_;
  }
  size_t GetArgsInfoSize() const {
    return args_infos_size_;
  }
  void *MutableArgsInfoBase() {
    return args_infos_;
  }
  const void *GetArgsInfoBase() const {
    return args_infos_;
  }
  const ArgsInfo *GetArgsInfoByIndex(size_t idx) const {
    if (idx >= GetArgsInfoNum()) {
      GELOGE(ge::PARAM_INVALID, "[PARAM][INVALID] Index[%zu], expect less than args info num[%zu]", idx,
             (input_args_info_num_ + output_args_info_num_));
      return nullptr;
    }
    auto args_info_base = reinterpret_cast<const ArgsInfo *>(args_infos_);
    return (args_info_base + idx);
  }

 private:
  size_t input_args_info_num_;
  size_t output_args_info_num_;
  size_t input_valid_num_;
  size_t output_valid_num_;
  size_t args_infos_size_;
  uint8_t args_infos_[8];
};
static_assert(std::is_standard_layout<ArgsInfosDesc>::value, "The class ArgsInfosDesc must be a POD");

class IoArgsInfo {
 public:
  struct IoArg {
    void SetIoArg(size_t cur_arg_offset, int32_t cur_start_index, size_t cur_arg_num, bool cur_is_input,
                  bool cur_is_need_merged_copy, bool cur_folded_first, DynDesc cur_dyn_desc, size_t cur_data_size) {
      arg_offset = cur_arg_offset;
      start_index = cur_start_index;
      arg_num = cur_arg_num;
      is_input = cur_is_input;
      is_need_merged_copy = cur_is_need_merged_copy;
      folded_first = cur_folded_first;
      dyn_desc = cur_dyn_desc;
      data_size = cur_data_size;
    }
    size_t arg_offset;
    int32_t start_index;
    size_t arg_num;
    bool is_input;
    bool is_need_merged_copy;
    bool folded_first;
    DynDesc dyn_desc;
    size_t data_size;
  };

  void Init(size_t io_arg_num, void *const io_arg_data) {
    io_arg_num_ = io_arg_num;
    io_arg_data_ = static_cast<IoArg *>(io_arg_data);
  }
  size_t GetIoArgNum() const {
    return io_arg_num_;
  }
  size_t GetIoArgSize() const {
    return io_arg_num_ * sizeof(IoArg);
  }
  void SetIoArgByIndex(size_t idx, const IoArg &io_arg) {
    if (idx >= io_arg_num_) {
      GELOGE(ge::PARAM_INVALID, "[PARAM][INVALID] Input index[%zu], expect in range[0, %zu).", idx, io_arg_num_);
      return;
    }
    io_arg_data_[idx].SetIoArg(io_arg.arg_offset, io_arg.start_index, io_arg.arg_num, io_arg.is_input,
                               io_arg.is_need_merged_copy, io_arg.folded_first, io_arg.dyn_desc, io_arg.data_size);
  }
  const IoArg *GetIoArgByIndex(size_t idx) const {
    if (idx >= io_arg_num_) {
      GELOGE(ge::PARAM_INVALID, "[PARAM][INVALID] Input index[%zu], expect in range[0, %zu).", idx, io_arg_num_);
      return nullptr;
    }
    auto io_args_base = static_cast<const IoArg *>(io_arg_data_);
    return (io_args_base + idx);
  }
 private:
  size_t io_arg_num_;
  IoArg *io_arg_data_;
};
static_assert(std::is_standard_layout<IoArgsInfo>::value, "The class IoArgsInfo must be a POD");

/*
   * RTS KernelLaunch到Device的args组成：
   *   |compiled-args| <--- rtArgsEx_t::args基址
   *   |inputs-addr|
   *   |outputs-addr|
   *   |shape-buffer-addr|
   *   |workspaces-addr|
   *   |tiling-data-addr|
   *   |overflow-addr|
   *   |tiling data|
   *   |host input data|
   */
struct RtKernelLaunchArgsEx {
 public:
  enum class ArgsType {
    kArgsCompiledArgs,
    kArgsInputsAddr,
    kArgsOutputsAddr,
    kShapeBufferAddr,
    kWorkspacesAddr,
    kTilingDataAddr,
    kOverflowAddr,
    kTilingData,
    kHostInputData,
    kHostInputInfo,
    // add new args here
    kArgsTypeEnd
  };
  enum class TilingCacheStatus {
    kDisabled,
    kMissed,
    kHit
  };
  struct HostInputInfo {
    void *host_input_addr;
    const ContinuousVector *inputs_index;
    size_t host_tensor_size;
  };
  struct ComputeNodeDesc {
    static std::unique_ptr<uint8_t[]> Create(size_t compiled_arg_size, size_t &total_size);
    bool ArgsCountValid() const;
    size_t GetCompiledArgsCap() const {
      return compiled_args_size;
    }

    size_t input_num;
    size_t output_num;
    size_t workspace_cap;
    size_t max_tiling_data;
    bool need_shape_buffer;
    bool need_overflow;
    size_t compiled_args_size;
    uint8_t compiled_args[8];
  };
  static_assert(std::is_standard_layout<ComputeNodeDesc>::value, "The class ComputeNodeDesc must be a POD");
  struct ArgsDesc {
    bool SetSize(ArgsType args_type, size_t len) {
      return !ge::AddOverflow(offsets[static_cast<uint32_t>(args_type)], len,
                              offsets[static_cast<uint32_t>(args_type) + 1]);
    }
    bool UpdateOffset(ArgsType args_type, size_t new_offset) {
      if ((new_offset < offsets[static_cast<uint32_t>(ArgsType::kTilingData)]) ||
          (new_offset > offsets[static_cast<uint32_t>(ArgsType::kHostInputInfo)])) {
        return false;
      }
      offsets[static_cast<uint32_t>(args_type)] = new_offset;
      return true;
    }
    size_t GetOffset(ArgsType args_type) const {
      return offsets[static_cast<uint32_t>(args_type)];
    }
    size_t GetCap(ArgsType args_type) const {
      return GetOffset(static_cast<ArgsType>(static_cast<uint32_t>(args_type) + 1)) - GetOffset(args_type);
    }
    size_t GetArgsTotalSize() const {
      return offsets[static_cast<uint32_t>(ArgsType::kArgsTypeEnd)];
    }
    size_t offsets[static_cast<uint32_t>(ArgsType::kArgsTypeEnd) + 2];
  };
  static_assert(std::is_standard_layout<ArgsDesc>::value, "The class ArgsDesc must be a POD");
  struct ArgsCacheInfo {
    TilingCacheStatus cache_status;
    void *tiling_cache;
  };
  static_assert(std::is_standard_layout<ArgsCacheInfo>::value, "The class ArgsCacheInfo must be a POD");

  template <typename T>
  T *GetArgsPointer(ArgsType args_type) {
    if (args_type > ArgsType::kArgsTypeEnd) {
      return nullptr;
    }
    auto arg_type_offset = args_desc_.GetOffset(args_type);
    auto args_buffer_base = reinterpret_cast<uint8_t *>(base.args);
    return reinterpret_cast<T *>(args_buffer_base + arg_type_offset);
  }

  template <typename T>
  const T *GetArgsPointer(ArgsType args_type) const{
    if (args_type > ArgsType::kArgsTypeEnd) {
      return nullptr;
    }
    auto arg_type_offset = args_desc_.GetOffset(args_type);
    auto args_buffer_base = reinterpret_cast<uint8_t *>(base.args);
    return reinterpret_cast<T *>(args_buffer_base + arg_type_offset);
  }

  size_t GetArgsCap(ArgsType args_type) const {
    return args_desc_.GetCap(args_type);
  }
  size_t GetArgsOffset(ArgsType args_Type) {
    return args_desc_.GetOffset(args_Type);
  }

  ge::graphStatus SetIoAddr(size_t io_arg_offset, void *addr) {
    FE_ASSERT_TRUE(io_arg_offset < base.argsSize, "[PARAM][INVALID] Arg offset[%zu], expect in range[0, %u).",
                   io_arg_offset, base.argsSize);
    auto args_buffer_base = static_cast<uint8_t *>(base.args);
    auto io_addr_addr = reinterpret_cast<TensorAddress *>(args_buffer_base + io_arg_offset);
    *io_addr_addr = addr;
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus SetWorkspaceAddr(size_t index, void *addr) {
    auto workspace = GetArgsPointer<TensorAddress>(ArgsType::kWorkspacesAddr);
    if (workspace == nullptr) {
      return ge::GRAPH_FAILED;
    }
    workspace[index] = addr;
    return ge::GRAPH_SUCCESS;
  }
  rtArgsEx_t *GetBase() {
    return &base;
  }

  const rtArgsEx_t *GetBase() const {
    return &base;
  }

  // 返回需要拷贝到Device的args基地址
  void *GetArgBase() {
    return base.args;
  }
  // 返回需要拷贝到Device的args基地址
  const void *GetArgBase() const {
    return base.args;
  }
  ge::graphStatus SetShapebufferAddr(void *addr) {
    auto shapebuffer = GetArgsPointer<TensorAddress>(ArgsType::kShapeBufferAddr);
    if (shapebuffer == nullptr) {
      return ge::GRAPH_FAILED;
    }
    shapebuffer[0] = addr;
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus UpdateBaseByTilingSize(size_t tiling_data_size) {
    if (tiling_data_size == 0U) {
      base.hasTiling = false;
    } else {
      base.hasTiling = true;
    }
    return ge::GRAPH_SUCCESS;
  }

  TilingData &GetTilingData() {
    return tilingData;
  }

  const TilingData &GetTilingData() const {
    return tilingData;
  }

  ge::graphStatus UpdateBaseArgsSize() {
    if (base.hostInputInfoNum != 0) {
      base.argsSize = args_desc_.GetOffset(ArgsType::kHostInputData) + host_input_data_size_;
      GELOGD("host input info num is %u, args size is %u", base.hostInputInfoNum, base.argsSize);
      return ge::GRAPH_SUCCESS;
    }
    size_t tiling_data_size = tilingData.GetDataSize();
    if (tiling_data_size != 0) {
      base.argsSize = args_desc_.GetOffset(ArgsType::kTilingData) + tiling_data_size;
    } else {
      base.argsSize = args_desc_.GetOffset(ArgsType::kTilingData);
    }
    GELOGD("args size is %u", base.argsSize);
    return ge::GRAPH_SUCCESS;
  }

  size_t GetHostInputDataSize() const {
    return host_input_data_size_;
  }

  ge::graphStatus UpdateHostInputArgs(HostInputInfo &host_mem_input);

  ge::graphStatus UpdateMergedCopyInfo();

  ge::graphStatus AlignHostInputSize();

  static std::unique_ptr<uint8_t[]> Create(const ComputeNodeDesc &node_desc, const ArgsInfosDesc &args_info_desc);

  std::unique_ptr<uint8_t[]> MakeCopy();

  size_t GetBackFillSecondaryAddrNum() const {
    return back_fill_secondary_addr_num_;
  }
  size_t GetMergedCopySize() const {
    return merged_copy_size_;
  }
  IoArgsInfo &GetIoArgsInfo() {
    return io_args_info_;
  }
  const IoArgsInfo &GetIoArgsInfo() const {
    return io_args_info_;
  }
  size_t GetWorkspaceCap() const {
    return node_desc_.workspace_cap;
  }

  const ArgsCacheInfo &GetArgsCacheInfo() const {
    return args_cache_info_;
  }

  void SetTilingCacheStatus(TilingCacheStatus cache_status) {
    args_cache_info_.cache_status = cache_status;
  }

  void SetTilingCache(void* tiling_cache) {
    args_cache_info_.tiling_cache = tiling_cache;
  }

 private:
  static bool CalcTotalSize(const ComputeNodeDesc &node_desc, const ArgsInfosDesc &args_info_desc, ArgsDesc &args_desc,
                            size_t &total_size);
  bool Init(const ComputeNodeDesc &node_desc, const ArgsInfosDesc &args_info_desc, const ArgsDesc &args_desc,
            const size_t total_size);
  ge::Status InitIoArgsInfo(const ArgsInfosDesc &args_info_desc, const ComputeNodeDesc &node_desc);
  ge::Status InitHostInputInfoByDynamicIo();

 private:
  rtArgsEx_t base;
  ComputeNodeDesc node_desc_;
  ArgsInfosDesc args_info_desc_;
  ArgsDesc args_desc_;
  TilingData tilingData;
  IoArgsInfo io_args_info_;
  ArgsCacheInfo args_cache_info_;
  size_t host_input_data_size_;
  size_t back_fill_secondary_addr_num_;  // 需要回刷的地址数量
  size_t merged_copy_size_;   // 需要合并拷贝的动态TensorList
  size_t total_size_;
  uint8_t data[8]; // data layout: | IoArgsInfo::io_arg_data_ | rtArgsEx_t::args
};
static_assert(std::is_standard_layout<RtKernelLaunchArgsEx>::value, "The class RtKernelLaunchArgsEx must be a POD");
#ifndef _MSC_VER  // TilingData 里有构造函数和赋值运算符，导致MSVC编译器判定POD失败
static_assert(std::is_pod<RtKernelLaunchArgsEx>::value, "The class RtKernelLaunchArgsEx must be a POD");
#endif

enum class AllocLaunchArgInputs {
  kNodeDesc,
  kArgsInfoDesc
};

enum class AllocLaunchArgOutputs {
  kHolder,
  kRtArg,
  kRtArgArgsBase,
  kTilingDataBase,
  kNum
};
ge::graphStatus ProcFoldedDescArgs(const ge::NodePtr &compute_node, std::vector<ArgsInfosDesc::ArgsInfo> &args_infos);
void SetDynShape(const Shape &shape, uint8_t *host_addr, const DynDesc &dyn_desc, size_t &shape_offset);
void AssignDynamicDesc(DynDesc &dyn_desc, size_t j, int64_t index, size_t dyn_num);
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_RT_KERNEL_LAUNCH_ARGS_EX_H_

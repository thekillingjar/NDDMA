/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_RT_KERNEL_LAUNCH_ARGS_EX_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_RT_KERNEL_LAUNCH_ARGS_EX_H_
#include <memory>
#include <cstdint>
#include "graph/utils/node_utils.h"
#include "graph/utils/math_util.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "runtime/kernel.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tiling_data.h"

namespace gert {
/*
  * FFTSPlus AICore args组成：
  *   |tiling_data|
  *   |tail_tiling_data|
  *   |args_addr|
  *   |atomic_tiling_data|
  *   |atomic_tail_tiling_data|
  *   |atomic_args_addr|
  *   |dynamic inputs desc|   save DynDesc using in runtime
  *   |dynamic inputs addr|
  *   |dynamic outputs desc|  save DynDesc using in runtime
  *   |dynamic outputs addr|
  */
struct RtFFTSKernelLaunchArgs {
 public:
  enum FFTSArgsType {
    kTilingData,
    kTailTilingData,
    kArgsHostAddr,
    kAtomTilingData,
    kAtomTailTilingData,
    kAtomArgsHostAddr,
    kDyInputsDescData,
    kDyInputsHostAddr,
    kDyOutputsDescData,
    kDyOutputsHostAddr,
    kArgsTypeEnd
  };
  struct ComputeNodeDesc {
    static std::unique_ptr<uint8_t[]> Create(size_t &total_size);
    size_t addr_num{0U}; // for mixl2
    size_t input_num{0U};
    size_t output_num{0U};
    size_t workspace_cap{0U};
    size_t max_tiling_data{0U};
    size_t max_tail_tiling_data{0U};
    size_t thread_num_max{0U};
    bool need_atomic{false};
    bool dynamic_folded{false};
    size_t max_atom_tiling_data{0U};
    size_t max_atom_tail_tiling_data{0U};
  };
  static_assert(std::is_standard_layout<ComputeNodeDesc>::value, "The class ComputeNodeDesc must be a POD");
  struct ArgsDesc {
    bool SetSize(FFTSArgsType args_type, size_t len) {
      return !ge::AddOverflow(offsets[args_type], len, offsets[static_cast<uint32_t>(args_type) + 1]);
    }
    size_t GetOffset(FFTSArgsType args_type) const {
      return offsets[args_type];
    }
    size_t GetCap(FFTSArgsType args_type) const {
      return GetOffset(static_cast<FFTSArgsType>(static_cast<uint32_t>(args_type) + 1)) - GetOffset(args_type);
    }
    size_t GetArgsTotalSize() const {
      return offsets[kArgsTypeEnd];
    }
    size_t offsets[static_cast<uint32_t>(kArgsTypeEnd) + 2];
  };
  static_assert(std::is_standard_layout<ArgsDesc>::value, "The class ArgsDesc must be a POD");

  template<typename T>
  T *GetArgsPointer(FFTSArgsType args_type) {
    if (args_type > kArgsTypeEnd) {
      return nullptr;
    }
    return reinterpret_cast<T *>(&args_[args_desc_.GetOffset(args_type)]);
  }
  size_t GetArgsCap(FFTSArgsType args_type) {
    return args_desc_.GetCap(args_type);
  }

  void *GetArgBase() {
    return args_;
  }

  TilingData& GetTilingData() {
    return tiling_data_;
  }
  size_t GetTilingAbsPos() const {
    return tiling_abs_pos_;
  }
  size_t GetAtomTilingAbsPos() const {
    return atom_tiling_abs_pos_;
  }
  TilingData& GetTailTilingData() {
    return tail_tiling_data_;
  }
  TilingData& GetAtomTilingData() {
    return atom_tiling_data_;
  }
  TilingData& GetAtomTailTilingData() {
    return atom_tail_tiling_data_;
  }
  size_t GetArgsSize() const {
    return args_size_;
  }
  size_t GetTilingOffset() const {
    return tiling_offset_;
  }
  size_t GetArgsPos() const {
    return args_pos_;
  }
  size_t GetArgsAbsPos() const {
    return args_abs_pos_;
  }
  size_t GetAtomTilingOffset() const {
    return atom_tiling_offset_;
  }
  size_t GetAtomArgsPos() const {
    return atom_args_pos_;
  }
  size_t GetAtomArgsAbsPos() const {
    return atom_args_abs_pos_;
  }
  bool NeedAtomicClean() const {
    return node_desc_.need_atomic;
  }
  size_t GetWorkspaceCap() const {
    return node_desc_.workspace_cap;
  }
  static std::unique_ptr<uint8_t[]> Create(const ge::NodePtr &node, ComputeNodeDesc &node_desc, size_t &size);
  ge::graphStatus RedirectTilingAddr();
  ge::graphStatus SetIoAddr(size_t io_index, size_t &arg_index, uintptr_t data_base,
                            const Shape &shape, void *args_dev_base);
  const DynDesc *GetDyDescByIoIndex(size_t io_index);
 private:
  static ge::graphStatus CalcDynamicArgsSize(const ge::NodePtr &node, RtFFTSKernelLaunchArgs::ArgsDesc &args_desc,
      std::vector<std::vector<int64_t>> &dyn_io_vv, bool is_input, DynDescInfo &dyn_desc_v);
  static ge::graphStatus CalcAtomicArgsSize(const RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc,
                                            RtFFTSKernelLaunchArgs::ArgsDesc &args_desc);
  static ge::graphStatus CalcTotalSize(const ge::NodePtr &node, ComputeNodeDesc &node_desc, ArgsDesc &args_desc,
                                       size_t &total_size, DynDescInfo &dyn_desc_v);
  ge::graphStatus Init(const ComputeNodeDesc &node_desc, const ArgsDesc &args_desc, DynDescInfo &dyn_desc_v);
  void SetDynInAddr(size_t in_index, size_t &arg_index, uintptr_t data_base, const Shape &shape, void *args_dev_base);
  void SetDynOutAddr(size_t out_index, size_t &arg_index, uintptr_t data_base, const Shape &shape, void *args_dev_base);
 private:
  ComputeNodeDesc node_desc_;
  ArgsDesc args_desc_;
  TilingData tiling_data_;
  TilingData tail_tiling_data_;
  TilingData atom_tiling_data_;
  TilingData atom_tail_tiling_data_;
  size_t tiling_offset_{0U};
  size_t tiling_abs_pos_{0U};
  size_t atom_tiling_offset_{0U};
  size_t atom_tiling_abs_pos_{0U};
  size_t args_pos_{0U};
  size_t args_abs_pos_{0U};
  size_t atom_args_pos_{0U};
  size_t atom_args_abs_pos_{0U};
  size_t args_size_{0U};
  size_t dyn_in_num_{0U};
  size_t dyn_out_num_{0U};
  size_t shape_offset_{0U};  // record dynamic io every group shape offset
  uint8_t args_[1];  // todo : warning！！！！！！！！！！！ this addr need to be align in 64byte ！！！！！！！！！！！！！！
};
static_assert(std::is_standard_layout<RtFFTSKernelLaunchArgs>::value, "The class RtFFTSKernelLaunchArgs must be a POD");

enum class AllocFFTSArgInputs {
  kArgsPara,
};

enum class AllocFFTSArgOutputs {
  kTilingDataBase,
  kTailTilingDataBase,
  kAtomTilingDataBase,
  kAtomTailTilingDataBase,
  kNum
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_RT_KERNEL_LAUNCH_ARGS_EX_H_

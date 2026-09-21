/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file norm.h
 * \brief
 */

#ifndef NORM_TILING_H
#define NORM_TILING_H

#include <cmath>
#include <vector>

#include "register/op_compile_info_base.h"
#include <nlohmann/json.hpp>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"

// #include "graph/utils/op_desc_utils.h"
// #include "vector_tiling.h"

namespace gert {
constexpr std::size_t NORM_MAX_DIM_LEN = 8;
constexpr std::size_t NORM_MAX_INPUT_NUMS = 70;
constexpr std::size_t NORM_MAX_WORKSPACE_NUMS = 20;
constexpr std::size_t NORM_MAX_VAR_NUMS = 40;
constexpr int64_t NORM_FAKE_WORKSPACE_SIZE = 32;
constexpr int32_t NORM_NONE_SPLIT_KEY = 9;
constexpr int32_t NORM_REDUCE_PATTERN_WEIGHT = 1000;
constexpr int32_t NORM_COMMON_SCH_TYPE = 0;
constexpr int32_t NORM_PARTIAL_REORDER_SCH_TYPE = 1;
constexpr int32_t NORM_ALIGNED_IN_UB_SCH_TYPE = 2;
constexpr int32_t NORM_LAST_REDUCE_ALIGN_SCH_TYPE = 3;

struct NormCompileInfo {
  // construct func
  // NormCompileInfo() = default;
  // NormCompileInfo(const std::string& op_type, const nlohmann::json &compile_info);
  // check value
  bool Check(const std::string& op_type) const;

  bool check_success{true};
  // reduce and broadcast axis
  // std::string reduce_attr_name;
  int32_t reduce_attr_index;
  bool is_reduce_attr_is_int{true};
  int32_t reduce_axis_type{0};
  std::vector<int32_t> ori_reduce_axis;
  std::vector<int32_t> broadcast_axis_type;
  std::vector<int32_t> ori_broadcast_axis;
  bool is_broadcast_axis_known{false};
  // disable fuse axes
  std::vector<int32_t> ori_disable_fuse_axes;
  // graph info
  std::vector<int32_t> input_type;
  bool exist_output_after_reduce{false};
  bool exist_workspace_after_reduce{false};
  std::unordered_map<int32_t, std::vector<int32_t>> available_ub_size;
  // common info
  int32_t core_num{-1};
  int32_t min_block_size{-1};
  int32_t pad_max_entire_size{-1};
  // workspace info
  std::unordered_map<int32_t, std::vector<int32_t>> workspace_info;
  // norm vars
  std::unordered_map<int32_t, std::vector<int32_t>> norm_vars;
  // fuse axis
  bool is_fuse_axis{true};
  // const
  bool is_const{false};
  bool is_const_post{false};
  std::unordered_map<int32_t, int32_t> const_block_dims;
  // std::unordered_map<std::uint64_t, vector<VarAttr>> var_attr_map;

  // private:
  void ParseAxisInfo(const nlohmann::json& compile_info);
  void ParseGraphInfo(const nlohmann::json& compile_info);
  void ParseCommonInfo(const nlohmann::json& compile_info);
  void ParseOtherInfo(const nlohmann::json& compile_info);
};

struct NormTilingInfo {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
};

struct NormReorderInfo {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  //    vector.idx     :      vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
};

enum class NormBroadcastMode {
  NO_BROADCAST = 0,
  SAME_REDUCE = 1,
  OPPOSITE_REDUCE = 2,
  ALL_BROADCAST = 3,
  OTHERS = 4
};

enum class NormAxisType {
  COMMON = 0,
  SAME_REDUCE = 1,
  OPPOSITE_REDUCE = 2,
  AFTER = 3,
  BEFORE = 4,
  OTHERS = 5
};

class Norm {
  public:
    explicit Norm(const std::string& op_type, TilingContext *context)
        : op_type(op_type), context(context) {
    }
    ~Norm() = default;
    bool DoTiling();
    // bool DoTiling(const OpInfo& op_info);

  private:
    bool CheckInputNum() const;
    bool GetMaxDimLen();
    bool GetInput();
    bool InitReduce();
    // bool InitReduce(const OpInfo& op_info);
    bool InitBroadcast();
    bool Init();
    bool GetNormPattern();
    bool EliminateOne();
    void InitAxisBaseFlag(std::array<int32_t, NORM_MAX_DIM_LEN>& flags, const int32_t& reduce_flag,
                          const int32_t& broadcast_flag, const int32_t& reduce_broadcast_flag) const;
    bool FusedAxis();
    bool GetUbSizeInfo();
    bool ProcessBlockTilingAndReorder();
    bool ProcessTiling();
    bool JudgePartialReorderZeroDimSplitBlock(const int64_t& right_product, const int64_t& right_align_product,
                                              const std::size_t& index);
    bool JudgeCurDimSplitBlock(const int64_t& right_product, const std::size_t& index);
    bool JudgeCurDimSplitBlock(const int64_t& left_product, const int64_t& right_product,
                               const std::size_t& index, const int64_t& max_block_factor = 0);
    bool GetPartialReorderBlockTilingInfo();
    bool GetWorkspaceBlockTilingInfo();
    bool GetBlockTilingInfo();
    bool ProcessReorderAxis();
    bool GetPartialReorderUbTilingInfo();
    bool CheckNormalCurUbFactor(const int64_t& cur_ub_factor, const int64_t& cur_dim,
                                const int64_t& cur_dim_tail, const int64_t& right_product) const;
    bool JudgeNormalCurDimSplitUb(const std::size_t& index);
    bool JudgeWorkspaceCurDimSplitUb(const std::size_t& index);
    bool GetUbTilingInfo();
    bool NeedRefineBlockTiling();
    bool ConstPostCalcTiling();
    bool CalcNormInfo();
    bool CalcTiling();
    bool ConstPostWriteTilingData();
    bool CalcTilingKey();
    bool CalcWorkspace();
    bool WriteTilingData();

    bool IsNeedPartialReorder();
    bool IsNeedWorkspace() const;
    bool IsNeedAlignedInUb() const;
    bool GetVarValue();
    bool CalcInputAlignShape();
    NormBroadcastMode JudgeBroadcastMode(const std::array<int64_t, NORM_MAX_DIM_LEN>& before_broadcast_shape) const;
    int32_t CalcMinEliminateOneIndex() const;
    int32_t GetBlockDim(int32_t tiling_axis, int64_t tiling_factor) const;
    int64_t CalcReorderShapeProduct(int32_t axis_index, bool exclude_reduce_axis) const;
    int64_t CalcReorderAlignShapeProduct(int32_t axis_index) const;

    const std::string &op_type;
    TilingContext *context;
    const NormCompileInfo* compileInfo;

    NormTilingInfo tilingInfo;
    NormReorderInfo reorderInfo;

    std::array<std::array<int64_t, NORM_MAX_DIM_LEN>, NORM_MAX_INPUT_NUMS> before_broadcast_shapes{};
    std::vector<int64_t> input_shape_ori{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> broadcast_axis_ori{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int64_t> input_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> reduce_axis{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int32_t> broadcast_axis{std::vector<int32_t>(NORM_MAX_DIM_LEN, 0)};

    // assistant
    std::vector<int64_t> input_align_shape{std::vector<int64_t>(NORM_MAX_DIM_LEN, 0)};
    std::vector<int64_t> workspace{std::vector<int64_t>(NORM_MAX_WORKSPACE_NUMS, 0)};
    std::vector<int32_t> var_value{std::vector<int32_t>(NORM_MAX_VAR_NUMS, 0)};

    bool is_last_axis_reduce{false};
    bool is_need_workspace{false};
    bool is_partial_reorder{false};
    bool is_continuous_data_move{false};
    bool is_discontinuous_reduce_axis{false};
    bool is_align_and_remove_pad{false};

    int64_t after_reduce_align_shape_product{-1};
    int64_t after_reduce_shape_product{-1};
    int64_t reduce_product{-1};

    int32_t last_r_axis_index{-1};
    int32_t first_a_axis_index{-1};
    int32_t block_tiling_axis_index_in_reorder{-1};
    int32_t last_a_axis_index_in_reorder{-1};

    int32_t reduce_pattern{0};
    int32_t broadcast_pattern{0};
    int32_t norm_pattern{0};
    int32_t sch_type{0};
    int32_t db{0};
    int32_t block_size{-1};
    int64_t ub_size{-1};
    int32_t tiling_key{0};
    std::size_t max_dim_len{0};
    std::size_t before_broadcast_input_num{0};

    int32_t common_max_ub_count{-1};
    int32_t workspace_max_ub_count{-1};
    int32_t pad_max_ub_count{-1};
};

}  // namespace gert

#endif  // NORM_TILING_H

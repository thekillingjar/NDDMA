/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_CXX_ATT_GEN_MODEL_INFO_TILING_DATA_GEN_TILING_DATA_MANAGER_H_
#define ATT_CXX_ATT_GEN_MODEL_INFO_TILING_DATA_GEN_TILING_DATA_MANAGER_H_

#include <unordered_map>
#include <vector>
#include "base/model_info.h"
#include "extra_info_gen/extra_info_config.h"
#include "ge_common/ge_api_error_codes.h"

namespace att {
enum class TilingDataGenType {
  AXES_TILING_DATA_GEN,
  GENERAL_TILING_DATA_GEN,
  MEMORY_TILING_DATA_GEN,
  ALL_TILING_DATA_GEN,
  TILING_DATA_GEN_TYPE_ERR = std::numeric_limits<int32_t>::max(),
};

using TilingDataMap = std::unordered_map<std::string, std::pair<TilingDataType, std::string>>;
class TilingDataGenBase {
 public:
  explicit TilingDataGenBase(const TilingDataGenType tiling_data_gen_type, const ModelInfo &model_info)
      : tiling_data_gen_type_(tiling_data_gen_type), model_info_(model_info){};
  virtual ~TilingDataGenBase() = default;
  virtual ge::Status Init() = 0;
  virtual std::vector<std::pair<std::string, std::string>> GetTilingDataWithAnnotation() const;
  virtual std::vector<std::string> GetTilingFuncImpl(const std::string &tiling_type) const {
    (void)tiling_type;
    return std::vector<std::string>();
  }
  virtual std::string GetTilingFuncInvoke() const {
    return "";
  }
  TilingDataGenType GetTilingDataGenType() const {
    return tiling_data_gen_type_;
  }

 protected:
  const TilingDataGenType tiling_data_gen_type_;
  const ModelInfo &model_info_;
  // key: tiling_data_name
  // value: {tiling_data_type, tiling data expr}
  TilingDataMap tiling_data_map_{};
};

class AxesTilingDataGen : public TilingDataGenBase {
  friend class BlockDimTilingDataGen;

 public:
  explicit AxesTilingDataGen(const ModelInfo &model_info)
      : TilingDataGenBase(TilingDataGenType::AXES_TILING_DATA_GEN, model_info){};
  ~AxesTilingDataGen() override = default;
  // gen model info
  ge::Status Init() override;
  bool IsInitialized() const {
    return is_initialized_;
  }
  // gen code
  std::vector<std::pair<std::string, std::string>> GetTilingDataWithAnnotation() const override;
  std::vector<std::string> GetTilingFuncImpl(const std::string &tiling_type) const override;
  std::string GetTilingFuncInvoke() const override;

 private:
  // 添加轴size对齐表达式
  ge::Status AddAxesAlignedSize();
  // 添加轴的tail_size和loop_num表达式
  ge::Status AddAxesTailSizeAndLoopNum();
  // 添加尾块的tail_size和尾块的loop_num表达式
  ge::Status AddSplitOuterAxisTailArgs();
  ge::Status SetAxisArgExpr(const std::string &axis_name, const AxisTilingData &axis_tiling_data);
  std::vector<AxisTilingData> GetAxisTilingData(const std::string &axis_name) const;
  std::pair<std::string, std::string> GetAxisTilingData(const std::string &axis_name,
                                                        const TilingDataType arg_type) const;
  void MakeSureParentAxesFirst();
  std::vector<std::pair<std::string, std::string>> GetAxesTilingDataWithExpr() const;
  Expr GetArgExpr(const std::string &axis_name) const;

  // key: axis_name
  // value: [AxisTilingData(AXIS_LOOP_NUM, s0t_loop_num, Ceil(s0T / s0t), AxisTilingData(AXIS_TAIL_SIZE, s0T % s0t)...]
  std::unordered_map<std::string, std::vector<AxisTilingData>> axes_tiling_data_map_{};
  // depend on MakeSureParentAxesFirst, get ordered(parent->child) tiling data names
  std::vector<std::string> ordered_axes_names_;
  bool is_initialized_{false};
};

class BlockDimTilingDataGen : public TilingDataGenBase {
 public:
  explicit BlockDimTilingDataGen(const std::shared_ptr<AxesTilingDataGen> &axes_tiling_data_gen,
                                 const ModelInfo &model_info)
      : TilingDataGenBase(TilingDataGenType::GENERAL_TILING_DATA_GEN, model_info),
        axes_tiling_data_gen_(axes_tiling_data_gen){};
  ~BlockDimTilingDataGen() override = default;
  ge::Status Init() override;
  std::vector<std::string> GetTilingFuncImpl(const std::string &tiling_type) const override;
  std::string GetTilingFuncInvoke() const override;

 private:
  ge::Status AddUsedCoreNum();

  const std::shared_ptr<const AxesTilingDataGen> axes_tiling_data_gen_;
};

class MemoryTilingDataGen : public TilingDataGenBase {
 public:
  explicit MemoryTilingDataGen(const ModelInfo &model_info)
      : TilingDataGenBase(TilingDataGenType::MEMORY_TILING_DATA_GEN, model_info){};
  ~MemoryTilingDataGen() override = default;
  ge::Status Init() override;
  std::vector<std::string> GetTilingFuncImpl(const std::string &tiling_type) const override;
  std::string GetTilingFuncInvoke() const override;

 private:
  std::string GenFuncImpl(const std::pair<std::string, Expr> &var_name_to_expr,
                          const std::string &tiling_type, const ExprExprMap &container_expr) const;
  std::string GenFuncInvoke(const std::string &var_name) const;
};

using TilingDataGenPtr = std::shared_ptr<TilingDataGenBase>;
class TilingDataGenerator {
 public:
  TilingDataGenerator(const std::vector<ModelInfo> &model_info_list, const ExtraInfoConfig &extra_info_config)
      : model_info_list_(model_info_list), extra_info_config_(extra_info_config){};
  virtual ~TilingDataGenerator() = default;
  // gen model info
  ge::Status Init();
  // gen code
  std::vector<std::pair<std::string, std::string>> GetTilingDataWithAnnotation(
      const uint32_t tiling_key, const TilingDataGenType tiling_data_gen_type) const;
  std::vector<std::pair<std::string, std::string>> GetTilingDataWithAnnotation(
      const TilingDataGenType tiling_data_gen_type) const;
  std::vector<std::string> GetTilingFuncImpl(const uint32_t tiling_key,
                                             const TilingDataGenType tiling_data_gen_type) const;
  std::string GetTilingFuncInvoke(const uint32_t tiling_key, const TilingDataGenType tiling_data_gen_type) const;

 private:
  TilingDataGenerator(const TilingDataGenerator &) = delete;
  TilingDataGenerator &operator=(const TilingDataGenerator &) = delete;
  std::vector<TilingDataGenPtr> GetTilingDataGens(const uint32_t tiling_key) const;
  ge::Status GenTilingData(const ModelInfo &model_info);

  // key: tiling_key
  std::unordered_map<uint32_t, std::vector<TilingDataGenPtr>> graphs_tiling_data_gens_;
  const std::vector<ModelInfo> &model_info_list_;
  const ExtraInfoConfig &extra_info_config_;
  bool inited_{false};
};
}  // namespace att

#endif  // ATT_CXX_ATT_GEN_MODEL_INFO_AXES_TILING_GEN_TILING_DATA_MANAGER_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_DSA_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_DSA_TASK_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "proto/task.pb.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/aicore_util_constants.h"
#include "graph/types.h"

namespace fe {
enum class DistributionType {
  DIS_BITMASK = 0b000,
  DIS_UNIFORM = 0b001,
  DIS_NORMAL = 0b010,
  DIS_TRUNCATED_NORMAL = 0b011
};

enum class InputVldType {
  VLD_BITMASK_DPT_RATIO = 0b00001,
  VLD_UNIFORM_MIN_MAX = 0b00110,
  VLD_NORMAL_MEAN_STDDEV = 0b11000
};

enum class DsaValueType {
  DSA_DATA_VALUE = 0b1,
  DSA_DATA_ADDR = 0b0
};

class DsaTaskBuilder {
  struct DSAFlags {
    DistributionType distribution_type;
    InputVldType input_vld;
    DsaValueType input1_type;
    DsaValueType input2_type;
    DsaValueType counter_type;
    DsaValueType seed_type;
    DsaValueType rand_count_type;
    uint32_t CalAddrOrValueFlag() const;
  };

  struct DsaWorkspace {
    uint64_t philox_count_addr;
    uint64_t input_addr;
  };

  struct DsaInput {
    std::string random_count;
    std::string input1;
    std::string input2;
    std::string counter;
    std::string seed;
  };

 public:
  DsaTaskBuilder();
  ~DsaTaskBuilder();

  /*
   * @ingroup fe
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenerateTask(const ge::Node &node, const ge::RunContext &context, std::vector<domi::TaskDef> &task_defs);

 private:
  DsaTaskBuilder(const DsaTaskBuilder &builder) = delete;
  DsaTaskBuilder &operator=(const DsaTaskBuilder &builder) = delete;

  Status GetDsaValueOrAddrFlags(const ge::Node &node, DSAFlags &flags) const;
  Status GetDataType(const ge::Node &node, uint32_t &type) const;
  Status GetWorkspaceInfo(const ge::Node &node, DsaWorkspace &workspace) const;
  Status GetOutputAddr(const ge::Node &node, uint64_t &addr) const;
  Status GetInputs(const ge::Node &node, const DSAFlags &flags, DsaInput &inputs) const;

  static bool IsConstInput(const ge::Node &node, uint32_t input_idx);
  static bool GetConstInputValue(const ge::Node &node, uint32_t input_idx, std::string &value);
  static bool GetInputDataType(const ge::Node &node, uint32_t input_idx, ge::DataType &data_type);
  static std::string GetValueDebugString(const ge::Node &node, uint32_t input_idx);

 private:
  TaskBuilderContext context_;
  const uint32_t dsa_type_ = 15;
  const uint32_t dsa_start_ = 1;
  const uint32_t dsa_philox_type_ = 0;
  const std::string engine_name_ = fe::kDsaCoreName;
  const std::string truncated_normal_ = "DSARandomTruncatedNormal";
  const std::string normal_ = "DSARandomNormal";
  const std::string bitmask_ = "DSAGenBitMask";
  const std::string uniform_ = "DSARandomUniform";
  const std::string stateless_truncated_normal_ = "DSAStatelessRandomTruncatedNormal";
  const std::string stateless_normal_ = "DSAStatelessRandomNormal";
  const std::string stateless_bitmask_ = "DSAStatelessGenBitMask";
  const std::string stateless_uniform_ = "DSAStatelessRandomUniform";
  const std::map<std::string, DSAFlags> dsa_opname_values_ = {
      {truncated_normal_,
      {DistributionType::DIS_TRUNCATED_NORMAL, InputVldType::VLD_NORMAL_MEAN_STDDEV, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {normal_,
      {DistributionType::DIS_NORMAL, InputVldType::VLD_NORMAL_MEAN_STDDEV, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {bitmask_,
      {DistributionType::DIS_BITMASK, InputVldType::VLD_BITMASK_DPT_RATIO, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {uniform_,
      {DistributionType::DIS_UNIFORM, InputVldType::VLD_UNIFORM_MIN_MAX, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {stateless_truncated_normal_,
      {DistributionType::DIS_TRUNCATED_NORMAL, InputVldType::VLD_NORMAL_MEAN_STDDEV, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {stateless_normal_,
      {DistributionType::DIS_NORMAL, InputVldType::VLD_NORMAL_MEAN_STDDEV, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {stateless_bitmask_,
      {DistributionType::DIS_BITMASK, InputVldType::VLD_BITMASK_DPT_RATIO, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
      {stateless_uniform_,
      {DistributionType::DIS_UNIFORM, InputVldType::VLD_UNIFORM_MIN_MAX, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR, DsaValueType::DSA_DATA_ADDR,
      DsaValueType::DSA_DATA_ADDR}},
  };

  const std::map<std::string, vector<uint32_t>> dsa_op_flags_idx_ = {
      {truncated_normal_, {0, 1, 2, 3}},
      {normal_, {0, 1, 2, 3}},
      {bitmask_, {0, 1, 2}},
      {uniform_, {0, 1, 2, 3}},
      {stateless_truncated_normal_, {0, 1, 2, 3, 4}},
      {stateless_normal_, {0, 1, 2, 3, 4}},
      {stateless_bitmask_, {0, 1, 2, 3}},
      {stateless_uniform_, {0, 1, 2, 3, 4}},
  };

  const std::map<ge::DataType, uint32_t> dsa_datatype_values_ = {
      {ge::DT_INT32, 0b000}, {ge::DT_INT64, 0b001},   {ge::DT_UINT32, 0b010}, {ge::DT_UINT64, 0b011},
      {ge::DT_BF16, 0b100},  {ge::DT_FLOAT16, 0b101}, {ge::DT_FLOAT, 0b110},
  };
};
}  // namespace fe

#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_DSA_TASK_BUILDER_H_

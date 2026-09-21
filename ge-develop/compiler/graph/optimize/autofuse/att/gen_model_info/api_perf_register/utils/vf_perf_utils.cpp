/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "vf_perf_utils.h"
#include <numeric>
#include "common_utils.h"
#include "api_perf_register/api_perf_factory.h"
namespace att {
namespace {
Expr GetDataTypeSize(const std::string &data_type) {
  constexpr int32_t kDefaultDataTypeSize = 4;
  const auto &iter = kDataTypeSizeMap.find(data_type);
  if (iter == kDataTypeSizeMap.end()) {
    GELOGW("data type %s not support, use default %d byte", data_type.c_str(), kDefaultDataTypeSize);
    return CreateExpr(kDefaultDataTypeSize);
  }
  return iter->second;
}
bool IsLoadStoreVfApi(const std::string &micro_api_type) {
  const std::set<std::string> kLoadStoreVfType{kMoveGmToUb, kMoveUbToGm};
  return kLoadStoreVfType.find(micro_api_type) != kLoadStoreVfType.end();
}

// 获取vf api的基础latency和throughput
ge::Status GetVFNodePerf(const NodePerfInfo &node_info, const uint32_t micro_api_len, Expr &latency, Expr &throughput) {
  GE_ASSERT_SUCCESS(VfPerfUtils::GetVfInstructPerf(node_info.optype, node_info.input_dtype, latency, throughput));
  Expr dim_product = std::accumulate(node_info.dims.begin(), node_info.dims.end(), CreateExpr(1),
                                     [](const Expr &a, const Expr &b) { return a * b; });
  // 简化计算，后续进一步考虑每个op的latency和throughput的掩盖
  Expr shape_size = dim_product * GetDataTypeSize(node_info.input_dtype);
  uint64_t shape_size_value = 0U;
  Expr api_count = ge::sym::Ceiling(shape_size / CreateExpr(micro_api_len));
  api_count = api_count.Simplify();
  throughput = throughput * api_count;
  throughput = throughput.Simplify();
  GELOGD("Got node %s input %s reg base latency %s, throughput %s, api_count %s, shape_size_value %lu",
         node_info.optype.c_str(), node_info.input_dtype.c_str(), latency.Serialize().get(),
         throughput.Serialize().get(), api_count.Serialize().get(), shape_size_value);
  return ge::SUCCESS;
}

const PerfParamTable *GetParamPerfTable() {
  const auto default_impl = ascgen_utils::GetAscIrAttImpl(kDefaultApi);
  GE_ASSERT_NOTNULL(default_impl);
  const auto api_name = ge::PtrToPtr<void, ge::char_t>(default_impl->GetApiPerf());
  GE_ASSERT_NOTNULL(api_name);
  auto api_perf = ApiPerfFactory::Instance().Create(api_name);
  GE_ASSERT_NOTNULL(api_perf);
  return api_perf->GetPerfParam();
}
}  // namespace

ge::Status VfPerfUtils::GetVfInstructPerf(const std::string &micro_api_type, const std::string &data_type, Expr &latency,
                                        Expr &throughput) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const auto &api_perf_table = param_table->GetVfInstructPerfTable(micro_api_type);
  for (const auto &api_perf : api_perf_table) {
    if (std::count(api_perf.support_data_types.begin(), api_perf.support_data_types.end(), data_type) > 0) {
      latency = CreateExpr(api_perf.latency);
      throughput = CreateExpr(api_perf.throughput);
      break;
    }
  }
  return ge::SUCCESS;
}

ge::Status VfPerfUtils::AddVfInstructPerf(const std::string &vf_instruct_type, const std::string &data_type,
                                          Expr &latency, Expr &throughput, Expr repeat_time) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const auto &api_perf_table = param_table->GetVfInstructPerfTable(vf_instruct_type);
  GELOGD("Begin to add perf of vf instruct [%s].", vf_instruct_type.c_str());
  for (const auto &api_perf : api_perf_table) {
    if (std::count(api_perf.support_data_types.begin(), api_perf.support_data_types.end(), data_type) > 0) {
      GELOGD("Found perf of vf instruct [%s]: latency is {%d}, throughput is {%d}, repeat_time is [%s].",
             vf_instruct_type.c_str(), api_perf.latency, api_perf.throughput,
             ge::SymbolicUtils::ToString(repeat_time).c_str());
      latency = ge::sym::Max(CreateExpr(api_perf.latency), latency);
      throughput = throughput + CreateExpr(api_perf.throughput) * repeat_time;
      break;
    }
  }
  return ge::SUCCESS;
}

Expr VfPerfUtils::GetVFHeadCost() {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  return param_table->GetVectorFunctionHeadCost();
}

// Vector Function建模影响因素：
// 1.Micro Api的latency, throughput;
// 2.调用Micro Api的次数
// 3.Vector Function的启动开销
// 4.调用Micro Api的循环轴每次的循环数对应的头开销
// 5.Micro Api的并发度
// 第一版建模简化模型，仅考虑1,2,3，每个op的latency求最大值，throughput求和
// 后续建模考虑4,5
ge::Status VfPerfUtils::GetVectorFunctionPerf(const std::vector<NodePerfInfo> &node_perf_infos, Expr &res) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const uint32_t micro_api_len = param_table->GetMicroApiLen();
  Expr all_micro_api_cost = CreateExpr(0);
  Expr max_latency = CreateExpr(0);
  for (const auto &node_info : node_perf_infos) {
    Expr latency = CreateExpr(0);
    Expr throughput = CreateExpr(0);
    if (IsLoadStoreVfApi(node_info.optype)) {
      // Load Store VF API暂认为非主导因素，待后续支持搬运类算子或Brc算子时再进行建模
      continue;
    }
    // 在vf function内的必须支持vector, 待校验
    GE_ASSERT_SUCCESS(GetVFNodePerf(node_info, micro_api_len, latency, throughput));
    // 每个op的latency相加, throughput求最大值
    all_micro_api_cost = ge::sym::Add(all_micro_api_cost, throughput);
    max_latency = ge::sym::Max(max_latency, latency);
  }
  max_latency = max_latency.Simplify();
  all_micro_api_cost = all_micro_api_cost.Simplify();
  // 加上最大的latency
  res = ge::sym::Add(all_micro_api_cost, max_latency);
  const auto vector_function_head_cost = param_table->GetVectorFunctionHeadCost();
  res = ge::sym::Add(res, vector_function_head_cost);
  res = res.Simplify();
  GELOGD("Got vector function perf %s, vector_function_head_cost %s, max_latency %s, all_micro_api_cost %s",
         res.Serialize().get(), vector_function_head_cost.Serialize().get(), max_latency.Serialize().get(),
         all_micro_api_cost.Serialize().get());
  return ge::SUCCESS;
}
}  // namespace att

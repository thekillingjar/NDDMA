/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_DATA_DEPENDENT_INTERPRETER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_DATA_DEPENDENT_INTERPRETER_H_
#include <cstdint>
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace gert {
class DataDependentInterpreter {
 public:
  // 兼容处理，先合入v2，然后删除V1构造函数
  DataDependentInterpreter(const ge::OpDescPtr &op_desc, const OpImplSpaceRegistryV2Array &space_registry);
  /**
   * 当前数据依赖的标记来源有三个：
   *
   * 1. runtime2.0方式，通过IMPL_OP标记了数据依赖
   * 2. runtime2.0方式，在InferShape时，通过OpDesc::SetOpInferDependes设置了数据依赖
   * 3. UB融合算子，在子图内的边界上出现了数据依赖
   *
   * 本函数综合上述几种数据依赖的标记来源，给出一个最终的数据依赖判定结果。详细展开来说，本函数的策略为：
   *
   * |序号|IMPL_OP标记|RT1.0 Style标记|UB融合算子子图解析|最终结果|备注             |
   * |--|-----|-----|------|------|-----------------------------------------|
   * |1 |true |true | NA   | true |数据依赖、编译时1.0、2.0标记正确             |
   * |2 |true |false| NA   | true |数据依赖、编译时2.0、2.0标记正确             |
   * |3 |false|true | NA   | true |数据依赖、编译时1.0、2.0标记错误，打印Warning |
   * |4 |false|false| NA   | true |非数据依赖、编译时1.0、2.0标记正确           |
   * |5 |true |true | true | true |UB融合算子、数据依赖、恰好正确               |
   * |6 |true |true | false| true |UB融合算子、编译时1.0、原型标记了数据依赖，但是UB融合子图不需要，打印warning  |
   * |7 |true |false| true | true |UB融合算子、编译时2.0、恰好正确             |
   * |8 |true |false| false| true |UB融合算子、编译时2.0、原型标记了数据依赖，但是UB融合子图不需要，打印warning  |
   * |9 |false|true | true | true |UB融合算子、数据依赖、编译时1.0、2.0标记错误，打印Warning                 |
   * |10|false|true | false| true |UB融合算子、编译时1.0、原型标记了数据依赖，但是UB融合子图不需要，1.0与2.0不一致，打印2 warning |
   * |11|false|false| true | true |UB融合算子、编译时2.0、UB子图带来了数据依赖   |
   * |12|false|false| false| false|UB融合算子、没有数据依赖                   |
   *
   * @param is_data_dependent
   * @return
   */
  ge::graphStatus IsDataDependent(const int32_t index, bool &is_data_dependent) const;
  ge::graphStatus IsTilingInputDataDependent(const int32_t index, bool &is_tiling_dependent) const;
  ge::graphStatus IsSupportTilingDependPlacement(const uint32_t placement, bool &is_support) const;
 private:
  ge::graphStatus IsDataDependentByImplOp(const int32_t input_index, bool &is_data_dependent) const;
  ge::graphStatus IsDataDependentByIr(int32_t index, bool &is_data_dependent) const;
  bool GetByIr(bool by_1_0, bool by_2_0, int32_t index_for_log) const;

  ge::graphStatus IsDataDependentByUbGraph(int32_t index, bool &is_data_dependent) const;
  bool GetByIrAndUb(bool by_ir, bool by_ub, int32_t index_for_log) const;

  ge::ComputeGraphPtr GetUbGraph() const;

  const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImplFunctionsV2() const;

 private:
  ge::NodePtr node_{nullptr};
  ge::OpDescPtr op_desc_{nullptr};
  mutable ge::ComputeGraphPtr ub_graph_cache_;
  // 兼容上库处理，先合入v2，然后删除V1构造函数
  OpImplSpaceRegistryV2Array space_registries_v2_;
};
}

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_DATA_DEPENDENT_INTERPRETER_H_

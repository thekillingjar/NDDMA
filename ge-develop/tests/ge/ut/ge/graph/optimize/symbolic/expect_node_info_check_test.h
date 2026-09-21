/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H
#define AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H
#include "depends/symbol/symbolic_shape_frame_test.h"
#include "graph/utils/tensor_adapter.h"
#include "mmpa/mmpa_api.h"

namespace ge {
class ExpectNodeInfo : public ExpectNodeInfoCheckBase {
 public:
  ExpectNodeInfo(std::string node_name,
                 std::vector<Expression> expect_symbol_output_shape,
                 std::set<std::string> expect_guard_infos,
                 std::set<std::string> expect_assert_infos,
                 std::vector<Expression> expect_symbolic_value)
      : ExpectNodeInfoCheckBase(
            std::move(node_name),
            std::move(expect_symbol_output_shape),
            std::move(expect_guard_infos),
            std::move(expect_assert_infos),
            std::move(expect_symbolic_value)){}
  ~ExpectNodeInfo() override = default;
  bool ExpectShapeCheck(const gert::SymbolShape &real_shape) const override;
  bool ExpectGuardInfoCheck(std::vector<SymbolCheckInfo> real_guard) const override;
  bool ExpectAssertInfoCheck(std::vector<SymbolCheckInfo> real_assert) const override;
  bool ExpectSymbolValCheck(const std::vector<ge::Expression> * real_val) const override;
};

Status RunSymbolInferenceTest(const ComputeGraphPtr &cg, const std::vector<ExpectNodeInfo> &node_info_vec,
                              const std::vector<ge::GeTensor> &input_vec);
Status SetNoStorage(ComputeGraphPtr &cg, const std::string &DataName, const DataInfo &di, int64_t idx) {
  auto data_node = cg->FindNode(DataName);
  GE_ASSERT_NOTNULL(data_node);
  auto op_desc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  ge::AttrUtils::SetInt(op_desc, "index", idx);
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(di.format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(di.format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(di.shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(di.shape));
    op_desc->MutableInputDesc(i)->SetDataType(di.dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(di.dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(di.format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(di.format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(di.shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(di.shape));
    op_desc->MutableOutputDesc(i)->SetDataType(di.dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(di.dt);
  }
  return SUCCESS;
}

template <typename T, ge::DataType DT>
GeTensor BuildGeTensor(std::vector<int64_t> dims, std::vector<T> value) {
  ge::Shape shape0({dims});
  ge::TensorDesc td0{shape0, FORMAT_ND, DT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  auto ge_tensor = ge::TensorAdapter::AsGeTensor(tensor0);
  if (!value.empty()) {
    std::vector<T> tensor_data(value);
    ge_tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data.data()), tensor_data.size() * sizeof(T));
  }
  return ge_tensor;
}
void EnableSliceScheduleEnv() {
  int32_t ret = 0;
  MM_SYS_SET_ENV(MM_ENV_AUTOFUSE_FLAGS, "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1, ret);
  (void) ret;
}
void DisableSliceScheduleEnv() {
  int32_t ret = 0;
  MM_SYS_UNSET_ENV(MM_ENV_AUTOFUSE_FLAGS, ret);
  (void) ret;
}

inline GeTensor BuildTensor(const std::vector<int64_t> &shape, const Format format, const DataType data_type) {
  const Shape shape0({shape});
  TensorDesc td0{shape0, format, data_type};
  td0.SetOriginShape(shape0);
  Tensor tensor0{td0};
  return TensorAdapter::AsGeTensor(tensor0);
}
}
#endif  // AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H

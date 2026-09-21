/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_factory_impl.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  return OperatorFactoryImpl::CreateOperator(operator_name, operator_type);
}

Operator OperatorFactory::CreateOperator(const char_t *const operator_name, const char_t *const operator_type) {
  if ((operator_name == nullptr) || (operator_type == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "Create Operator input parameter is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Create Operator input parameter is nullptr.");
    return Operator();
  }
  const std::string op_name = operator_name;
  const std::string op_type = operator_type;
  return OperatorFactoryImpl::CreateOperator(op_name, op_type);
}

graphStatus OperatorFactory::GetOpsTypeList(std::vector<std::string> &all_ops) {
  return OperatorFactoryImpl::GetOpsTypeList(all_ops);
}

graphStatus OperatorFactory::GetOpsTypeList(std::vector<AscendString> &all_ops) {
  std::vector<std::string> all_op_types;
  if (OperatorFactoryImpl::GetOpsTypeList(all_op_types) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Get ops type list failed.");
    GELOGE(GRAPH_FAILED, "[Get][OpsTypeList] failed.");
    return GRAPH_FAILED;
  }
  for (auto &op_type : all_op_types) {
    all_ops.emplace_back(op_type.c_str());
  }
  return GRAPH_SUCCESS;
}

bool OperatorFactory::IsExistOp(const std::string &operator_type) {
  return OperatorFactoryImpl::IsExistOp(operator_type);
}

bool OperatorFactory::IsExistOp(const char_t *const operator_type) {
  if (operator_type == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Operator type is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator type is nullptr.");
    return false;
  }
  const std::string op_type = operator_type;
  return OperatorFactoryImpl::IsExistOp(op_type);
}

OperatorCreatorRegister::OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator) {
  (void)OperatorFactoryImpl::RegisterOperatorCreator(operator_type, op_creator);
}

OperatorCreatorRegister::OperatorCreatorRegister(const char_t *const operator_type, OpCreatorV2 const &op_creator) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator);
}

InferShapeFuncRegister::InferShapeFuncRegister(const std::string &operator_type,
                                               const InferShapeFunc &infer_shape_func) {
  (void)OperatorFactoryImpl::RegisterInferShapeFunc(operator_type, infer_shape_func);
}

InferShapeFuncRegister::InferShapeFuncRegister(const char_t *const operator_type,
                                               const InferShapeFunc &infer_shape_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferShapeFunc(op_type, infer_shape_func);
}

InferFormatFuncRegister::InferFormatFuncRegister(const std::string &operator_type,
                                                 const InferFormatFunc &infer_format_func) {
  (void)OperatorFactoryImpl::RegisterInferFormatFunc(operator_type, infer_format_func);
}

InferFormatFuncRegister::InferFormatFuncRegister(const char_t *const operator_type,
                                                 const InferFormatFunc &infer_format_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferFormatFunc(op_type, infer_format_func);
}

InferValueRangeFuncRegister::InferValueRangeFuncRegister(const char_t *const operator_type,
                                                         const WHEN_CALL when_call,
                                                         const InferValueRangeFunc &infer_value_range_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferValueRangeFunc(op_type, when_call, false, infer_value_range_func);
}

InferValueRangeFuncRegister::InferValueRangeFuncRegister(const char_t *const operator_type) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferValueRangeFunc(op_type);
}

VerifyFuncRegister::VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func) {
  (void)OperatorFactoryImpl::RegisterVerifyFunc(operator_type, verify_func);
}

VerifyFuncRegister::VerifyFuncRegister(const char_t *const operator_type, const VerifyFunc &verify_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterVerifyFunc(op_type, verify_func);
}
}  // namespace ge

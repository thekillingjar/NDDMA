/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_CUSTOM_OP_FACTORY_IMPL_H
#define CANN_GRAPH_ENGINE_CUSTOM_OP_FACTORY_IMPL_H
#include <map>
#include <memory>
#include "graph/custom_op_factory.h"

namespace ge {
class CustomOpFactoryImpl {
public:
  graphStatus RegisterCustomOpCreator(const AscendString &op_type, const BaseOpCreator &op_creator);

  std::unique_ptr<BaseCustomOp> CreateCustomOp(const AscendString &op_type);

  graphStatus GetAllRegisteredOps(std::vector<AscendString> &all_registered_ops);

  bool IsExistOp(const AscendString &op_type);

  static CustomOpFactoryImpl &GetInstance() {
    static CustomOpFactoryImpl instance;
    return instance;
  }

private:
  std::map<AscendString, BaseOpCreator> custom_op_creators_;
  CustomOpFactoryImpl();
};
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_CUSTOM_OP_FACTORY_IMPL_H

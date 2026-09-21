/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_

#include <map>
#include <memory>
#include <functional>
#include "subgraph_adapter.h"

namespace ge {
/**
 * @brief Used to create OpParser
 *
 */
class PARSER_FUNC_VISIBILITY SubgraphAdapterFactory {
public:
  /**
   * @brief Returns the SubgraphAdapterFactory instance
   * @return SubgraphAdapterFactory object
   */
  static SubgraphAdapterFactory* Instance();

  /**
   * @brief Create SubgraphAdapter based on input type
   * @param [in] op_type Op type
   * @return Created SubgraphAdapter
   */
  std::shared_ptr<SubgraphAdapter> CreateSubgraphAdapter(const std::string &op_type);

  ~SubgraphAdapterFactory() = default;
protected:
  /**
   * @brief SubgraphAdapter creation function
   * @return Created SubgraphAdapter
   */
  using CREATOR_FUN = std::function<std::shared_ptr<SubgraphAdapter>(void)>;

  /**
   * @brief Factory instances can only be created automatically, not new methods, so the constructor is not public.
   */
  SubgraphAdapterFactory() {}

  /**
   * @brief Register creation function
   * @param [in] type Op type
   * @param [in] fun OpParser creation function
   */
  void RegisterCreator(const std::string &type, CREATOR_FUN fun);

private:
  std::map<std::string, CREATOR_FUN> subgraph_adapter_creator_map_;  // lint !e1073

  friend class SubgraphAdapterRegisterar;
};

/**
 * @brief For registering Creator functions for different types of subgraph adapter
 *
 */
class PARSER_FUNC_VISIBILITY SubgraphAdapterRegisterar {
public:
  /**
   * @brief Constructor
   * @param [in] op_type      Op type
   * @param [in] fun          Creator function corresponding to Subgrap adapter
   */
  SubgraphAdapterRegisterar(const std::string &op_type, SubgraphAdapterFactory::CREATOR_FUN fun) {
    SubgraphAdapterFactory::Instance()->RegisterCreator(op_type, fun);
  }
  ~SubgraphAdapterRegisterar() {}
};

/**
 * @brief SubgraphAdapter Registration Macro
 * @param [in] op_type      Op type
 * @param [in] clazz        SubgraphAdapter implementation class
 */
#define REGISTER_SUBGRAPH_ADAPTER_CREATOR(op_type, clazz)                         \
  std::shared_ptr<SubgraphAdapter> Creator_##op_type##_Subgraph_Adapter() {     \
    std::shared_ptr<clazz> ptr = nullptr;                                       \
    GE_MAKE_SHARED(ptr = std::make_shared<clazz>(),                             \
    ptr = nullptr);                                                             \
    return std::shared_ptr<SubgraphAdapter>(ptr);                               \
  }                                                                             \
  ge::SubgraphAdapterRegisterar g_##op_type##_Subgraph_Adapter_Creator(op_type, \
                                                                       Creator_##op_type##_Subgraph_Adapter)
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_IR2TF_PARSE_FACTORY_H_
#define AICPU_IR2TF_PARSE_FACTORY_H_

#include "ir2tf_base_parser.h"
#include "graph/compute_graph.h"

namespace aicpu {
using CREATOR_FUN = std::function<std::shared_ptr<Ir2tfBaseParser>(void)>;
class Ir2tfParserFactory {
 public:
  /**
   * @brief instance of ir2tf parser factory
   * @return IRParserFactory instance
   */
  static Ir2tfParserFactory& Instance();

  /**
   * @brief create IR Parser according to opType
   * @param op_type Op Type
   * @return Ir2tfBaseParser
   */
  virtual std::shared_ptr<Ir2tfBaseParser> CreateIRParser(const std::string &op_type);

  /**
   * @brief ParserFactory deconstruct function
   */
  virtual ~Ir2tfParserFactory() = default;

 protected:
  /**
  * @brief construct func of factory, it should not be public工厂实例只能自动创建，不能通过new的方式创建，所以构造函数不对外公开
  */
  Ir2tfParserFactory() = default;

  /**
   * @brief to register parser creator
   * @param op_type Op type
   * @param fun IRParser creator func
   */
  void RegisterCreator(const std::string &op_type, CREATOR_FUN fun);

 private:
  /**
   * @brief every type of OP has a Creator func
   */
  std::map<std::string, CREATOR_FUN> creator_map_;
  friend class IRParserRegister;
};

/**
 * @brief Register creator func for different type of IR
 */
class IRParserRegister {
 public:
  /**
   * @brief Construct func
   * @param op_type type of op
   * @param fun Creator func of IR
   */
  IRParserRegister(const std::string &op_type, CREATOR_FUN fun) {
    Ir2tfParserFactory::Instance().RegisterCreator(op_type, fun);
  }

  ~IRParserRegister() = default;
};

/**
 * @brief IRParser register Macros
 * @param [in] opType      Type of OP
 * @param [in] clazz       Implementation of IRParser
 */
#define REGISTER_IR_PARSER_CREATOR(op_type, clazz)                         \
  std::shared_ptr<Ir2tfBaseParser> Creator_##op_type##_Ir_Parser() {       \
    std::shared_ptr<clazz> ptr = nullptr;                                  \
    try {                                                                  \
      ptr = std::make_shared<clazz>();                                     \
    } catch(...) {                                                         \
      ptr = nullptr;                                                       \
    }                                                                      \
    return std::shared_ptr<Ir2tfBaseParser>(ptr);                          \
  }                                                                        \
  IRParserRegister g_##opType##_IR_Parser_Creator(                         \
    opType, Creator_##opType##_IR_Parser)
}
#endif // AICPU_IR2TF_PARSE_FACTORY_H_

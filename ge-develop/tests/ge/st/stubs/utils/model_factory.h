/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_STUBS_UTILS_MODEL_FACTORY_H_
#define AIR_CXX_TESTS_ST_STUBS_UTILS_MODEL_FACTORY_H_
#include <string>
#include <unordered_map>
#include "ge_running_env/fake_ns.h"
#include "proto/ge_ir.pb.h"
#include "graph/graph.h"

FAKE_NS_BEGIN

class ModelFactory {
 public:
  static const std::string &GenerateModel_1(bool is_dynamic = true, const bool with_fusion = false);
  static const std::string &GenerateModel_2(bool is_dynamic = true, const bool with_fusion = false);
  static const std::string &GenerateModel_switch(bool is_dynamic = true, const bool with_fusion = false);
  static const std::string &GenerateModel_4(bool is_dynamic = true, const bool with_fusion = false);
  static const std::string &GenerateModel_data_to_netoutput(bool is_dynamic = true, const bool with_fusion = false);
  static const std::string &GenerateModel_refdata(bool is_dynamic = true, const bool with_fusion = false);
 private:
  static const std::string &SaveAsModel(const std::string &name, const Graph &graph, const bool with_fusion);
  static std::unordered_map<std::string, std::string> model_names_to_path_;
};

FAKE_NS_END

#endif //AIR_CXX_TESTS_ST_STUBS_UTILS_MODEL_FACTORY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generator_manager.h"
#include "utils.h"
#include <iostream>

namespace ge {
namespace es {

void GeneratorManager::AddGenerator(std::unique_ptr<ICodeGenerator> generator) {
  generators_.push_back(std::move(generator));
}

std::vector<std::string> GeneratorManager::CollectGeneratedOpTypes() const {
  std::vector<std::string> op_types;
  if (!generators_.empty()) {
    const auto &first_generator_contents = generators_[0]->GetPerOpContents();
    op_types.reserve(first_generator_contents.size());
    for (const auto &kv : first_generator_contents) {
      op_types.emplace_back(kv.first);
    }
    std::sort(op_types.begin(), op_types.end());
  }
  return op_types;
}

std::vector<std::string> GeneratorManager::CollectGeneratedUtilNames() const {
  return std::vector<std::string>{"log"};
}

void GeneratorManager::ResetAllGeneratorsOnFailure(const OpDescPtr &op) {
  for (auto &generator : generators_) {
    generator->ResetWhenGenFailed(op);
  }
}

void GeneratorManager::HandleGenerationException(
    const std::exception &e, const OpDescPtr &op,
    std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops) {
  ResetAllGeneratorsOnFailure(op);
  unsupported_reasons_to_ops[e.what()].push_back(op->GetType());
}

void GeneratorManager::GenAllOps(const std::vector<OpDescPtr> &all_ops,
                                 std::unordered_map<std::string, std::vector<std::string>> &unsupported_reasons_to_ops,
                                 std::vector<std::string> &exclude_ops,
                                 size_t &supported_num) {
  for (const auto &op : all_ops) {
    if (IsOpSkip(op->GetType())) {
      continue;
    }

    if (IsOpExclude(op->GetType(), exclude_ops)) {
      continue;
    }

    const char *reason = nullptr;
    if (!IsOpSupport(op, &reason)) {
      unsupported_reasons_to_ops[reason].push_back(op->GetType());
      continue;
    }

    try {
      for (auto &generator : generators_) {
        generator->GenOp(op);
      }
      ++supported_num;
    } catch (const NotSupportException &e) {
      HandleGenerationException(e, op, unsupported_reasons_to_ops);
    } catch (const std::runtime_error &e) {
      HandleGenerationException(e, op, unsupported_reasons_to_ops);
    }
  }
}

void GeneratorManager::GenAggregateFiles(const std::string &output_dir) {
  auto op_types = CollectGeneratedOpTypes();

  // 为所有生成器生成聚合文件
  for (auto &generator : generators_) {
    generator->GenAggregateIncludes(op_types);
  }

  // 写入聚合文件
  for (auto &generator : generators_) {
    const std::string file_path = output_dir + generator->GetAggregateFileName();
    WriteOut(file_path.c_str(), generator->GetAggregateContentStream());
    std::cout << "Generated " << generator->GetGeneratorName() << " aggregate file: " << file_path << std::endl;
  }
}

void GeneratorManager::GenPerOpFiles(const std::string &output_dir) {
  auto op_types = CollectGeneratedOpTypes();

  for (auto &generator : generators_) {
    generator->GenPerOpFiles(output_dir, op_types);
  }
}

void GeneratorManager::GenAggregateHeaders() {
  for (auto &generator : generators_) {
    generator->GenAggregateHeader();
  }
}

void GeneratorManager::SetUtilsGenerator(std::unique_ptr<UtilsGenerator> generator) {
  utils_generator_ = std::move(generator);
}

void GeneratorManager::GenUtils(const std::string &output_dir) {
  auto util_names = CollectGeneratedUtilNames();
  for (auto &util_name : util_names) {
    utils_generator_->GenUtils(util_name);
  }
  utils_generator_->GenPerUtilFiles(output_dir, util_names);
}
}  // namespace es
}  // namespace ge
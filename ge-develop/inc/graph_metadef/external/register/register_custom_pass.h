/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_REGISTER_PASS_H_
#define INC_EXTERNAL_REGISTER_REGISTER_PASS_H_

#include <functional>
#include <memory>
#include <string>

#include "graph/graph.h"
#include "external/ge_common/ge_api_error_codes.h"
#include "register/register_types.h"

namespace ge {
class PassRegistrationDataImpl;
class CustomPassContext;
class CustomPassContextImpl;
class StreamPassContext;
class StreamPassContextImpl;
using ConstGraphPtr = std::shared_ptr<const Graph>;
using CustomPassFunc = std::function<Status(ge::GraphPtr &, CustomPassContext &)>;
using CustomAllocateStreamPassFunc = std::function<Status(const ConstGraphPtr &, StreamPassContext &)>;
constexpr int64_t INVALID_STREAM_ID = -1;

/**
 * 自定义pass执行阶段，若需扩展，请在kInvalid之前添加
 */
enum class CustomPassStage : uint32_t {
  kBeforeInferShape = 0,
  kAfterInferShape = 1,
  kAfterAssignLogicStream = 2, // only support CustomAllocateStreamPassFunc in this stage
  kAfterBuiltinFusionPass = 3,
  kAfterOriginGraphOptimize = 4,
  kCompatibleInherited = 5,
  kInvalid
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY PassRegistrationData {
 public:
  PassRegistrationData() = default;
  ~PassRegistrationData() = default;

  PassRegistrationData(std::string pass_name);

  PassRegistrationData &CustomPassFn(const CustomPassFunc &custom_pass_fn);

  std::string GetPassName() const;

  CustomPassFunc GetCustomPassFn() const;

  PassRegistrationData &Stage(const CustomPassStage stage);

  CustomPassStage GetStage() const;

  PassRegistrationData &CustomAllocateStreamPassFn(const CustomAllocateStreamPassFunc &allocate_stream_pass_fn);

  CustomAllocateStreamPassFunc GetCustomAllocateStreamPass() const;

 private:
  std::shared_ptr<PassRegistrationDataImpl> impl_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY PassReceiver {
 public:
  PassReceiver(PassRegistrationData &reg_data);
  ~PassReceiver() = default;
};

class CustomPassContext {
 public:
  CustomPassContext();
  virtual ~CustomPassContext();

  void SetErrorMessage(const AscendString &error_message);

  void SetPassName(const AscendString &pass_name);

  AscendString GetErrorMessage() const;

  AscendString GetPassName() const;

  /**
   * 通过option的key，从上下文中获取option的值
   * 若option key不存在，返回失败
   * @param option_key
   * @param option_value 出参
   * @return graphStatus
   */
  graphStatus GetOptionValue(const AscendString &option_key, AscendString &option_value) const;

 private:
  std::unique_ptr<CustomPassContextImpl> impl_;
};

class StreamPassContext : public CustomPassContext {
public:
 explicit StreamPassContext(int64_t current_max_stream_id);

 ~StreamPassContext() override = default;

 graphStatus SetStreamId(const GNode &node, int64_t stream_id);

 int64_t GetStreamId(const GNode &node) const;

 int64_t AllocateNextStreamId();

 int64_t GetCurrMaxStreamId() const;

private:
 std::unique_ptr<StreamPassContextImpl> impl_;
};
}  // namespace ge

#define REGISTER_CUSTOM_PASS(name) REGISTER_CUSTOM_PASS_UNIQ_HELPER(__COUNTER__, (name))
#define REGISTER_CUSTOM_PASS_UNIQ_HELPER(ctr, name) REGISTER_CUSTOM_PASS_UNIQ(ctr, (name))
#define REGISTER_CUSTOM_PASS_UNIQ(ctr, name)        \
  static ::ge::PassReceiver register_pass##ctr      \
      __attribute__((unused)) =                     \
          ::ge::PassRegistrationData((name))

#endif  // INC_EXTERNAL_REGISTER_REGISTER_PASS_H_

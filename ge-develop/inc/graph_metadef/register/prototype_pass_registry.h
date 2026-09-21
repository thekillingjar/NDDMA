/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_PROTOTYPE_PASS_REGISTRY_H
#define METADEF_PROTOTYPE_PASS_REGISTRY_H

#include <google/protobuf/message.h>

#include <functional>
#include <vector>
#include <memory>

#include "external/ge_common/ge_api_error_codes.h"
#include "graph/types.h"
#include "register/register_error_codes.h"
#include "register/register_fmk_types.h"

namespace ge {
class ProtoTypeBasePass {
 public:
  ProtoTypeBasePass() = default;
  virtual Status Run(google::protobuf::Message *message) = 0;
  virtual ~ProtoTypeBasePass() = default;

 private:
  ProtoTypeBasePass(const ProtoTypeBasePass &) = delete;
  ProtoTypeBasePass &operator=(const ProtoTypeBasePass &) & = delete;
};

class ProtoTypePassRegistry {
 public:
  using CreateFn = std::function<ProtoTypeBasePass *(void)>;
  ~ProtoTypePassRegistry();

  static ProtoTypePassRegistry &GetInstance();

  void RegisterProtoTypePass(const char_t *const pass_name, const CreateFn &create_fn,
                             const domi::FrameworkType fmk_type);

  std::vector<std::pair<std::string, CreateFn>> GetCreateFnByType(const domi::FrameworkType fmk_type) const;

 private:
  ProtoTypePassRegistry();
  class ProtoTypePassRegistryImpl;
  std::unique_ptr<ProtoTypePassRegistryImpl> impl_;
};

class ProtoTypePassRegistrar {
 public:
  ProtoTypePassRegistrar(const char_t *const pass_name, ProtoTypeBasePass *(*const create_fn)(),
                         const domi::FrameworkType fmk_type);
  ~ProtoTypePassRegistrar() = default;
};
}  // namespace ge

#define REGISTER_PROTOTYPE_PASS(pass_name, pass, fmk_type) \
  REGISTER_PROTOTYPE_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass, fmk_type)

#define REGISTER_PROTOTYPE_PASS_UNIQ_HELPER(ctr, pass_name, pass, fmk_type) \
  REGISTER_PROTOTYPE_PASS_UNIQ(ctr, pass_name, pass, fmk_type)

#define REGISTER_PROTOTYPE_PASS_UNIQ(ctr, pass_name, pass, fmk_type)                         \
  static ::ge::ProtoTypePassRegistrar register_prototype_pass##ctr __attribute__((unused)) = \
      ::ge::ProtoTypePassRegistrar(                                                          \
          (pass_name), []()->::ge::ProtoTypeBasePass * { return new (std::nothrow) pass(); }, (fmk_type))

#endif  // METADEF_PROTOTYPE_PASS_REGISTRY_H

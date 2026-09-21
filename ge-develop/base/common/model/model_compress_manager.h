/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_
#define GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_

#include <mutex>
#include "common/model/ge_model.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/any_value.h"

namespace ge {
class ModelCompressManager {
 public:
  static Status Compress(const GeModelPtr &ge_model);
  static Status Decompress(const GeModelPtr &ge_model);
  static Status CpyModelAttrs2Dst(const GeModelPtr &src_ge_model, const GeModelPtr &dst_ge_model);
  static void DeleteModelAttrs(const GeModelPtr &ge_model);

 private:
  static Status ProcessAttrsForOp(const OpDescPtr &op_desc, const bool is_compress);
  static Status ProcessKernelNameAttrsForOp(const OpDescPtr &op_desc);
  static Status ProcessKernelName(const OpDescPtr &op_desc);
  static Status ProcessAtomicKernelName(const OpDescPtr &op_desc);
  static Status ProcessAttrsForTensor(const OpDescPtr &op_desc, const bool is_compress);
  static Status ProcessForTensor(const GeTensorDescPtr &tensor, const bool is_compress);
  static Status AddEnumAttrsToModel(const GeModelPtr &ge_model);
  static Status GetEnumAttrsFromModel(const GeModelPtr &ge_model);
  static bool DeleteUnusedAttrs(const OpDescPtr &op_desc, const string &attr_name);
  static Status EnumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &enum_attr_name,
                          GeAttrValue &enum_attr_value);
  static Status DenumAttrs(const pair<const string, GeAttrValue> &name_to_value, string &attr_name,
                           GeAttrValue &attr_value);
  static void UpdateStatus(const bool is_new_attr, const bool is_string_type);
  static bool CheckNeedCompress(const GeModelPtr &ge_model);
  static void PrintOpInfo(const OpDescPtr &op_desc);
  static void CacheClear();

 private:
  static int64_t om_compress_version_;
  static vector<string> enum_attr_names_;
  static vector<string> enum_attr_values_;
  static vector_bit_t name_use_string_values_;
  static std::mutex mutex_;
};
}  // namespace ge
#endif  // GE_COMMON_MODEL_MODEL_COMPRESS_MANAGER_H_

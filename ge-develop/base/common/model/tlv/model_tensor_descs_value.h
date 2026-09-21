/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODEL_TENSOR_DESCS_VALUE_
#define MODEL_TENSOR_DESCS_VALUE_

#include "base_tlv_block.h"
#include "graph/def_types.h"
#include "framework/common/ge_model_inout_types.h"

namespace ge {
class ModelTensorDesc {
public:
  friend class ModelIntroduction;
  size_t Size();
  bool Serilize(uint8_t ** const addr, size_t &left_size);
private:
  ModelTensorDescBaseInfo base_info;
  std::string name;
  std::vector<int64_t> dims;
  std::vector<int64_t> dimsV2;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
};

class ModelTensorDescsValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override;
  bool Serilize(uint8_t ** const addr, size_t &left_size) override;
  bool NeedSave() override;
  virtual ~ModelTensorDescsValue() = default;
private:
  uint32_t tensor_desc_size = 0U;
  std::vector<ModelTensorDesc> descs;
};
}
#endif

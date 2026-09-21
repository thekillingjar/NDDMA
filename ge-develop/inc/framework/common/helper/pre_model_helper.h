/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_

#include "framework/common/helper/model_helper.h"

namespace ge {
class GE_FUNC_VISIBILITY PreModelHelper : public ModelHelper {
 public:
  PreModelHelper() = default;
  virtual ~PreModelHelper() override = default;
  PreModelHelper(const PreModelHelper &) = default;
  PreModelHelper &operator=(const PreModelHelper &) & = default;

  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                   ModelBufferData &model, const bool is_unknown_shape) override;

 private:
  Status SaveToExeOmModel(const GeModelPtr &ge_model, const std::string &output_file, ModelBufferData &model,
                          const GeRootModelPtr &ge_root_model = nullptr);
  Status SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              const size_t model_index = 0UL);
  Status SavePreModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model) const;
  Status SaveModelDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                       const size_t model_index = 0UL);
  Status SaveTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      const size_t model_index = 0UL);
  Status SaveKernelArgs(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                        const size_t model_index = 0UL);
  Status SaveKernelBin(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                       const size_t model_index = 0UL);
  Status SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0UL) const;

 private:
  void SetKernelArgsInfo(const std::shared_ptr<uint8_t> &buff) {
    kernel_args_info_ = buff;
  }
  void SetTaskDescInfo(const std::shared_ptr<uint8_t> &buff) {
    task_desc_info_ = buff;
  }
  void SetModelDescInfo(const std::shared_ptr<uint8_t> &buff) {
    model_desc_info_ = buff;
  }
  void SetKernelBinInfo(const std::shared_ptr<uint8_t> &buff) {
    kernel_bin_info_ = buff;
  }

  std::shared_ptr<uint8_t> kernel_args_info_ = nullptr;
  std::shared_ptr<uint8_t> task_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> model_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> kernel_bin_info_ = nullptr;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_

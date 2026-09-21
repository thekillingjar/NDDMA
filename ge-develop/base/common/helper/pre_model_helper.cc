/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/pre_model_helper.h"
#include "common/preload/model/pre_davinci_model.h"
#include "common/preload/partition/model_desc_partition.h"
#include "common/preload/partition/model_task_desc_partition.h"
#include "common/preload/partition/model_kernel_args_partition.h"
#include "common/preload/partition/model_kernel_bin_partition.h"
#include "framework/common/ge_types.h"
#include "common/helper/file_saver.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/omg/omg_inner_types.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const string kExeOmPreffix = ".exeom";
}
Status PreModelHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                         ModelBufferData &model, const bool is_unknown_shape) {
  GELOGD("begin to save om root model, output_file:%s.", output_file.c_str());
  GE_ASSERT_NOTNULL(ge_root_model, "[Check][GERootModel] ge_root_model is nullptr");

  auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!(name_to_ge_model.empty()), "[Get][SubModel] ge_root_model has no sub model");
  GE_ASSERT_TRUE(!(output_file.empty()), "[Save][Model] GraphBuilder SaveModel received invalid file name prefix");
  if (is_unknown_shape) {
    GELOGE(FAILED, "[Save][Model] Unknown shape not support.");
    return PARAM_INVALID;
  }

  auto &model_root = name_to_ge_model.begin()->second;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(model_root.get()), ATTR_MODEL_ATC_CMDLINE, domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);

  // remove suffix
  const string suffix = ".om";
  const auto pos = output_file.rfind(suffix);
  GE_ASSERT_TRUE(!(pos == std::string::npos), "[Save][Model] GraphBuilder SaveModel received invalid file name prefix");
  string file_without_suffix = output_file;
  (void)file_without_suffix.erase(pos, suffix.length());

  const string file_exeom = file_without_suffix + kExeOmPreffix;
  GE_CHK_STATUS_RET(SaveToExeOmModel(model_root, file_exeom, model), "[Call][SaveToExeOmModel] failed.");
  GELOGD("success save om root model, output_file:%s.", output_file.c_str());
  return SUCCESS;
}

Status PreModelHelper::SaveToExeOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                                        ModelBufferData &model, const GeRootModelPtr &ge_root_model) {
  GELOGD("begin to save exeom model");
  (void)ge_root_model;
  GE_ASSERT_NOTNULL(ge_model, "Ge_model is nullptr");

  if (output_file.empty()) {
    GELOGE(FAILED,
           "[Save][Model]GraphBuilder SaveModel received invalid file name prefix, "
           "model %s",
           ge_model->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphBuilder SaveModel received invalid file name prefix, "
                      "model %s",
                      ge_model->GetName().c_str());
    return FAILED;
  }

  const std::unique_ptr<PreDavinciModel> pre_davinci_model = ge::MakeUnique<PreDavinciModel>();
  GE_ASSERT_NOTNULL(pre_davinci_model, "New PreDavinciModel fail");
  pre_davinci_model->Assign(ge_model);
  auto ret = pre_davinci_model->Init();
  if (ret != SUCCESS) {
    return ret;
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_ATC_CMDLINE, domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);
  std::string cur_version;
  ret = GetOppVersion(cur_version);
  if ((ret != SUCCESS) || (!ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_OPP_VERSION, cur_version))) {
    GELOGW("Ge model set opp version failed!");
  }

  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_ASSERT_NOTNULL(om_file_save_helper, "New OmFileSaveHelper fail");

  GE_ASSERT_SUCCESS(SaveAllModelPartiton(om_file_save_helper, ge_model), "[Save][AllModelPartition]Failed, model %s",
                    ge_model->GetName().c_str());

  GELOGD("begin to save exeom header part");
  GE_ASSERT_SUCCESS(SavePreModelHeader(om_file_save_helper, ge_model), "[Save][SavePreModelHeader]Failed, model %s",
                    ge_model->GetName().c_str());

  GELOGD("begin to save exeom file, output_file:%s.", output_file.c_str());
  GE_ASSERT_SUCCESS(om_file_save_helper->SaveModel(output_file.c_str(), model, true),
                    "[Save][Model]Failed, model %s, output file %s", ge_model->GetName().c_str(), output_file.c_str());
  return SUCCESS;
}

Status PreModelHelper::SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                            const GeModelPtr &ge_model, const size_t model_index) {
  GE_ASSERT_SUCCESS(EnsureKernelBuilt(ge_model), "ensure kernel built failed, model=%s.", ge_model->GetName().c_str());

  GE_ASSERT_SUCCESS(SaveModelIntroduction(om_file_save_helper, ge_model),
                    "[Save][SaveModelIntroduction]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  GE_ASSERT_SUCCESS(SaveModelDesc(om_file_save_helper, ge_model),
                    "[Save][SaveModelDesc]Failed, model %s, model index %zu", ge_model->GetName().c_str(), model_index);

  GE_ASSERT_SUCCESS(SaveTaskDesc(om_file_save_helper, ge_model),
                    "[Save][SaveTaskDesc]Failed, model %s, model index %zu", ge_model->GetName().c_str(), model_index);

  GE_ASSERT_SUCCESS(SaveKernelArgs(om_file_save_helper, ge_model),
                    "[Save][SaveKernelArgs]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  GE_ASSERT_SUCCESS(SaveModelWeights(om_file_save_helper, ge_model),
                    "[Save][SaveKernelArgs]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  GE_ASSERT_SUCCESS(SaveKernelBin(om_file_save_helper, ge_model),
                    "[Save][SaveKernelBin]Failed, model %s, model index %zu", ge_model->GetName().c_str(), model_index);

  GE_ASSERT_SUCCESS(SaveModelCustAICPU(om_file_save_helper, ge_model),
                    "[Save][SaveModelCustAICPU]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  return SUCCESS;
}

Status PreModelHelper::SaveModelDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     const size_t model_index) {
  const std::unique_ptr<ModelDescPartition> model_desc = ge::MakeUnique<ModelDescPartition>();
  GE_ASSERT_NOTNULL(model_desc, "New ModelDescPartition fail");

  GE_ASSERT_SUCCESS(model_desc->Init(ge_model), "ModelDescPartition Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = model_desc->Data();
  SetModelDescInfo(buff);
  GELOGD("PRE_MODEL_DESC size is %zu", model_desc->DataSize());
  if (model_desc->DataSize() > 0UL) {
    GE_CHK_STATUS_RET(
        SaveModelPartition(om_file_save_helper, ModelPartitionType::PRE_MODEL_DESC, model_desc->Data().get(),
                           static_cast<uint64_t>(model_desc->DataSize()), model_index),
        "Add model desc partition failed");
  }
  return SUCCESS;
}

Status PreModelHelper::SaveTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                    const size_t model_index) {
  const std::unique_ptr<ModelTaskDescPartition> task_desc = ge::MakeUnique<ModelTaskDescPartition>();
  GE_ASSERT_NOTNULL(task_desc, "New ModelTaskDescPartition fail");

  GE_ASSERT_SUCCESS(task_desc->Init(ge_model), "ModelTaskDescPartition Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = task_desc->Data();
  SetTaskDescInfo(buff);
  GELOGD("PRE_MODEL_SQE size is %zu", task_desc->DataSize());
  if (task_desc->DataSize() > 0UL) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper, ModelPartitionType::PRE_MODEL_SQE, buff.get(),
                                         static_cast<uint64_t>(task_desc->DataSize()), model_index),
                      "Add task desc partition failed");
  }
  return SUCCESS;
}

Status PreModelHelper::SaveKernelArgs(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                      const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<ModelKernelArgsPartition> model_kernel_args = ge::MakeUnique<ModelKernelArgsPartition>();
  GE_ASSERT_NOTNULL(model_kernel_args, "New ModelKernelArgsPartition fail");

  GE_ASSERT_SUCCESS(model_kernel_args->Init(ge_model),
                    "ModelKernelArgsPartition Init Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  std::shared_ptr<uint8_t> buff = model_kernel_args->Data();
  SetKernelArgsInfo(buff);
  GELOGD("PRE_KERNEL_ARGS size is %zu", model_kernel_args->DataSize());
  if (model_kernel_args->DataSize() > 0UL) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper, ModelPartitionType::PRE_KERNEL_ARGS, buff.get(),
                                         static_cast<uint64_t>(model_kernel_args->DataSize()), model_index),
                      "Add model kernel args partition failed");
  }
  return SUCCESS;
}

Status PreModelHelper::SaveKernelBin(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                                     const size_t model_index) {
  const std::unique_ptr<ModelKernelBinPartition> kernel_bin = ge::MakeUnique<ModelKernelBinPartition>();
  GE_ASSERT_NOTNULL(kernel_bin, "New KernelBinPartition fail");

  GE_ASSERT_SUCCESS(kernel_bin->Init(ge_model), "ModelKernelBinPartition Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = kernel_bin->Data();
  SetKernelBinInfo(buff);
  GELOGD("KERNEL_BIN_INFO size is %zu", kernel_bin->DataSize());
  if (kernel_bin->DataSize() > 0UL) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS, buff.get(),
                                         static_cast<uint64_t>(kernel_bin->DataSize()), model_index),
                      "Add kernel bin partition failed");
  }
  return SUCCESS;
}

Status PreModelHelper::SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                          const GeModelPtr &ge_model, const size_t model_index) const {
  const CustAICPUKernelStore cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  GELOGD("cust aicpu kernels size is %zu", cust_aicpu_kernel_store.PreDataSize());
  if (cust_aicpu_kernel_store.PreDataSize() > 0UL) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper, ModelPartitionType::CUST_AICPU_KERNELS,
                                         ge_model->GetCustAICPUKernelStore().PreData(),
                                         cust_aicpu_kernel_store.PreDataSize(), model_index),
                      "Add cust aicpu kernel partition failed");
  }
  return SUCCESS;
}

Status PreModelHelper::SavePreModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                          const GeModelPtr &ge_model) const {
  GE_ASSERT_SUCCESS(SaveModelHeader(om_file_save_helper, ge_model, 1U, is_so_store_), "save model header fail");

  ModelFileHeader &model_header = om_file_save_helper->GetModelFileHeader();
  model_header.modeltype = static_cast<uint8_t>(MODEL_TYPE_LITE_MODEL);
  return SUCCESS;
}

REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_LITE, PreModelHelper);
}  // namespace ge

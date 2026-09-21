/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/nano_model_save_helper.h"
#include "framework/common/ge_types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "common/helper/file_saver.h"
#include "common/preload/dbg/nano_dbg_data.h"
#include "common/preload/model/nano_davinci_model.h"
#include "common/preload/partition/model_desc_partition.h"
#include "common/preload/partition/nano_task_desc_partition.h"
#include "common/preload/partition/model_desc_extend_partition.h"
#include "base/err_msg.h"

namespace {
  constexpr uint32_t KAlignBytes8 = 8U;
}

namespace ge {
Status NanoModelSaveHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                              ModelBufferData &model, const bool is_unknown_shape) {
  GE_ASSERT_NOTNULL(ge_root_model, "[Check][GERootModel] ge_root_model is nullptr");

  auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!(name_to_ge_model.empty()), "[Get][SubModel] ge_root_model has no sub model");
  GE_ASSERT_TRUE(!(output_file.empty()), "[Save][Model] GraphBuilder SaveModel received invalid file name prefix");

  if (is_unknown_shape) {
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10055", std::vector<const char *>({"reason"}),
        std::vector<const char *>(
            {"Dynamic shape is not supported when the ATC tool is used for model conversion on a nano chip"}));
    GELOGE(FAILED, "[Save][Model] Unknown shape not support.");
    return PARAM_INVALID;
  }

  auto &model_root = name_to_ge_model.begin()->second;
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(*(model_root.get()), ATTR_MODEL_ATC_CMDLINE, domi::GetContext().atc_cmdline),
                   GELOGE(FAILED, "SetStr for atc_cmdline failed.");
                   return FAILED);

  // remove suffix
  const string suffix = ".exeom";
  const auto pos = output_file.rfind(suffix);
  GE_ASSERT_TRUE(!(pos == std::string::npos), "[Save][Model] GraphBuilder SaveModel received invalid file name prefix");
  string file_without_suffix = output_file;
  (void)file_without_suffix.erase(pos, suffix.length());
  GELOGD("save nano dbg file and exeom, output_file_without_suffix:%s.", file_without_suffix.c_str());

  const string file_exeom = file_without_suffix + kFilePreffix;
  GE_CHK_STATUS_RET(SaveToExeOmModel(model_root, file_exeom, model), "[Call][SaveToOmModel] failed.");

  const string file_dbg = file_without_suffix + kDebugPreffix;
  GE_CHK_STATUS_RET(SaveToDbg(model_root, file_dbg), "[Call][SaveToDbg] failed.");

  return SUCCESS;
}

Status NanoModelSaveHelper::SaveToExeOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                                             ModelBufferData &model) {
  GELOGD("begin to save exeom model, output_file:%s", output_file.c_str());
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

  const std::unique_ptr<NanoDavinciModel> nano_davinci_model = ge::MakeUnique<NanoDavinciModel>();
  GE_ASSERT_NOTNULL(nano_davinci_model, "New NanoDavinciModel fail");
  std::shared_ptr<OmFileSaveHelper> om_file_save_helper = ge::MakeShared<OmFileSaveHelper>();
  GE_CHECK_NOTNULL(om_file_save_helper);

  nano_davinci_model->Assign(ge_model);
  auto ret = nano_davinci_model->Init();
  GE_ASSERT_SUCCESS(ret);
  search_ids_ = PreModelPartitionUtils::GetInstance().GetZeroCopyTable();

  ret = SaveAllModelPartiton(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][AllModelPartition]Failed, model %s, error_code %u", ge_model->GetName().c_str(), ret);

  GELOGD("begin to save nano model header");
  ret = SaveModelHeader(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveModelHeader]Failed, model %s, error_code %u", ge_model->GetName().c_str(), ret);

  GELOGD("begin to save exeom file, output_file:%s.", output_file.c_str());
  ret = om_file_save_helper->SaveModel(output_file.c_str(), model, true, true, KAlignBytes8);
  GE_ASSERT_SUCCESS(ret, "[Save][Model]Failed, model %s, output file %s", ge_model->GetName().c_str(),
                    output_file.c_str());
  GELOGD("success save exeom model, output_file:%s", output_file.c_str());
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                                 const GeModelPtr &ge_model, const size_t model_index) {
  auto ret = SaveModelIntroduction(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][ModelIntroduction]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveModelDesc(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][ModelDesc]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveStaticTaskDesc(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveStaticTaskDesc]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveDynamicTaskDesc(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveDynamicTaskDesc]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveTaskParam(om_file_save_helper, ge_model);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveTaskParam]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveModelWeights(om_file_save_helper, ge_model, model_index);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveModelWeights]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveModelTbeKernel(om_file_save_helper, ge_model, model_index);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveModelTbeKernel]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  ret = SaveModelDescExtend(om_file_save_helper, ge_model, model_index);
  GE_ASSERT_SUCCESS(ret, "[Save][SaveModelDescExtend]Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  return SUCCESS;
}

Status NanoModelSaveHelper::SaveStaticTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                               const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<NanoTaskDescPartition> static_task_desc = ge::MakeUnique<NanoTaskDescPartition>();
  GE_ASSERT_NOTNULL(static_task_desc, "[Create][NanoTaskDescPartition]Static failed, it is nullptr, model %s",
                    ge_model->GetName().c_str());

  Status ret = static_task_desc->Init(ge_model, HWTS_STATIC_TASK_DESC);
  GE_ASSERT_SUCCESS(ret, "NanoTaskDescPartition Static Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = static_task_desc->Data();
  SetStaticTaskInfo(buff);
  GELOGD("STATIC_TASK_DESC size is %zu", static_task_desc->DataSize());
  if (static_task_desc->DataSize() > 0UL) {
    ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::STATIC_TASK_DESC, buff.get(),
                             static_cast<uint64_t>(static_task_desc->DataSize()), model_index);
    GE_ASSERT_SUCCESS(ret, "Add static model desc partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveDynamicTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                                const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<NanoTaskDescPartition> dynamic_task_desc = ge::MakeUnique<NanoTaskDescPartition>();
  GE_ASSERT_NOTNULL(dynamic_task_desc, "[Create][NanoTaskDescPartition]Dynamic failed, it is nullptr, model %s",
                    ge_model->GetName().c_str());

  Status ret = dynamic_task_desc->Init(ge_model, HWTS_DYNAMIC_TASK_DESC);
  GE_ASSERT_SUCCESS(ret, "NanoTaskDescPartition Dynamic Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = dynamic_task_desc->Data();
  SetDynamicTaskInfo(buff);
  GELOGD("DYNAMIC_TASK_DESC size is %zu", dynamic_task_desc->DataSize());
  if (dynamic_task_desc->DataSize() > 0UL) {
    ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::DYNAMIC_TASK_DESC, buff.get(),
                             static_cast<uint64_t>(dynamic_task_desc->DataSize()), model_index);
    GE_ASSERT_SUCCESS(ret, "Add dynamic model desc partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveTaskParam(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                          const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<NanoTaskDescPartition> task_param = ge::MakeUnique<NanoTaskDescPartition>();
  GE_ASSERT_NOTNULL(task_param, "[Create][NanoTaskDescPartition]task_param failed, it is nullptr, model %s",
                    ge_model->GetName().c_str());

  Status ret = task_param->Init(ge_model, PARAM_TASK_INFO_DESC);
  GE_ASSERT_SUCCESS(ret, "NanoTaskDescPartition Taskparam Init Failed, model %s, model index %zu",
                    ge_model->GetName().c_str(), model_index);

  std::shared_ptr<uint8_t> buff = task_param->Data();
  SetTaskParamInfo(buff);
  GELOGD("TASK_PARAM size is %zu", task_param->DataSize());
  if (task_param->DataSize() > 0UL) {
    ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::TASK_PARAM, buff.get(),
                             static_cast<uint64_t>(task_param->DataSize()), model_index);
    GE_ASSERT_SUCCESS(ret, "Add task param model desc partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveModelWeights(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                             const GeModelPtr &ge_model, const size_t model_index) const {
  GELOGD("WEIGHTS_DATA size is %zu, %p", ge_model->GetWeightSize(), ge_model->GetWeightData());
  // weight is not necessary
  if (ge_model->GetWeightSize() > 0U) {
    GE_CHK_STATUS_RET(SaveModelPartition(om_file_save_helper, ModelPartitionType::WEIGHTS_DATA,
                                         ge_model->GetWeightData(), ge_model->GetWeightSize(), model_index),
                      "Add weight partition failed");
  }
  uint8_t *data_buf = nullptr;
  uint32_t buf_len = 0U;
  (void)PreModelPartitionUtils::GetInstance().GetNanoModelPartitionData(
      static_cast<uint8_t>(ModelPartitionType::WEIGHTS_DATA), &data_buf, buf_len);
  if ((buf_len > 0U) && (data_buf != nullptr)) {
    const Status ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::WEIGHTS_DATA, data_buf,
                                          static_cast<uint64_t>(buf_len), model_index);
    GE_ASSERT_SUCCESS(ret, "Add nano weight partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveModelDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                          const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<ModelDescPartition> model_desc = ge::MakeUnique<ModelDescPartition>();
  GE_ASSERT_NOTNULL(model_desc, "ModelDescPartition failed, it is nullptr, model %s", ge_model->GetName().c_str());

  Status ret = model_desc->Init(ge_model);
  GE_ASSERT_SUCCESS(ret, "ModelDescPartition Init Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  std::shared_ptr<uint8_t> buff = model_desc->Data();
  SetModelDescInfo(buff);
  GELOGD("PRE_MODEL_DESC size is %zu", model_desc->DataSize());
  if (model_desc->DataSize() > 0UL) {
    ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::PRE_MODEL_DESC, buff.get(),
                             static_cast<uint64_t>(model_desc->DataSize()), model_index);
    GE_ASSERT_SUCCESS(ret, "Add model desc partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveModelTbeKernel(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                               const GeModelPtr &ge_model, const size_t model_index) const {
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  GELOGD("TBE_KERNELS size is %zu", tbe_kernel_store.PreDataSize());
  if (tbe_kernel_store.PreDataSize() > 0UL) {
    const auto ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::TBE_KERNELS,
                                        tbe_kernel_store.PreData(), tbe_kernel_store.PreDataSize(), model_index);
    GE_ASSERT_SUCCESS(ret, "Add tbe kernel partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveModelDescExtend(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                                const GeModelPtr &ge_model, const size_t model_index) {
  const std::unique_ptr<ModelDescExtendPartition> model_desc_extend = ge::MakeUnique<ModelDescExtendPartition>();
  GE_ASSERT_NOTNULL(model_desc_extend, "ModelDescExtendPartition failed, it is nullptr, model %s",
                    ge_model->GetName().c_str());

  Status ret = model_desc_extend->Init(ge_model);
  GE_ASSERT_SUCCESS(ret, "ModelDescExtendPartition Init Failed, model %s, model index %zu", ge_model->GetName().c_str(),
                    model_index);

  GELOGD("model[%s] PRE_MODEL_DESC_EXTEND size is %zu", ge_model->GetName().c_str(), model_desc_extend->DataSize());
  if (model_desc_extend->DataSize() > 0UL) {
    std::shared_ptr<uint8_t> buff = model_desc_extend->Data();
    SetModelExtendInfo(buff);
    ret = SaveModelPartition(om_file_save_helper, ModelPartitionType::PRE_MODEL_DESC_EXTEND, buff.get(),
                             static_cast<uint64_t>(model_desc_extend->DataSize()), model_index);
    GE_ASSERT_SUCCESS(ret, "Add model desc extend partition failed");
  }
  return SUCCESS;
}

Status NanoModelSaveHelper::SaveToDbg(const GeModelPtr &ge_model, const std::string &output_file) const {
  NanoDbgData dbg_data(ge_model, search_ids_);
  GE_CHK_STATUS_RET(dbg_data.Init(), "[Call][Init] failed.");
  return FileSaver::SaveToFile(output_file, dbg_data.GetDbgData(), dbg_data.GetDbgDataSize());
}
REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_NANO, NanoModelSaveHelper);
}  // namespace ge

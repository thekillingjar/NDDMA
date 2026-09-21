/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/partition/model_desc_extend_partition.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
Status ModelDescExtendPartition::SaveModelDescExtendPartitonTlv() {
  GELOGD("save model desc extend partition begin");
  ModelExtendHead *extend_head = PtrToPtr<uint8_t, ModelExtendHead>(des_addr_);  // for update total tlv len

  ModelExtendHead head;
  head.magic = MODEL_EXTEND_DESC_MAGIC_NUM;
  head.len = 0UL;
  GE_ASSERT_SUCCESS((UpdateBuffer(static_cast<void *>(&head), sizeof(ModelExtendHead))), "UpdateBuffer TlvHead failed");
  const uint32_t data_pos = des_size_;  // for update head len, since this pos, the behind is TLV value

  GE_ASSERT_SUCCESS(SaveModelGlobalDataInfoTlv());

  extend_head->len = static_cast<uint64_t>(data_pos) - static_cast<uint64_t>(des_size_);
  GELOGD("save model desc extend partition end, tlv len:%lu", extend_head->len);
  return SUCCESS;
}

void ModelDescExtendPartition::GenModelDescExtendPartitonLen() {
  // L1 tlv
  const auto global_data_table = PreModelPartitionUtils::GetInstance().GetGlobalDataTensorSizes();
  if (global_data_table.size() != 0U) {
    buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
    buff_size_ += static_cast<uint32_t>(sizeof(ModelGlobalDataInfo));
    buff_size_ += static_cast<uint32_t>(sizeof(uint64_t) * global_data_table.size());
  }
  if (buff_size_ > 0U) {
    buff_size_ += static_cast<uint32_t>(sizeof(ModelExtendHead));
  }
}

Status ModelDescExtendPartition::UpdateBuffer(const void *src, const size_t count) {
  if (count == 0U) {
    GELOGD("nothing to memcpy");
    return SUCCESS;
  }
  const auto ret = memcpy_s(des_addr_, static_cast<size_t>(des_size_), src, count);
  GE_ASSERT_EOK(ret, "memcpy_s failed, return: %d", ret);

  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + static_cast<uint32_t>(count)));
  des_size_ -= static_cast<uint32_t>(count);
  return SUCCESS;
}

Status ModelDescExtendPartition::SaveModelGlobalDataInfoTlv() {
  GELOGD("save model desc extend partition global data info tlv begin");
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);  // for update tlv len

  TlvHead head;
  head.type = MODEL_EXTEND_L1_TLV_TYPE_MODEL_GLOBAL_DATA_INFO;
  head.len = 0U;
  GE_ASSERT_SUCCESS((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead))), "UpdateBuffer TlvHead failed");
  const uint32_t data_pos = des_size_;  // for update tlv len, since this pos, the behind is TLV value

  ModelGlobalDataInfo *global_data_ptr =
      PtrToPtr<uint8_t, ModelGlobalDataInfo>(des_addr_);  // for update total_size and num

  ModelGlobalDataInfo global_data;
  global_data.num = 0U;
  global_data.total_size = 0UL;
  GE_ASSERT_SUCCESS((UpdateBuffer(static_cast<void *>(&global_data), sizeof(ModelGlobalDataInfo))),
                    "UpdateBuffer TlvHead failed");

  uint64_t total_size = 0UL;
  const std::vector<uint64_t> global_data_size_vec = PreModelPartitionUtils::GetInstance().GetGlobalDataTensorSizes();
  for (const auto &iter : global_data_size_vec) {
    GE_ASSERT_SUCCESS((UpdateBuffer(PtrToPtr<const uint64_t, const void>(&iter), sizeof(uint64_t))),
                      "UpdateBuffer name_len failed");
    total_size += iter;
  }
  global_data_ptr->total_size = total_size;
  global_data_ptr->num = static_cast<uint32_t>(global_data_size_vec.size());
  tlv_head_ptr->len = data_pos - des_size_;

  GELOGD("save model desc extend partition global data info tlv end, total size:%lu, size num:%u", total_size,
         global_data_size_vec.size());
  return SUCCESS;
}

Status ModelDescExtendPartition::Init(const GeModelPtr &ge_model, const uint8_t type) {
  (void)type;
  (void)ge_model;
  GELOGD("begin to generate model desc extend.");

  // gen buff_size
  GenModelDescExtendPartitonLen();
  if (buff_size_ == 0U) {
    GELOGD("end generate model desc extend, no need model desc extend partition");
    return SUCCESS;
  }
  buff_.reset(new (std::nothrow) uint8_t[buff_size_], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(buff_);
  des_addr_ = buff_.get();
  des_size_ = buff_size_;

  // save model desc extend partition data
  GE_ASSERT_SUCCESS(SaveModelDescExtendPartitonTlv());
  GELOGD("end generate model desc extend, buff_size_:%u", buff_size_);
  return SUCCESS;
}
}  // namespace ge

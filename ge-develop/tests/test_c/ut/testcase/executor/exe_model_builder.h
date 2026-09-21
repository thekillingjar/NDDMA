/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_MODEL_MANAGE_DAVINCI_EXE_MODEL_BUILD_H
#define GE_MODEL_MANAGE_DAVINCI_EXE_MODEL_BUILD_H
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "securec.h"
#include "types.h"

namespace ge {
class ExeModelBuilder {
 public:
  ExeModelBuilder& AddPartition(ModelPartitionType type,
      std::function<void(std::vector<uint8_t> &data)> builder) {
    partitionBuilder_.emplace_back(type, builder);
    return *this;
  }

  ExeModelBuilder& AddPartitionPos(ModelPartitionType type,
      int64_t sizeDeviation, int64_t offsetDeviation = 0) {
    partitionPosDeviation_[type] = {sizeDeviation, offsetDeviation};
    return *this;
  }

  ExeModelBuilder& AddInputDesc(std::string name, size_t size, Format format, DataType dt,
      std::vector<uint64_t> dims1, std::vector<uint64_t> dims2,
      std::vector<std::pair<uint64_t, uint64_t>> shapeRange) {
    inputsDesc_.emplace_back(name, size, format, dt, dims1, dims2, shapeRange);
    return *this;
  }

  ExeModelBuilder& AddOutputDesc(std::string name, size_t size, Format format, DataType dt,
      std::vector<uint64_t> dims1, std::vector<uint64_t> dims2,
      std::vector<std::pair<uint64_t, uint64_t>> shapeRange) {
    outputsDesc_.emplace_back(name, size, format, dt, dims1, dims2, shapeRange);
    return *this;
  }

  void Build() {
    model_.clear();
    AddModelIoDescPartition();
    BuildHead();
    BuildPartitionTable();
    ModelFileHeader *head = reinterpret_cast<ModelFileHeader*>(model_.data());
    head->length = model_.size() - sizeof(ModelFileHeader);
  }
  void BuildMdlErr() {
    model_.clear();
    AddModelIoDescPartition();
    BuildHeadErr();
    BuildPartitionTable();
    ModelFileHeader *head = reinterpret_cast<ModelFileHeader*>(model_.data());
    head->length = model_.size() - sizeof(ModelFileHeader);
  }

  void *ModelData() {
    return model_.data();
  }

  size_t ModelLen() {
    return model_.size();
  }

 private:
  void BuildHead() {
    ModelFileHeader head;
    head.magic = MODEL_FILE_MAGIC_NUM;
    std::copy_n((uint8_t*)(&head), sizeof(ModelFileHeader), std::back_inserter(model_));
  }

  void BuildHeadErr() {
    ModelFileHeader head;
    head.magic = 0x1;
    std::copy_n((uint8_t*)(&head), sizeof(ModelFileHeader), std::back_inserter(model_));
  }

  void BuildPartitionTable() {
    model_.resize(model_.size() + sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * partitionBuilder_.size());
    ModelFileHeader *head = reinterpret_cast<ModelFileHeader*>(model_.data());
    ModelPartitionTable *table = reinterpret_cast<ModelPartitionTable*>(head + 1);
    table->num = partitionBuilder_.size();
    int i = 0;
    size_t basicOffset = model_.size();
    std::for_each(partitionBuilder_.begin(), partitionBuilder_.end(), [&i, this, basicOffset] (auto &it) mutable {
    ModelPartitionMemInfo *memInfo = GetPartition(i);
    memInfo->type = (ModelPartitionType)it.first;
    memInfo->mem_offset = model_.size() - basicOffset;
    it.second(model_);
    memInfo = GetPartition(i);
    memInfo->mem_size = model_.size() - basicOffset - memInfo->mem_offset;

    auto deviation = partitionPosDeviation_.find(it.first);
    if (deviation != partitionPosDeviation_.cend()) {
      memInfo->mem_offset += deviation->second.second;
      memInfo->mem_size += deviation->second.first;
    }
    i++;
    });
  }

  void AddModelIoDescPartition() {
    if (!inputsDesc_.empty() || !outputsDesc_.empty()) {
      AddPartition(MODEL_INOUT_INFO, [this](std::vector<uint8_t> &data) {
        auto fillIoTlv = [this, &data](uint32_t tlvType, const std::vector<IoDescInfo> &ioInfo) {
          uint64_t maxLen = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t);
          for (auto it : ioInfo) {
            maxLen += sizeof(ModelTensorDescBaseInfo);
            maxLen += it.name.size();
            maxLen += it.dims1.size() * sizeof(uint64_t);
            maxLen += it.dims2.size() * sizeof(uint64_t);
            maxLen += it.shapeRange.size() * (sizeof(uint64_t) + sizeof(uint64_t));
          }
          uint64_t offset = data.size();
          data.resize(data.size() + maxLen);
          uint8_t *tmp = data.data() + offset;
          *(uint32_t*)tmp = tlvType;
          tmp += sizeof(uint32_t);
          *(uint32_t*)tmp = maxLen - (sizeof(uint32_t) * 2);
          tmp += sizeof(uint32_t);
          *(uint32_t*)tmp = ioInfo.size();
          tmp += sizeof(uint32_t);
          for (auto it : ioInfo) {
            ModelTensorDescBaseInfo *pBaseInfo = (ModelTensorDescBaseInfo*)tmp;
            pBaseInfo->size = it.size;
            pBaseInfo->format = it.format;
            pBaseInfo->dt = it.dt;
            pBaseInfo->name_len = it.name.size();
            pBaseInfo->dims_len = it.dims1.size() * sizeof(uint64_t);
            pBaseInfo->dimsV2_len = it.dims2.size() * sizeof(uint64_t);
            pBaseInfo->shape_range_len = it.shapeRange.size() * (sizeof(uint64_t) + sizeof(uint64_t));
            tmp += sizeof(ModelTensorDescBaseInfo);
            std::copy_n(it.name.c_str(), pBaseInfo->name_len, tmp);
            tmp += pBaseInfo->name_len;
            std::copy_n((uint8_t*)it.dims1.data(), pBaseInfo->dims_len, tmp);
            tmp += pBaseInfo->dims_len;
            std::copy_n((uint8_t*)it.dims2.data(), pBaseInfo->dimsV2_len, tmp);
            tmp += pBaseInfo->dimsV2_len;
            std::copy_n((uint8_t*)it.shapeRange.data(), pBaseInfo->shape_range_len, tmp);
            tmp += pBaseInfo->shape_range_len;
          }
        };
        if (!inputsDesc_.empty()) {
          fillIoTlv(MODEL_INPUT_DESC, inputsDesc_);
        }
        if (!outputsDesc_.empty()) {
          fillIoTlv(MODEL_OUTPUT_DESC, outputsDesc_);
        }
      });
    }
  }

  ModelPartitionMemInfo* GetPartition(int i) {
    ModelFileHeader *head =  reinterpret_cast<ModelFileHeader*>(model_.data());
    ModelPartitionTable *table = reinterpret_cast<ModelPartitionTable*>(head + 1);
    return &table->partition[i];
  }

  struct IoDescInfo {
    std::string name;
    size_t size;
    Format format;
    DataType dt;
    std::vector<uint64_t> dims1;
    std::vector<uint64_t> dims2;
    std::vector<std::pair<uint64_t, uint64_t>> shapeRange;

    IoDescInfo(std::string name_, size_t size_, Format format_, DataType dt_,
    std::vector<uint64_t> dims1_, std::vector<uint64_t> dims2_,
    std::vector<std::pair<uint64_t, uint64_t>> shapeRange_) :
    name(name_), size(size_), format(format_), dt(dt_), dims1(dims1_), dims2(dims2_),
    shapeRange(shapeRange_) {};
  };

  std::vector<std::pair<ModelPartitionType, std::function<void(std::vector<uint8_t> &data)>>> partitionBuilder_;
  std::map<ModelPartitionType, std::pair<int64_t, int64_t>> partitionPosDeviation_;
  std::vector<IoDescInfo> inputsDesc_;
  std::vector<IoDescInfo> outputsDesc_;
  std::vector<uint8_t> model_;
};
}
#endif

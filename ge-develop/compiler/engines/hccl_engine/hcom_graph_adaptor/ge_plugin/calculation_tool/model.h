/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_SPLIT_MODEL_H
#define AUTO_TUNING_SPLIT_MODEL_H
#include <iostream>
#include <vector>
#include "layers.h"
#include "cluster.h"
#include "hccl/base.h"
#include "hcom_log.h"

constexpr uint32_t DEFAULT_BATCH_SIZE = 32;

class Model {
 public:
  struct SliceMeth {
    std::vector<int> sliceRatio;
    std::vector<float> sliceSize;
  };

  Model(std::vector<uint64_t> &graInfoCost, std::vector<uint64_t> &graInfoSize, int batchs, u64 tensorLimit);

  ~Model();

  std::vector<int> GradientSlicing(Cluster &cluster, const Communication &op, int batchSize = DEFAULT_BATCH_SIZE);

 private:
  float CalculateParamSize(std::vector<Layer> layers, int start = 0, int end = -1);

  float CalculateCost(std::vector<Layer> layers, int start = 0, int end = -1);

  int CalculateNextSlice(std::vector<Layer> &layers, int sliceStart, float costCommunication, float startUpCommTime);

  SliceMeth CalculateTrail(Cluster &cluster, const Communication &op, std::vector<Layer> &layersOriginal,
                           int firstSliceEnd, float &trailCostNew);

  std::vector<Layer> mBps_;
  std::string mName_;
  int mBatchs_;
  float mLimit_;
};

#endif

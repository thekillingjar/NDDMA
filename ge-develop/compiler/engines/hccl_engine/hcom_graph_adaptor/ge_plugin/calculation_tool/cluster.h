/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_CLUSTER_H
#define AUTO_TUNING_CLUSTER_H

#include <iostream>
#include <vector>
#include <typeinfo>
#include <sstream>
#include <fstream>
#include <cmath>
#include "topology.h"
#include "hccl/base.h"
#include "hcom_log.h"

constexpr double EPSILON = 1e-6; /* 双精度比较可接受的误差 */

class Cluster {
 public:
  explicit Cluster(std::string workPath, int gpuNum = 16, float fixedJetter = 0);

  ~Cluster();

  float CalculateCost(const Communication &op, float size = 0, float divisor = 1);

  float CalculateCostWithJetter(const Communication &op, float size, float fixedJetter = 0, float divisor = 1);

  float CalculateStartUpCost(const Communication &op, float size = 0, float divisor = 1);

 private:
  float mFixedJetter_;
  int gpuNum_;
  int mDeviceMemory_;
  std::vector<Topology> mTopoList_;
  std::vector<int> mTopoNumList_;
};

std::string GetModelPara(std::string workPath, std::string str);
void InitDefaultTopoInfo(struct TopoInfo &topoFirst, struct TopoInfo &topoSecond);
void InitModelPara(struct ModelPara &modelPara);
void GetTopoInfoFromFile(std::vector<TopoInfo> &toposInfo);
#endif

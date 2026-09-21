/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_COMMUNICATION_H
#define AUTO_TUNING_COMMUNICATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "hccl/base.h"
#include "hcom_log.h"

constexpr float PARA_FLOAT_ONE = 1.0;
constexpr float PARA_FLOAT_TWO = 2.0;
constexpr float PARA_FLOAT_HALF = 0.5;
constexpr int PARA_INT_TWO = 2;

class Communication {
 public:
  explicit Communication(int chunkNum = 1);

  virtual ~Communication();

  bool CheckParams(int gpuNum, std::string algorithm);

  virtual float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const;

  virtual float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const;

  virtual float CalculateXferBubbles(int gpuNum, std::string algorithm, std::vector<int> &slices) const;

  virtual float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const;

  std::string mName_;
  int mChunkNum_;
};

class Scatter : public Communication {
 public:
  explicit Scatter(int chunkNum = 1);
  ~Scatter() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Gather : public Communication {
 public:
  explicit Gather(int chunkNum = 1);
  ~Gather() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Allgather : public Communication {
 public:
  explicit Allgather(int chunkNum = 1);
  ~Allgather() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Allreduce : public Communication {
 public:
  explicit Allreduce(int chunkNum = 1);
  ~Allreduce() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
  float CalculateXferBubbles(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
  float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Broadcast : public Communication {
 public:
  explicit Broadcast(int chunkNum = 1);
  ~Broadcast() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferBubbles(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Reduce : public Communication {
 public:
  explicit Reduce(int chunkNum = 1);
  ~Reduce() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferBubbles(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Reducescatter : public Communication {
 public:
  explicit Reducescatter(int chunkNum = 1);
  ~Reducescatter() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class Sendrecv : public Communication {
 public:
  explicit Sendrecv(int chunkNum = 1);
  ~Sendrecv() override;
};

class MultirootReduce : public Communication {
 public:
  explicit MultirootReduce(int chunkNum = 1);
  ~MultirootReduce() override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class MultirootBroadcast : public Communication {
 public:
  explicit MultirootBroadcast(int chunkNum = 1);
  ~MultirootBroadcast() override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

class AlltoAll : public Communication {
 public:
  explicit AlltoAll(int chunkNum = 1);
  ~AlltoAll() override;

  float CalculateXferFrequency(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateXferPercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;

  float CalculateComputePercentage(int gpuNum, std::string algorithm, std::vector<int> &slices) const override;
};

int GetMinSlices(std::vector<int> &slices);
int GetMaxSlices(std::vector<int> &slices);

#endif
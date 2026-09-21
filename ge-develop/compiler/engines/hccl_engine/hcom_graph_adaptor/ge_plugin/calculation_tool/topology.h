/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_TOPOLOGY_H
#define AUTO_TUNING_TOPOLOGY_H
#include <iostream>
#include <vector>
#include "communication.h"

constexpr int MAX_BUF_SIZE = 100;
constexpr uint32_t DEFAULT_TOPOFIRST_STACK_NUM = 1;
constexpr uint32_t DEFAULT_TOPOFIRST_GPU_NUM = 8;
constexpr uint32_t DEFAULT_TOPOFIRST_BW_PER_GPU = 14;
constexpr uint32_t DEFAULT_TOPOFIRST_BW_PORT_PER_GPU = 4;
constexpr float DEFAULT_TOPOFIRST_SYNC_COST_PER_PORT = 0.009f;
constexpr float DEFAULT_TOPOFIRST_COST_SYNC = 0.001f;
constexpr uint32_t DEFAULT_TOPOFIRST_DEVICE_MEMORY = 32768;
constexpr uint32_t DEFAULT_TOPOFIRST_BW_COMPUTATION = 0;
constexpr float DEFAULT_TOPOFIRST_FIXED_JITTER = 0.1f;
constexpr uint32_t DEFAULT_TOPOSECOND_STACK_NUM = 8;
constexpr uint32_t DEFAULT_TOPOSECOND_GPU_NUM = 128;
constexpr uint32_t DEFAULT_TOPOSECOND_BW_PER_GPU = 10;
constexpr uint32_t DEFAULT_TOPOSECOND_BW_PORT_PER_GPU = 1;
constexpr uint32_t DEFAULT_TOPOSECOND_SYNC_COST_PER_PORT = 0;
constexpr float DEFAULT_TOPOSECOND_COST_SYNC = 0.015f;
constexpr uint32_t DEFAULT_TOPOSECOND_DEVICE_MEMORY = 32768;
constexpr uint32_t DEFAULT_TOPOSECOND_BW_COMPUTATION = 200;
constexpr uint32_t DEFAULT_TOPOSECOND_FIXED_JITTER = 1;

constexpr int SERVER_NUM = 8;
constexpr uint32_t DEV_NUM_IN_SERVER = 8;
constexpr uint32_t FOUR_DEV = 4;
constexpr uint32_t BW_PER_GPU_FOURTEEN = 14;
constexpr uint32_t BW_PER_GPU_TEN = 10;
constexpr uint32_t BW_PORT_PER_GPU = 4;
constexpr uint32_t DEFAULT_SERVER_PORT_NUM = 8;
constexpr uint32_t DEFAULT_SERVER_NUM = 128;

const float SYNC_COST_FIXED = 0.0001f;

struct TopoInfo {
  std::string algorithm;
  int gpuNum;  // 平面内芯片数量
  std::string topoType;
  float fixedJitter;      // 通信抖动
  float syncCostPerPort;  // 多端口同步开销
  float bwPerGpu;         // 芯片出口有效带宽
  float bwPortPerGpu;     // 芯片出口带宽端口数
  float costSync;         // 单次同步开销
  float bwComputation;    // 计算带宽
  int topoStackNum;       // 平面数量
  int deviceMemory;       // 单芯片显存大小
};

class Topology {
 public:
  Topology();
  explicit Topology(struct TopoInfo &topoInfo);
  virtual ~Topology();
  float CalculateJitter() const;

  virtual float CalculateCost(const Communication &op, float size, std::vector<int> &slices, int divisor) const;

  virtual float CalculateStartUpCost(const Communication &op, float size, std::vector<int> &slices, int divisor) const;

  virtual float CalculateXferCost(const Communication &op, float size, std::vector<int> &slices) const;

  virtual float CalculateComputeCost(const Communication &op, float size, std::vector<int> &slices) const;

  virtual float CalculateSyncCost(const Communication &op, float size, std::vector<int> &slices) const;

  virtual float CalculateBubbleCost(const Communication &op, float size, std::vector<int> &slices) const;

  virtual float CalculateFixedCost(const Communication &op, float size, std::vector<int> &slices) const;

  std::string algorithm_;
  int gpuNum_;
  float bwComputation_;

 private:
  float portNum_;
  float bw_;
  std::string topoType_;
  float syncCostPerXfer_;
  float syncCostPerPort_;
  float syncCostFixed_;
  float fixedJitter_;
};

class Mesh : public Topology {
 public:
  explicit Mesh(struct TopoInfo &topo_info);
  ~Mesh() override;

  float CalculateComputeCost(const Communication &op, float size, std::vector<int> &slices) const override;
};

class Ring : public Topology {
 public:
  explicit Ring(struct TopoInfo &topo_info);
  ~Ring() override;
};

class Torus2D : public Topology {
 public:
  explicit Torus2D(struct TopoInfo &topo_info);
  ~Torus2D() override;
};

class Star : public Topology {
 public:
  explicit Star(struct TopoInfo &topo_info);
  ~Star() override;
};

class Tree : public Topology {
 public:
  explicit Tree(struct TopoInfo &topo_info);
  ~Tree() override;
};

#endif
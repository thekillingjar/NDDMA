/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "communication.h"
using namespace std;

int GetMinSlices(vector<int> &slices) {
  int minSlices = 0;

  for (size_t i = 0; i < slices.size(); i++) {
    if (slices[i] > minSlices) {
      minSlices = slices[i];
    }
  }
  return minSlices;
}

int GetMaxSlices(vector<int> &slices) {
  int maxSlices = 0;

  for (size_t i = 0; i < slices.size(); i++) {
    if (slices[i] > maxSlices) {
      maxSlices = slices[i];
    }
  }
  return maxSlices;
}

Communication::Communication(int chunkNum) {
  this->mChunkNum_ = chunkNum;
  this->mName_ = "Communication";
}

Communication::~Communication() {}

bool Communication::CheckParams(int gpuNum, string algorithm) {
  if (gpuNum > 0) {
    if (algorithm == "ring" || algorithm == "mesh" || algorithm == "mesh-serial" || algorithm == "H-D" ||
        algorithm == "tree" || algorithm == "2D-torus" || algorithm == "direct") {
      return true;
    }
  }
  return false;
}

float Communication::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Communication][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  return PARA_FLOAT_ONE;
}

float Communication::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Communication][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  return PARA_FLOAT_ONE;
}

float Communication::CalculateXferBubbles(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Communication][CalculateXferBubbles] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  return 0.0;
}

float Communication::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Communication][CalculateComputePercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  return 0.0;
}

float Scatter::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Scatter][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float frequency = 0.0;
  if (algorithm == "ring") {
    frequency = (PARA_FLOAT_ONE + (gpuNum / PARA_FLOAT_TWO) - PARA_FLOAT_ONE) * (gpuNum / PARA_FLOAT_TWO) /
                PARA_FLOAT_TWO * PARA_FLOAT_TWO;
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Scatter", algorithm.c_str());
  }

  return frequency;
}

float Scatter::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Scatter][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float percentage = PARA_FLOAT_ONE;
  if (gpuNum == 0) {
    HCCL_ERROR("[Calculate][XferPercentage]In Calculate XferPercentage the value of gpuNum is 0");
    return 0.0;
  }
  if (algorithm == "ring") {
    percentage =
        ((PARA_FLOAT_ONE + (gpuNum / PARA_FLOAT_TWO) - PARA_FLOAT_ONE) * (gpuNum / PARA_FLOAT_TWO) / PARA_FLOAT_TWO) /
        gpuNum * PARA_FLOAT_TWO;
    if (gpuNum % PARA_INT_TWO == 0) {  // lint !e587
      percentage -= (PARA_FLOAT_HALF * ((gpuNum / PARA_FLOAT_TWO) - PARA_FLOAT_ONE)) / gpuNum * PARA_FLOAT_TWO;
    }
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Scatter", algorithm.c_str());
  }

  return percentage;
}

float Gather::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Gather][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float frequency = 0.0;
  if (algorithm == "ring") {
    frequency = (PARA_FLOAT_ONE + (static_cast<float>(gpuNum / PARA_FLOAT_TWO)) - PARA_FLOAT_ONE) *
                (static_cast<float>(gpuNum / PARA_FLOAT_TWO)) / PARA_FLOAT_TWO * PARA_FLOAT_TWO;
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Gather", algorithm.c_str());
  }

  return frequency;
}

float Gather::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Gather][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float percentage = PARA_FLOAT_ONE;
  if (algorithm == "ring") {
    percentage = ((PARA_FLOAT_ONE + (static_cast<float>(gpuNum / PARA_FLOAT_TWO)) - PARA_FLOAT_ONE) *
                  (static_cast<float>(gpuNum / PARA_FLOAT_TWO)) / PARA_FLOAT_TWO) /
                 gpuNum * PARA_FLOAT_TWO;
    if (gpuNum % PARA_INT_TWO == 0) {  // lint !e587
      percentage -= (PARA_FLOAT_HALF * ((gpuNum / PARA_FLOAT_TWO) - PARA_FLOAT_ONE)) /
                    static_cast<float>(gpuNum * PARA_FLOAT_TWO);
    }
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Gather", algorithm.c_str());
  }

  return percentage;
}

float Allgather::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Allgather][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float frequency = 0.0;
  float logValue;
  int temp;

  if (algorithm == "ring") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "H-D") {
    logValue = log(gpuNum) / log(PARA_FLOAT_TWO);
    temp = static_cast<int>(floor(logValue));
    frequency = temp;
    if (logValue - temp > 0) {
      frequency = PARA_FLOAT_ONE + temp;
    }
  } else if (algorithm == "mesh-serial") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "mesh" || algorithm == "direct") {
    frequency = PARA_FLOAT_ONE;
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Allgather", algorithm.c_str());
  }

  return frequency;
}

float Allgather::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = PARA_FLOAT_ONE;
  float logValue;
  int sumSlices = accumulate(slices.begin(), slices.end(), 0);
  int minSlices = GetMinSlices(slices);
  int maxSlices = GetMaxSlices(slices);

  if (algorithm == "ring") {
    percentage = gpuNum - PARA_FLOAT_ONE;
    if (!slices.empty()) {
      percentage = sumSlices - minSlices;
    }
  } else if (algorithm == "H-D") {
    logValue = log(gpuNum) / log(PARA_INT_TWO);
    percentage = pow(PARA_INT_TWO, floor(logValue)) - PARA_FLOAT_ONE;
    if (logValue - floor(logValue) > 0) {
      percentage *= PARA_FLOAT_TWO;
    }
  } else if (algorithm == "mesh") {
    percentage = gpuNum - PARA_FLOAT_ONE;
    if (!slices.empty()) {
      percentage = maxSlices * slices.size();
    }
  } else if (algorithm == "direct") {
    percentage = gpuNum - PARA_FLOAT_ONE;
    if (!slices.empty()) {
      percentage = sumSlices - minSlices;
    }
  } else if (algorithm == "mesh-serial") {
    percentage = gpuNum - PARA_FLOAT_ONE;
    if (!slices.empty()) {
      percentage = sumSlices - minSlices;
    }
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Allgather", algorithm.c_str());
  }

  return percentage;
}

float Allreduce::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  float frequency = 0.0;
  vector<Communication *> opList;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;

  Reducescatter reducescatter(this->mChunkNum_);  // lint !e1542
  Allgather allgather(this->mChunkNum_);          // lint !e1542
  Allreduce allreduce(this->mChunkNum_);          // lint !e1542
  Reduce reduce(this->mChunkNum_);                // lint !e1542
  Broadcast broadcast(this->mChunkNum_);          // lint !e1542

  opList.push_back(&reducescatter);
  opList.push_back(&allgather);

  if (algorithm == "tree") {
    opList.clear();
    opList.push_back(&reduce);
    opList.push_back(&broadcast);
  } else if (algorithm == "2D-torus") {
    opList.clear();
    opList.push_back(&allreduce);
    opList.push_back(&allreduce);
    algorithmImpl = "ring";
  } else {
    if (!slices.empty()) {
      opList.clear();
      opList.push_back(&reduce);
      opList.push_back(&broadcast);
      slicesImpl.clear();
    }
  }

  frequency = 0.0;
  for (size_t i = 0; i < opList.size(); i++) {
    frequency += opList[i]->CalculateXferFrequency(gpuNum, algorithmImpl, slicesImpl);
  }

  return frequency;
}

float Allreduce::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = PARA_FLOAT_ONE;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);
  int sumSlices = accumulate(slices.begin(), slices.end(), 0);
  int maxSlices = GetMaxSlices(slices);
  initialPecentages.push_back(PARA_FLOAT_ONE);
  initialPecentages.push_back(PARA_FLOAT_ONE / gpuNum);
  opList.push_back(&reducescatter);
  opList.push_back(&allgather);

  if (algorithm == "H-D" && (!slices.empty())) {
    int nonZeroSliceNum = 0;
    for (size_t i = 0; i < slices.size(); i++) {
      if (slices[i] != 0) {
        nonZeroSliceNum++;
      }
    }
    opList.clear();
    opList.push_back(&multirootReduce);
    opList.push_back(&multirootBroadcast);
    initialPecentages.clear();
    initialPecentages.push_back(PARA_FLOAT_ONE);
    if (nonZeroSliceNum == 0) {
      nonZeroSliceNum = 1;
    }
    initialPecentages.push_back(PARA_FLOAT_ONE / nonZeroSliceNum);
    slicesImpl = slices;
  } else if (algorithm == "tree") {
    opList.clear();
    opList.push_back(&reduce);
    opList.push_back(&broadcast);
    initialPecentages.clear();
    initialPecentages.push_back(PARA_FLOAT_ONE);
    initialPecentages.push_back(PARA_FLOAT_ONE);
  } else if (algorithm == "2D-torus") {
    opList.clear();
    opList.push_back(&allreduce);
    opList.push_back(&allreduce);
    initialPecentages.clear();
    initialPecentages.push_back(PARA_FLOAT_HALF);
    initialPecentages.push_back(PARA_FLOAT_HALF);
    algorithmImpl = "ring";
  } else {
    if (!slices.empty()) {
      opList.clear();
      opList.push_back(&reduce);
      opList.push_back(&broadcast);
      initialPecentages.clear();
      if (sumSlices == 0) {
        sumSlices = 1;
      }
      initialPecentages.push_back(static_cast<float>(maxSlices) / sumSlices);
      initialPecentages.push_back(static_cast<float>(maxSlices) / sumSlices);
      slicesImpl.clear();
    }
  }

  percentage = 0.0;
  for (size_t i = 0; i < opList.size(); i++) {
    percentage += opList[i]->CalculateXferPercentage(gpuNum, algorithmImpl, slicesImpl) * initialPecentages[i];
  }

  return percentage;
}

float Allreduce::CalculateXferBubbles(int gpuNum, string algorithm, vector<int> &slices) const {
  float bublleNum = PARA_FLOAT_ONE;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);

  initialPecentages.push_back(PARA_FLOAT_ONE);
  initialPecentages.push_back(PARA_FLOAT_ONE / gpuNum);
  opList.push_back(&reducescatter);
  opList.push_back(&allgather);

  if (algorithm == "tree") {
    opList.clear();
    opList.push_back(&reduce);
    opList.push_back(&broadcast);
  } else {
    if (!slices.empty()) {
      opList.clear();
      opList.push_back(&reduce);
      opList.push_back(&broadcast);
      initialPecentages.clear();
      initialPecentages.push_back(1.0);
      initialPecentages.push_back(1.0);
      slicesImpl.clear();
    }
  }

  bublleNum = 0.0;
  for (size_t i = 0; i < opList.size(); i++) {
    bublleNum += opList[i]->CalculateXferBubbles(gpuNum, algorithmImpl, slicesImpl);
  }

  return bublleNum;
}

float Allreduce::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = PARA_FLOAT_ONE;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);
  int sumSlices = accumulate(slices.begin(), slices.end(), 0);
  int maxSlices = GetMaxSlices(slices);

  initialPecentages.push_back(PARA_FLOAT_ONE);
  initialPecentages.push_back(PARA_FLOAT_ONE / gpuNum);
  opList.push_back(&reducescatter);
  opList.push_back(&allgather);

  if (algorithm == "H-D" && (!slices.empty())) {
    int nonZeroSliceNum = 0;
    for (size_t i = 0; i < slices.size(); i++) {
      if (slices[i] > 0) {
        nonZeroSliceNum++;
      }
    }
    opList.clear();
    opList.push_back(&multirootReduce);
    opList.push_back(&multirootBroadcast);
    initialPecentages.clear();
    initialPecentages.push_back(PARA_FLOAT_ONE);
    if (nonZeroSliceNum == 0) {
      nonZeroSliceNum = 1;
    }
    initialPecentages.push_back(PARA_FLOAT_ONE / nonZeroSliceNum);
    slicesImpl = slices;
  } else if (algorithm == "tree") {
    opList.clear();
    opList.push_back(&reduce);
    opList.push_back(&broadcast);
    initialPecentages.clear();
    initialPecentages.push_back(PARA_FLOAT_ONE);
    initialPecentages.push_back(PARA_FLOAT_ONE);
  } else {
    if (!slices.empty()) {
      opList.clear();
      opList.push_back(&reduce);
      opList.push_back(&broadcast);
      initialPecentages.clear();
      if (sumSlices == 0) {
        sumSlices = 1;
      }
      initialPecentages.push_back(static_cast<float>(maxSlices) / sumSlices);
      initialPecentages.push_back(static_cast<float>(maxSlices) / sumSlices);
      slicesImpl.clear();
    }
  }

  percentage = 0.0;
  for (size_t i = 0; i < opList.size(); i++) {
    percentage += opList[i]->CalculateComputePercentage(gpuNum, algorithmImpl, slicesImpl) * initialPecentages[i];
  }

  return percentage;
}

float Broadcast::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Broadcast][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float frequency = 0.0;
  float logValue;
  int temp;

  if (gpuNum == 1) {
    return 0.0;
  }

  if (algorithm == "tree" || algorithm == "ring") {
    frequency = PARA_FLOAT_ONE * this->mChunkNum_;  // ring algorithm default, so does tree algorithm
  }

  if (algorithm == "H-D") {
    logValue = log(gpuNum) / log(PARA_INT_TWO);
    temp = static_cast<int>(floor(logValue));
    frequency = temp;
    if (logValue - temp > 0) {
      frequency = PARA_FLOAT_ONE + temp;
    }
  } else if (algorithm == "mesh-serial") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "mesh" || algorithm == "direct") {
    frequency = PARA_FLOAT_ONE;
  } else {
    HCCL_WARNING("WARNING : unsupported algorithm = %s, op = Broadcast", algorithm.c_str());
  }
  return frequency;
}

float Broadcast::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Broadcast][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float percentage = PARA_FLOAT_ONE;

  if (algorithm == "ring") {
    percentage = PARA_FLOAT_ONE / this->mChunkNum_;
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    percentage = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "tree") {
    percentage = PARA_FLOAT_TWO;
  }
  return percentage;
}

float Broadcast::CalculateXferBubbles(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Broadcast][CalculateXferBubbles] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float bublleNum = PARA_FLOAT_ONE;

  if (gpuNum == 1) {
    return 0.0;
  }
  if (algorithm == "ring") {
    bublleNum = gpuNum - PARA_FLOAT_TWO + (this->mChunkNum_ - PARA_FLOAT_ONE);
  } else if (algorithm == "H-D") {
    bublleNum = 0.0;
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    bublleNum = 0.0;
  } else if (algorithm == "tree") {
    bublleNum = log(gpuNum) / log(PARA_INT_TWO) - PARA_FLOAT_ONE;
    if (gpuNum > PARA_INT_TWO && (gpuNum % PARA_INT_TWO) == 0) {  // lint !e587
      bublleNum -= PARA_FLOAT_HALF;                               // even root only transfer once
    }
  }
  bublleNum += 0.0;
  return bublleNum;
}

float Reduce::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reduce][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float frequency = 0.0;

  if (gpuNum == 1) {
    return 0.0;
  }

  if (algorithm == "ring") {
    frequency = PARA_FLOAT_ONE * this->mChunkNum_;  // ring algorithm default, so does tree algorithm
  } else if (algorithm == "H-D") {
    frequency = log(gpuNum) / log(PARA_INT_TWO);
  } else if (algorithm == "mesh") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "mesh-serial") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "direct") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "tree") {
    frequency = PARA_FLOAT_TWO;
    if (gpuNum == PARA_INT_TWO) {
      frequency = PARA_FLOAT_ONE;
    }
  }

  return frequency;
}

float Reduce::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reduce][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float percentage = 0.0;

  if (algorithm == "ring") {
    percentage = static_cast<float>(PARA_FLOAT_ONE) / this->mChunkNum_;
  } else if (algorithm == "H-D") {
    percentage = log(gpuNum) / log(PARA_INT_TWO);
  } else if (algorithm == "mesh") {
    percentage = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "mesh-serial") {
    percentage = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "direct") {
    percentage = PARA_FLOAT_ONE;
  } else if (algorithm == "tree") {
    percentage = PARA_FLOAT_TWO;
    if (gpuNum == PARA_INT_TWO) {
      percentage = PARA_FLOAT_ONE;
    }
  }
  return percentage;
}

float Reduce::CalculateXferBubbles(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reduce][CalculateXferBubbles] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float bublleNum = 0.0;

  if (gpuNum == 1) {
    return 0.0;
  }
  if (algorithm == "ring") {
    bublleNum = gpuNum - PARA_FLOAT_TWO + (this->mChunkNum_ - PARA_FLOAT_ONE);
  } else if (algorithm == "H-D") {
    bublleNum = 0.0;
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    bublleNum = 0.0;
  } else if (algorithm == "tree") {
    bublleNum = log(gpuNum) / log(PARA_INT_TWO) - PARA_FLOAT_ONE;
    if (gpuNum > PARA_INT_TWO && (gpuNum % PARA_INT_TWO) == 0) {  // lint !e587
      bublleNum -= PARA_FLOAT_HALF;                               // even root only transfer once
    }
  }
  bublleNum += 0.0;
  return bublleNum;
}

float Reduce::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reduce][CalculateComputePercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float percentage = 0.0;

  if (gpuNum == 1) {
    return 0.0;
  }
  if (algorithm == "ring") {
    percentage = static_cast<float>((static_cast<float>(gpuNum) - PARA_FLOAT_ONE) / gpuNum);
  } else if (algorithm == "H-D") {
    percentage = static_cast<float>((gpuNum - PARA_FLOAT_ONE) / gpuNum);
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    percentage = static_cast<float>((gpuNum - PARA_FLOAT_ONE) / gpuNum);
  } else if (algorithm == "tree") {
    percentage = PARA_FLOAT_TWO;
    if (gpuNum == PARA_INT_TWO) {
      percentage = PARA_FLOAT_ONE;
    }
  }

  return percentage;
}

float Reducescatter::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reducescatter][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float frequency = gpuNum - PARA_FLOAT_ONE;
  float logValue;
  int temp;

  if (algorithm == "H-D") {
    logValue = log(gpuNum) / log(PARA_INT_TWO);
    temp = static_cast<int>(floor(logValue));
    frequency = temp;
    if (logValue - temp > 0) {
      frequency = PARA_FLOAT_ONE + temp;
    }
  } else if (algorithm == "mesh" || algorithm == "mesh-serial") {
    frequency = gpuNum - PARA_FLOAT_ONE;
  } else if (algorithm == "direct") {
    frequency = PARA_FLOAT_ONE;
  }

  return frequency;
}

float Reducescatter::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reducescatter][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float percentage = static_cast<float>(static_cast<float>(gpuNum - PARA_FLOAT_ONE) / gpuNum);
  float logValue;
  float gpu_num_new;

  if (algorithm == "H-D") {
    logValue = log(gpuNum) / log(PARA_INT_TWO);
    gpu_num_new = pow(PARA_INT_TWO, floor(logValue));
    percentage = (gpu_num_new - PARA_FLOAT_ONE) / gpu_num_new;
    if (logValue - floor(logValue) > 0) {
      percentage += PARA_FLOAT_ONE;
    }
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    percentage = static_cast<float>((gpuNum - PARA_FLOAT_ONE) / gpuNum);
  }
  return percentage;
}

float Reducescatter::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[Reducescatter][CalculateComputePercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float percentage = static_cast<float>(static_cast<float>(gpuNum - PARA_FLOAT_ONE) / gpuNum);

  if (algorithm == "H-D") {
    percentage = static_cast<float>((static_cast<float>(gpuNum) - PARA_FLOAT_ONE) / gpuNum);
  } else if (algorithm == "mesh" || algorithm == "direct" || algorithm == "mesh-serial") {
    percentage = static_cast<float>((static_cast<float>(gpuNum) - PARA_FLOAT_ONE) / gpuNum);
  }
  return percentage;
}

float MultirootReduce::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = 0;
  float percentage_step;
  float step_num_reduce_scatter;
  float step_num_reduce;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);

  if (algorithm == "H-D") {
    if (slices.empty()) {
      percentage = reduce.CalculateComputePercentage(gpuNum, algorithm, slices);
    } else {
      int nonZeroSliceNum = 0;
      for (size_t i = 0; i < slices.size(); i++) {
        if (slices[i] > 0) {
          nonZeroSliceNum++;
        }
      }
      opList.clear();
      opList.push_back(&reducescatter);
      opList.push_back(&reduce);
      percentage_step = PARA_FLOAT_ONE;
      step_num_reduce_scatter = floor(log(nonZeroSliceNum) / log(PARA_INT_TWO));
      for (int i = 0; i < static_cast<int>(step_num_reduce_scatter); i++) {
        percentage_step /= PARA_FLOAT_TWO;
        percentage += percentage_step;
      }

      step_num_reduce = floor(log(gpuNum) / log(PARA_INT_TWO)) - step_num_reduce_scatter;
      for (int i = 0; i < static_cast<int>(step_num_reduce); i++) {
        percentage += percentage_step;
      }
    }
  }

  return percentage;
}

float MultirootReduce::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = 0;
  float percentage_step;
  float step_num_reduce_scatter;
  float step_num_reduce;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);

  if (algorithm == "H-D") {
    if (slices.empty()) {
      percentage = reduce.CalculateComputePercentage(gpuNum, algorithm, slices);
    } else {
      int nonZeroSliceNum = 0;
      for (size_t i = 0; i < slices.size(); i++) {
        if (slices[i] > 0) {
          nonZeroSliceNum++;
        }
      }
      opList.clear();
      opList.push_back(&reducescatter);
      opList.push_back(&reduce);
      percentage_step = PARA_FLOAT_ONE;
      step_num_reduce_scatter = floor(log(nonZeroSliceNum) / log(PARA_INT_TWO));
      for (int i = 0; i < static_cast<int>(step_num_reduce_scatter); i++) {
        percentage_step /= PARA_FLOAT_TWO;
        percentage += percentage_step;
      }

      step_num_reduce = floor(log(gpuNum) / log(PARA_INT_TWO)) - step_num_reduce_scatter;
      for (int i = 0; i < static_cast<int>(step_num_reduce); i++) {
        percentage += percentage_step;
      }
    }
  }

  return percentage;
}

float MultirootBroadcast::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  float percentage = 0;
  float percentage_step;
  float step_num_all_gather;
  float step_num_broadcast;
  vector<Communication *> opList;
  vector<float> initialPecentages;
  string algorithmImpl = algorithm;
  vector<int> slicesImpl = slices;
  Reducescatter reducescatter(this->mChunkNum_);
  Allgather allgather(this->mChunkNum_);
  Allreduce allreduce(this->mChunkNum_);
  Reduce reduce(this->mChunkNum_);
  Broadcast broadcast(this->mChunkNum_);
  MultirootReduce multirootReduce(this->mChunkNum_);
  MultirootBroadcast multirootBroadcast(this->mChunkNum_);

  if (algorithm == "H-D") {
    if (slices.empty()) {
      percentage = broadcast.CalculateComputePercentage(gpuNum, algorithm, slices);
    } else {
      int nonZeroSliceNum = 0;
      for (size_t i = 0; i < slices.size(); i++) {
        if (slices[i] > 0) {
          nonZeroSliceNum++;
        }
      }
      opList.clear();
      opList.push_back(&allgather);
      opList.push_back(&broadcast);
      percentage_step = PARA_FLOAT_ONE;
      step_num_all_gather = floor(log(nonZeroSliceNum) / log(PARA_INT_TWO));
      for (int i = 0; i < static_cast<int>(step_num_all_gather); i++) {
        percentage_step *= PARA_FLOAT_TWO;
        percentage += percentage_step;
      }

      step_num_broadcast = floor(log(gpuNum) / log(PARA_INT_TWO)) - step_num_all_gather;
      for (int i = 0; i < static_cast<int>(step_num_broadcast); i++) {
        percentage += percentage_step;
      }
    }
  }

  return percentage;
}

float AlltoAll::CalculateXferFrequency(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[AlltoAll][CalculateXferFrequency] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum, algorithm.c_str(),
             slices.size());
  float frequency = 0;

  frequency = (PARA_FLOAT_ONE + (gpuNum - PARA_FLOAT_ONE)) * (gpuNum - PARA_FLOAT_ONE) / PARA_FLOAT_TWO;
  if (algorithm == "H-D") {
    frequency = ((gpuNum / PARA_FLOAT_TWO) - PARA_FLOAT_HALF) / (PARA_FLOAT_ONE - PARA_FLOAT_HALF);
  } else if (algorithm == "mesh" || algorithm == "direct") {
    frequency = PARA_FLOAT_ONE;
  }

  return frequency;
}

float AlltoAll::CalculateXferPercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[AlltoAll][CalculateXferPercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  float percentage = PARA_FLOAT_ONE;
  if (gpuNum == 0) {
    HCCL_WARNING("para error,gpuNum is zero");
    return percentage;
  }

  percentage = (gpuNum - PARA_FLOAT_ONE) / PARA_FLOAT_TWO *
               (PARA_FLOAT_ONE / static_cast<float>(gpuNum) + (static_cast<float>(gpuNum) - PARA_FLOAT_ONE) / gpuNum);
  if (algorithm == "H-D") {
    float sub_block_gpu_num = gpuNum;
    percentage = 0.0;
    float initial_percentage = PARA_FLOAT_ONE;
    while (sub_block_gpu_num > PARA_FLOAT_ONE) {
      percentage += initial_percentage * (PARA_FLOAT_HALF - PARA_FLOAT_HALF / sub_block_gpu_num) /
                    (PARA_FLOAT_ONE - PARA_FLOAT_HALF);
      sub_block_gpu_num /= PARA_FLOAT_TWO;
      initial_percentage /= PARA_FLOAT_TWO;
    }
  } else if (algorithm == "mesh" || algorithm == "direct") {
    percentage = static_cast<float>((gpuNum - PARA_FLOAT_ONE) / gpuNum);
  }

  return percentage;
}

float AlltoAll::CalculateComputePercentage(int gpuNum, string algorithm, vector<int> &slices) const {
  HCCL_DEBUG("[AlltoAll][CalculateComputePercentage] gpuNum[%d], algorithm[%s], slices_size[%u]", gpuNum,
             algorithm.c_str(), slices.size());
  if (gpuNum == 0) {
    HCCL_WARNING("para error,gpuNum is zero");
    return PARA_FLOAT_ONE;
  }
  return PARA_FLOAT_ONE / gpuNum;
}

Scatter::Scatter(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Scatter";
}

Scatter::~Scatter() {}

Gather::Gather(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Gather";
}

Gather::~Gather() {}

Allgather::Allgather(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Gather";
}

Allgather::~Allgather() {}

Allreduce::Allreduce(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Allreduce";
}

Allreduce::~Allreduce() {}

Broadcast::Broadcast(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Broadcast";
}

Broadcast::~Broadcast() {}

Reduce::Reduce(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Reduce";
}

Reduce::~Reduce() {}

Reducescatter::Reducescatter(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Reducescatter";
}

Reducescatter::~Reducescatter() {}

Sendrecv::Sendrecv(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "Sendrecv";
}

Sendrecv::~Sendrecv() {}

MultirootReduce::MultirootReduce(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "MultirootReduce";
}

MultirootReduce::~MultirootReduce() {}

MultirootBroadcast::MultirootBroadcast(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "MultirootBroadcast";
}

MultirootBroadcast::~MultirootBroadcast() {}

AlltoAll::AlltoAll(int chunkNum) : Communication(chunkNum) {
  this->mName_ = "AlltoAll";
}

AlltoAll::~AlltoAll() {}
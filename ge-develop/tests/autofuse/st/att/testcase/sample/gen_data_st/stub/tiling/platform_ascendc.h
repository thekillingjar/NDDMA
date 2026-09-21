/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_ascendc.h
 * \brief
 */

#ifndef SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_PLATFORM_ASCENDC_H
#define SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_PLATFORM_ASCENDC_H

#include <cstdint>
#include <mutex>

namespace fe {
class PlatFormInfos;
}

namespace platform_ascendc {
enum class CoreMemType { L0_A = 0, L0_B = 1, L0_C = 2, L1 = 3, L2 = 4, UB = 5, HBM = 6, RESERVED };

enum class SocVersion {
  ASCEND910 = 0,  // Ascend910A, Ascend910B
  ASCEND910B,     // Ascend910B1~4, Ascend910B2C
  ASCEND310P,     // Ascend310P1, Ascend310P3
  ASCEND310B,     // Ascend310B1, Ascend310B2, Ascend310B3, Ascend310B4
  RESERVED_VERSION = 99999
};

class PlatformAscendC {
 public:
  PlatformAscendC() = delete;
  ~PlatformAscendC() = default;
  explicit PlatformAscendC(fe::PlatFormInfos *platformInfo) : platformInfo_(platformInfo) {}
  /**
   * Get Core Number
   * On Ascend910B MIX model, return AICore number
   * @return core number by core type
   */
  uint32_t GetCoreNum(void) const {
    return 48u;
  }
  /**
   * Get Core Number AiCore
   * @return ai_core_num
   */
  uint32_t GetCoreNumAic(void) const {
    return 24u;
  }
  /**
   * Get Core Number VectorCore
   * @return vector_core_num
   */
  uint32_t GetCoreNumAiv(void) const {
    return 48u;
  }
  /**
   * Get Core Number VectorCore for m200
   * @return vector_core_num if m200, otherwise 0
   */
  uint32_t GetCoreNumVector(void) const {
    return 40u;
  }
  /**
   * Calc task schedule block dim
   * @sliceNum number slice of data division
   * @aicCoreNum value of GetCoreNumAic() if used cube API, otherwise 0
   * @aivCoreNum value of GetCoreNumAiv() if used vector API, otherwise 0
   * @return task schedule block dim
   */
  uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const;
  /**
   * Get Work Space Size
   * @return work sapce size by chip type
   */
  uint32_t GetLibApiWorkSpaceSize(void) const {
    return 0u;
  }
  uint32_t GetResCubeGroupWorkSpaceSize(void) const {
    return 0u;
  }
  uint32_t GetResGroupBarrierWorkSpaceSize(void) const {
    return 0u;
  }
  void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const {
    if (memType == CoreMemType::L0_A) {
      constexpr uint64_t kL0AMemSize = 64 * 1024;
      size = kL0AMemSize;
      return;
    }
    if (memType == CoreMemType::L0_B) {
      constexpr uint64_t kL0BMemSize = 64 * 1024;
      size = kL0BMemSize;
      return;
    }
    if (memType == CoreMemType::L0_C) {
      constexpr uint64_t kL0CMemSize = 128 * 1024;
      size = kL0CMemSize;
      return;
    }
    if (memType == CoreMemType::L1) {
      constexpr uint64_t kL1MemSize = 512 * 1024;
      size = kL1MemSize;
      return;
    }
    if (memType == CoreMemType::L2) {
      constexpr uint64_t kL2MemSize = 192 * 1024 * 1024;
      size = kL2MemSize;
      return;
    }
    if (memType == CoreMemType::UB) {
      constexpr uint64_t kUBMemSize = 240 * 1024;
      size = kUBMemSize;
      return;
    }
    if (memType == CoreMemType::HBM) {
      constexpr uint64_t kUBMemSize = 10 * 1024 * 1024;
      size = kUBMemSize;
      return;
    }
    return;
  }
  void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const {
    bwSize = 64u;
  }
  /**
   * Get Soc Version Enum
   * @return Enum SocVersion
   */
  SocVersion GetSocVersion(void) const {
    return SocVersion::ASCEND910B;
  }

 private:
  fe::PlatFormInfos *platformInfo_;
};
class PlatformAscendCManager {
 public:
  static PlatformAscendC *GetInstance() {
    const std::lock_guard<std::mutex> lock(platformInitMtx);
    if (platformInfo == nullptr) {
      PlatformAscendCManagerInit(nullptr);
      if (platformInfo == nullptr) {
        return nullptr;
      }
    }
    return platformInfo;
  }
  static PlatformAscendC *GetInstance(const char *customSocVersion) {
    const std::lock_guard<std::mutex> lock(platformInitMtx);
    if (platformInfo == nullptr) {
      PlatformAscendCManagerInit(customSocVersion);
      if (platformInfo == nullptr) {
        return nullptr;
      }
    }
    return platformInfo;
  }

 private:
  static PlatformAscendC *platformInfo;
  static std::mutex platformInitMtx;
  static PlatformAscendC *PlatformAscendCManagerInit(const char *customSocVersion);
  static SocVersion SocVersionMap(const char *socVersionStr);
  static fe::PlatFormInfos *PlatformAscendCInit(const char *customSocVersion);
  PlatformAscendCManager();
  ~PlatformAscendCManager() = default;
};
}  // namespace platform_ascendc
#endif // SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_PLATFORM_ASCENDC_H

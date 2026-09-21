/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PLUGIN_ENGINE_DNNENGINES_H_
#define GE_PLUGIN_ENGINE_DNNENGINES_H_

#include <map>
#include <memory>
#include <string>

#include "framework/engine/dnnengine.h"
#include "engines/manager/engine/engine_manager.h"

namespace ge {
class GE_FUNC_VISIBILITY AICoreDNNEngine : public DNNEngine {
 public:
  explicit AICoreDNNEngine(const std::string &engine_name);
  explicit AICoreDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~AICoreDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY CustomDNNEngine : public DNNEngine {
 public:
  explicit CustomDNNEngine(const std::string &engine_name);
  explicit CustomDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~CustomDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY VectorCoreDNNEngine : public DNNEngine {
 public:
  explicit VectorCoreDNNEngine(const std::string &engine_name);
  explicit VectorCoreDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~VectorCoreDNNEngine() override = default;
};


class GE_FUNC_VISIBILITY AICpuDNNEngine : public DNNEngine {
 public:
  explicit AICpuDNNEngine(const std::string &engine_name);
  explicit AICpuDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~AICpuDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY AICpuFftsPlusDNNEngine : public AICpuDNNEngine {
 public:
  explicit AICpuFftsPlusDNNEngine(const std::string &engine_name) : AICpuDNNEngine(engine_name) {}
  explicit AICpuFftsPlusDNNEngine(const DNNEngineAttribute &attrs) : AICpuDNNEngine(attrs) {}
  ~AICpuFftsPlusDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY AICpuAscendFftsPlusDNNEngine : public AICpuDNNEngine {
 public:
  explicit AICpuAscendFftsPlusDNNEngine(const std::string &engine_name) : AICpuDNNEngine(engine_name) {}
  explicit AICpuAscendFftsPlusDNNEngine(const DNNEngineAttribute &attrs) : AICpuDNNEngine(attrs) {}
  ~AICpuAscendFftsPlusDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY DvppDNNEngine : public DNNEngine {
 public:
  explicit DvppDNNEngine(const std::string &engine_name);
  explicit DvppDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~DvppDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY AICpuTFDNNEngine : public DNNEngine {
 public:
  explicit AICpuTFDNNEngine(const std::string &engine_name);
  explicit AICpuTFDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~AICpuTFDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY GeLocalDNNEngine : public DNNEngine {
 public:
  explicit GeLocalDNNEngine(const std::string &engine_name);
  explicit GeLocalDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~GeLocalDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY HostCpuDNNEngine : public DNNEngine {
public:
  explicit HostCpuDNNEngine(const std::string &engine_name);
  explicit HostCpuDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~HostCpuDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY RtsDNNEngine : public DNNEngine {
 public:
  explicit RtsDNNEngine(const std::string &engine_name);
  explicit RtsDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~RtsDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY RtsFftsPlusDNNEngine : public DNNEngine {
 public:
  explicit RtsFftsPlusDNNEngine(const std::string &engine_name);
  explicit RtsFftsPlusDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~RtsFftsPlusDNNEngine() override = default;
};


class GE_FUNC_VISIBILITY HcclDNNEngine : public DNNEngine {
 public:
  explicit HcclDNNEngine(const std::string &engine_name);
  explicit HcclDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~HcclDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY FftsPlusDNNEngine : public DNNEngine {
 public:
  explicit FftsPlusDNNEngine(const std::string &engine_name);
  explicit FftsPlusDNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~FftsPlusDNNEngine() override = default;
};

class GE_FUNC_VISIBILITY DSADNNEngine : public DNNEngine {
 public:
  explicit DSADNNEngine(const std::string &engine_name);
  explicit DSADNNEngine(const DNNEngineAttribute &attrs) : DNNEngine(attrs) {}
  ~DSADNNEngine() override = default;
};
}  // namespace ge
#endif  // GE_PLUGIN_ENGINE_DNNENGINES_H_

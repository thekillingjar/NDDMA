/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "model_inference.h"
#include <acl.h>
#include <iostream>
#include <utility>
#include <tensorflow_parser.h>

namespace gerec {
ModelInference::Builder::Builder(const std::string& modelPath, const std::string& modelType) {
  infer_params_.model_path_ = modelPath;
  infer_params_.model_type_ = modelType;
}

ModelInference::Builder &ModelInference::Builder::AiCoreNum(const std::string &aiCoreNum) {
  infer_params_.ai_core_num_ = aiCoreNum;
  return *this;
}

ModelInference::Builder &ModelInference::Builder::InputBatchCopy(bool enableBatchH2D) {
  infer_params_.enable_input_batch_cpy_ = enableBatchH2D;
  return *this;
}

ModelInference::Builder &ModelInference::Builder::MultiInstanceNum(int32_t multiInstanceNum) {
  infer_params_.multi_instance_num_ = multiInstanceNum;
  return *this;
}

ModelInference::Builder &ModelInference::Builder::MaxQueueSize(size_t maxSize) {
  infer_params_.global_max_queue_size_ = maxSize;
  return *this;
}

ModelInference::Builder &ModelInference::Builder::GraphParserParams(
    const std::map<ge::AscendString, ge::AscendString> &parserParams) {
  infer_params_.parser_params_ = parserParams;
  return *this;
}

std::unique_ptr<ModelInference> ModelInference::Builder::Build() {
  return std::make_unique<ModelInference>(infer_params_);
}

ModelInference::ModelInference(const InferenceParams &inference_params)
  : model_path_(inference_params.model_path_),
    model_type_(inference_params.model_type_),
    multi_instance_num_(inference_params.multi_instance_num_),
    ai_core_num_(inference_params.ai_core_num_),
    parser_params_(inference_params.parser_params_),
    enable_input_batch_cpy_(inference_params.enable_input_batch_cpy_),
    global_max_queue_size_(inference_params.global_max_queue_size_) {
}

void ModelInference::GraphWorker::Run(const std::shared_ptr<ge::Session> &session) {
  aclError aerr = aclrtSetDevice(deviceId);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << "aclrtSetDevice failed, ret=" << aerr << std::endl;
    return;
  }
  aerr = aclrtCreateStream(&stream);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << "aclrtCreateStream failed, ret=" << aerr << std::endl;
    return;
  }
  ge::Status gerr = session->LoadGraph(graphId, {}, stream);
  if (gerr != ge::SUCCESS) {
    std::cerr << "LoadGraph failed!" << std::endl;
    aclrtDestroyStream(stream);
    return;
  }
  std::cout << "LoadGraph [" << graphId << "] success!" << std::endl;

  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [this]() {
        return stop || !tasks.empty();
      });
      if (stop && tasks.empty()) break;
      task = std::move(tasks.front());
      tasks.pop();
    }
    try {
      task();
    } catch (const std::exception &ex) {
      std::cerr << "GraphWorker task threw exception: " << ex.what() << std::endl;
    } catch (...) {
      std::cerr << "GraphWorker task threw unknown exception" << std::endl;
    }
  }
  if (stream != nullptr) {
    // 确保所有任务执行完成
    aclError aerr = aclrtSynchronizeStream(stream);
    if (aerr != ACL_ERROR_NONE) {
      std::cerr << "Synchronize stream failed, ret=" << aerr << std::endl;
    }
    aclrtDestroyStream(stream);
    stream = nullptr;
  }
}

bool ModelInference::GraphWorker::Enqueue(std::function<void()> task) {
  std::unique_lock<std::mutex> lock(mtx);
  if (stop) return false;
  if (tasks.size() >= max_queue_size) {
    return false;
  }
  tasks.push(std::move(task));
  cv.notify_one();
  return true;
}

inline void FreeDevice(std::vector<gert::Tensor> &device_tensors) {
  for (auto &t : device_tensors) {
    if (t.GetAddr() != nullptr) aclrtFree(t.GetAddr());
  }
}

inline bool CopyBatch(MemcpyBatchParam &memcpy_batch_param) {
  size_t failIndex = SIZE_MAX;
  aclError aerr = aclrtMemcpyBatch(
      memcpy_batch_param.dsts.data(),
      memcpy_batch_param.destMaxs.data(),
      memcpy_batch_param.srcs.data(),
      memcpy_batch_param.sizes.data(),
      memcpy_batch_param.numBatches,
      memcpy_batch_param.attrs.data(),
      memcpy_batch_param.attrsIndexes.data(),
      memcpy_batch_param.numAttrs,
      &failIndex);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << " batch failed, ret=" << aerr
        << ", failIndex=" << failIndex << std::endl;
    return false;
  }
  return true;
}

inline bool CopySingle(const std::vector<gert::Tensor> &src, std::vector<gert::Tensor> &dst, aclrtMemcpyKind kind) {
  if (src.size() != dst.size()) return false;

  for (size_t i = 0; i < src.size(); ++i) {
    size_t bytes = src[i].GetSize();
    aclError aerr = aclrtMemcpy(dst[i].GetAddr(), bytes,
                                src[i].GetAddr(), bytes,
                                kind);
    if (aerr != ACL_ERROR_NONE) {
      std::cerr << "CopySingle failed at index " << i
          << ", ret=" << aerr << std::endl;
      return false;
    }
  }
  return true;
}

inline void BuildMemcpyBatchParams(const std::vector<gert::Tensor> &src, const std::vector<gert::Tensor> &dst,
                                   MemcpyBatchParam &memcpy_batch_params, aclrtMemLocationType srcLoc,
                                   aclrtMemLocationType dstLoc, int32_t deviceId) {
  for (size_t i = 0U; i < dst.size(); ++i) {
    aclrtMemcpyBatchAttr attr;
    attr.dstLoc.type = dstLoc;
    attr.srcLoc.type = srcLoc;
    attr.srcLoc.id = (srcLoc == ACL_MEM_LOCATION_TYPE_DEVICE) ? deviceId : 0;
    attr.dstLoc.id = (dstLoc == ACL_MEM_LOCATION_TYPE_DEVICE) ? deviceId : 0;
    for (uint32_t i = 0U; i < sizeof(aclrtMemcpyBatchAttr::rsv); ++i) {
      attr.rsv[i] = 0U;
    }

    memcpy_batch_params.dsts.emplace_back(const_cast<void *>(dst[i].GetAddr()));
    memcpy_batch_params.destMaxs.emplace_back(dst[i].GetSize());
    memcpy_batch_params.srcs.emplace_back(const_cast<void *>(src[i].GetAddr()));
    memcpy_batch_params.sizes.emplace_back(src[i].GetSize());
    memcpy_batch_params.attrs.emplace_back(attr);
    memcpy_batch_params.attrsIndexes.emplace_back(i);
  }
  memcpy_batch_params.numBatches = static_cast<uint32_t>(dst.size());
  memcpy_batch_params.numAttrs = static_cast<uint32_t>(memcpy_batch_params.attrs.size());
}

inline bool CopyH2D(const std::vector<gert::Tensor> &host, std::vector<gert::Tensor> &dev, bool enableBatch) {
  if (enableBatch) {
    int32_t deviceId = -1;
    aclError aerr = aclrtGetDevice(&deviceId);
    if (aerr != ge::SUCCESS) {
      std::cerr << "aclrtGetDevice failed! Please set device id firstly!" << std::endl;
      return false;
    }
    MemcpyBatchParam memcpy_batch_params;
    BuildMemcpyBatchParams(host, dev, memcpy_batch_params, ACL_MEM_LOCATION_TYPE_HOST,
                           ACL_MEM_LOCATION_TYPE_DEVICE,
                           deviceId);
    return CopyBatch(memcpy_batch_params);
  }
  return CopySingle(host, dev, ACL_MEMCPY_HOST_TO_DEVICE);
}

inline bool CopyD2H(const std::vector<gert::Tensor> &dev, std::vector<gert::Tensor> &host, bool enableBatch) {
  if (enableBatch) {
    int32_t deviceId = -1;
    aclError aerr = aclrtGetDevice(&deviceId);
    if (aerr != ge::SUCCESS) {
      std::cerr << "aclrtGetDevice failed! Please set device id firstly!" << std::endl;
      return false;
    }
    MemcpyBatchParam memcpy_batch_params;
    BuildMemcpyBatchParams(dev, host, memcpy_batch_params, ACL_MEM_LOCATION_TYPE_DEVICE, ACL_MEM_LOCATION_TYPE_HOST,
                           deviceId);
    return CopyBatch(memcpy_batch_params);
  }
  return CopySingle(dev, host, ACL_MEMCPY_DEVICE_TO_HOST);
}

inline bool MakeDeviceTensorFromHost(const std::vector<gert::Tensor> &host_tensors,
                                     std::vector<gert::Tensor> &device_tensors) {
  device_tensors.clear();
  device_tensors.reserve(host_tensors.size());
  for (const auto &ht : host_tensors) {
    const size_t bytes = ht.GetSize();
    void *dev = nullptr;
    aclError aerr = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aerr != ACL_ERROR_NONE) {
      std::cerr << "aclrtMalloc failed, ret=" << aerr << std::endl;
      return false;
    }
    gert::TensorData td(dev, nullptr, bytes, gert::kOnDeviceHbm);
    gert::Tensor dt;
    dt.SetData(std::move(td));
    device_tensors.emplace_back(std::move(dt));
  }
  return true;
}

void ModelInference::GraphTask::operator()() {
  auto exec_start = std::chrono::high_resolution_clock::now();

  auto safe_callback = [&](std::shared_ptr<std::vector<gert::Tensor>> outputs,
                           std::shared_ptr<std::vector<gert::Tensor>> inputs,
                           bool status, long long exec_us) {
    try {
      if (callback) callback(outputs, inputs, status, exec_us);
    } catch (const std::exception &ex) {
      std::cerr << "Callback exception: " << ex.what() << std::endl;
    } catch (...) {
      std::cerr << "Callback unknown exception" << std::endl;
    }
  };
  std::vector<gert::Tensor> device_outputs;
  std::vector<gert::Tensor> device_inputs;
  if (!MakeDeviceTensorFromHost(*host_inputs, device_inputs)) {
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }
  if (!MakeDeviceTensorFromHost(*host_outputs, device_outputs)) {
    FreeDevice(device_inputs);
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }
  if (!CopyH2D(*host_inputs, device_inputs, worker->batchCopy)) {
    FreeDevice(device_inputs);
    FreeDevice(device_outputs);
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }

  ge::Status ret = session->
      ExecuteGraphWithStreamAsync(worker->graphId, worker->stream, device_inputs,
                                  device_outputs);
  if (ret != ge::SUCCESS) {
    std::cerr << "ExecuteGraphWithStreamAsync failed!" << std::endl;
    FreeDevice(device_inputs);
    FreeDevice(device_outputs);
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }

  aclError aerr = aclrtSynchronizeStream(worker->stream);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << "Synchronize stream failed after executeGraphWithStreamAsync, ret=" << aerr << std::endl;
    FreeDevice(device_inputs);
    FreeDevice(device_outputs);
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }

  if (!CopyD2H(device_outputs, *host_outputs, worker->batchCopy)) {
    FreeDevice(device_inputs);
    FreeDevice(device_outputs);
    safe_callback(host_outputs, host_inputs, false, 0);
    return;
  }

  FreeDevice(device_inputs);
  FreeDevice(device_outputs);

  auto exec_end = std::chrono::high_resolution_clock::now();
  long long exec_us = std::chrono::duration_cast<std::chrono::microseconds>(
      exec_end - exec_start).count();

  safe_callback(host_outputs, host_inputs, true, exec_us);
}

ModelInference::~ModelInference() {
  for (auto &w : workers_) {
    {
      std::unique_lock<std::mutex> lock(w->mtx);
      w->stop = true;
    }
    w->cv.notify_all();
    if (w->worker.joinable()) {
      w->worker.join();
    }
  }
  workers_.clear();
  ge::GEFinalize();
}

ge::Status ModelInference::Init() {
  if (model_type_ != "TensorFlow") {
    std::cout << "Unsupport Model Type: " << model_type_ << std::endl;
    return ge::FAILED;
  }

  std::map<ge::AscendString, ge::AscendString> session_options = {
      {ge::AscendString("ge.exec.precision_mode"), ge::AscendString("allow_fp32_to_fp16")}
  };
  if (!ai_core_num_.empty()) {
    session_options.emplace(ge::AscendString("ge.aicoreNum"), ge::AscendString(ai_core_num_.c_str()));
  }

  std::map<ge::AscendString, ge::AscendString> global_options = {
      {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")}
  };

  ge::Status gerr = ge::GEInitialize(global_options);
  if (gerr != ge::SUCCESS) {
    std::cerr << "GEInitialize failed!" << std::endl;
    return ge::FAILED;
  }
  std::cout << "GE Initialize success!" << std::endl;

  session_ = std::make_shared<ge::Session>(session_options);

  for (int i = 0; i < multi_instance_num_; ++i) {
    auto graph = std::make_unique<ge::Graph>();
    ge::graphStatus pstat = aclgrphParseTensorFlow(model_path_.c_str(), parser_params_, *graph);
    if (pstat != ge::GRAPH_SUCCESS) {
      std::cerr << "Failed to parse TF graph, status=" << pstat << std::endl;
      return ge::FAILED;
    }

    const uint32_t graphId = static_cast<uint32_t>(i);
    gerr = session_->AddGraph(graphId, *graph);
    if (gerr != ge::SUCCESS) {
      std::cerr << "AddGraph failed for graph " << i << std::endl;
      return ge::FAILED;
    }
    gerr = session_->CompileGraph(graphId);
    if (gerr != ge::SUCCESS) {
      std::cerr << "Compile failed for graph " << i << std::endl;
      return ge::FAILED;
    }
    std::cout << "Graph[" << graphId << "] compiled successfully!" << std::endl;

    int32_t deviceId = -1;
    aclError aerr = aclrtGetDevice(&deviceId);
    if (aerr != ge::SUCCESS) {
      std::cerr << "aclrtGetDevice failed! Please set device id firstly!" << std::endl;
      return ge::FAILED;
    }

    auto worker = std::make_unique<GraphWorker>();
    worker->graphId = graphId;
    worker->deviceId = deviceId;
    worker->graph = std::move(graph);
    worker->max_queue_size = global_max_queue_size_;
    worker->worker = std::thread([this, wptr = worker.get()]() {
      wptr->Run(session_);
    });
    worker->batchCopy = enable_input_batch_cpy_;
    workers_.push_back(std::move(worker));
  }

  if (workers_.empty()) {
    std::cerr << "No workers created!" << std::endl;
    return ge::FAILED;
  }
  std::cout << "ModelInference Init success. There are [" << multi_instance_num_ <<
      "] inference instance , input_batch_cpy is " << enable_input_batch_cpy_ << ", ai_core_num_ is [" << ai_core_num_
      << "]" << std::endl;
  return ge::SUCCESS;
}

ge::Status ModelInference::RunGraphAsync(const std::shared_ptr<std::vector<gert::Tensor>> &host_inputs,
                                         const std::shared_ptr<std::vector<gert::Tensor>> &host_outputs,
                                         const Callback &callback) {
  if (!session_) {
    std::cerr << "Session not initialized!" << std::endl;
    return ge::FAILED;
  }

  size_t worker_count = workers_.size();
  size_t idx = next_worker_index_.fetch_add(1) % worker_count;
  auto &worker = workers_[idx];

  GraphTask graphTask{session_, worker.get(), host_inputs, host_outputs, callback};

  if (!worker->Enqueue([task = std::move(graphTask)]() mutable {
    task();
  })) {
    std::cerr << "Task queue full for worker " << idx << std::endl;
    return ge::FAILED;
  }
  return ge::SUCCESS;
}
} // namespace gerec
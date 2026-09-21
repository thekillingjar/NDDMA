/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl.h>
#include <acl_rt.h>
#include <sstream>
#include <random>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <complex>
#include <iostream>
#include <vector>
#include <map>
#include "model_inference.h"
#include <getopt.h>
#include <string>

struct TensorSpec {
  std::string name;
  ge::DataType dtype;
  gert::StorageShape shape;
  gert::StorageFormat format;
};

struct ModelConfig {
  int32_t batch_size{1};
  std::string model_path;
  std::string model_type;
  std::vector<TensorSpec> input_specs;
  std::vector<TensorSpec> output_specs;
  std::map<ge::AscendString, ge::AscendString> parser_params;
};

void PrintHelpInfo() {
  std::cout << "Usage: ./recomand_exec [-h or --help] [-r or --runs=<number of runs>]" << std::endl;
  std::cout << "  --runs               Number of runs (default: 10000)" << std::endl;
  std::cout << "  --batchSize          Batch size (default: 128)" << std::endl;
  std::cout << "  --multiInstanceNum   Multi-instance number (default: 1)" << std::endl;
  std::cout << "  --enableBatchH2D     Enable batch H2D (true/false, 1/0) (default: false)" << std::endl;
  std::cout << "  --aiCoreNum          AI core numbers (default:\"\")" << std::endl;
  std::cout << "  --help               Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  ./recomand_exec --runs=1000 --batchSize=128" << std::endl;
  std::cout << "  ./recomand_exec --multiInstanceNum=4 --enableBatchH2D=true --aiCoreNum=\"16|16\"" << std::endl;
  std::cout << "  ./recomand_exec --help" << std::endl;
}

uint32_t GetDataTypeSize(ge::DataType dt) {
  static const std::unordered_map<ge::DataType, uint32_t> type_size_map = {
      {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_INT8, 1},
      {ge::DT_INT16, 2}, {ge::DT_INT32, 4}, {ge::DT_INT64, 8},
      {ge::DT_UINT32, 4}, {ge::DT_UINT64, 8}
  };
  auto it = type_size_map.find(dt);
  return (it != type_size_map.end()) ? it->second : 1;
}

ge::Status AllocateBufferForHostTensor(gert::Tensor &tensor) {
  const uint32_t elem_count = tensor.GetShapeSize();
  const uint32_t elem_size = GetDataTypeSize(tensor.GetDataType());
  if (elem_count > (std::numeric_limits<uint32_t>::max() / elem_size)) {
    std::cerr << "Multiplication overflow: datalen exceeds uint32_t limits" << std::endl;
    return ge::FAILED;
  }
  const uint32_t data_len = elem_count * elem_size;

  void *host_buf = nullptr;
  aclError ret = aclrtMallocHost(&host_buf, data_len);
  if (ret != ACL_ERROR_NONE || host_buf == nullptr) {
    std::cerr << "aclrtMallocHost failed, ret=" << ret << std::endl;
    return ge::FAILED;
  }
  gert::TensorData td(host_buf, nullptr, data_len, gert::kOnHost);
  tensor.SetData(std::move(td));
  return ge::SUCCESS;
}

ge::Status GenRandomDataForHostTensor(gert::Tensor &tensor) {
  void *host_buf = tensor.GetAddr();
  const uint32_t elem_count = tensor.GetShapeSize();

  static thread_local std::mt19937 rng(std::random_device{}());

  switch (tensor.GetDataType()) {
    case ge::DT_INT8: {
      auto *ptr = reinterpret_cast<int8_t *>(host_buf);
      std::uniform_int_distribution<int> dist(0, 127);
      for (uint32_t i = 0; i < elem_count; ++i) ptr[i] = static_cast<int8_t>(dist(rng));
      break;
    }
    case ge::DT_INT16: {
      auto *ptr = reinterpret_cast<int16_t *>(host_buf);
      std::uniform_int_distribution<int> dist(0, 32767);
      for (uint32_t i = 0; i < elem_count; ++i) ptr[i] = static_cast<int16_t>(dist(rng));
      break;
    }
    case ge::DT_INT32: {
      auto *ptr = reinterpret_cast<int32_t *>(host_buf);
      std::uniform_int_distribution<int32_t> dist(0, 1000);
      for (uint32_t i = 0; i < elem_count; ++i) ptr[i] = dist(rng);
      break;
    }
    case ge::DT_INT64: {
      auto *ptr = reinterpret_cast<int64_t *>(host_buf);
      std::uniform_int_distribution<int64_t> dist(0, 1000000);
      for (uint32_t i = 0; i < elem_count; ++i) ptr[i] = dist(rng);
      break;
    }
    case ge::DT_FLOAT: {
      auto *ptr = reinterpret_cast<float *>(host_buf);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      for (uint32_t i = 0; i < elem_count; ++i) ptr[i] = dist(rng);
      break;
    }
    default:
      std::cerr << "Unsupported dtype in GenRandomDataForHostTensor: " << tensor.GetDataType() << std::endl;
      return ge::FAILED;
  }
  return ge::SUCCESS;
}

void FreeHostTensors(const std::shared_ptr<std::vector<gert::Tensor>> &tensors) {
  for (auto &t : *tensors) {
    if (t.GetAddr()) aclrtFreeHost(t.GetAddr());
  }
}

ge::Status PrepareHostData(const ModelConfig &cfg, int num_runs,
                           std::vector<std::shared_ptr<std::vector<gert::Tensor>>> &all_inputs,
                           std::vector<std::shared_ptr<std::vector<gert::Tensor>>> &all_outputs) {
  all_inputs.reserve(num_runs);
  all_outputs.reserve(num_runs);
  for (int i = 0; i < num_runs; ++i) {
    auto inputs = std::make_shared<std::vector<gert::Tensor>>();
    auto outputs = std::make_shared<std::vector<gert::Tensor>>();

    for (const TensorSpec &spec : cfg.input_specs) {
      auto tensor = gert::Tensor(spec.shape, spec.format, spec.dtype);
      if (AllocateBufferForHostTensor(tensor) != ge::SUCCESS) return ge::FAILED;
      if (GenRandomDataForHostTensor(tensor) != ge::SUCCESS) return ge::FAILED;
      inputs->push_back(std::move(tensor));
    }

    for (const auto &spec : cfg.output_specs) {
      auto tensor = gert::Tensor(spec.shape, spec.format, spec.dtype);
      if (AllocateBufferForHostTensor(tensor) != ge::SUCCESS) return ge::FAILED;
      outputs->push_back(std::move(tensor));
    }

    all_inputs.push_back(inputs);
    all_outputs.push_back(outputs);
  }
  return ge::SUCCESS;
}

struct Args {
  int runs{10000};
  int32_t batchSize{128};
  int32_t multiInstanceNum{1};
  bool enableBatchH2D{false};
  std::string aiCoreNum;
  bool help{false};
};

inline bool ParseArgs(Args &args, int argc, char **argv) {
  static struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"runs", required_argument, 0, 'r'},
      {"batchSize", required_argument, 0, 'b'},
      {"multiInstanceNum", required_argument, 0, 'm'},
      {"enableBatchH2D", required_argument, 0, 'e'},
      {"aiCoreNum", required_argument, 0, 'a'},
      {nullptr, 0, nullptr, 0}
  };

  while (true) {
    int option_index = 0;
    int arg = getopt_long(argc, argv, "hr:b:m:e:a:", long_options, &option_index);
    if (arg == -1) break;
    switch (arg) {
      case 'h':
        args.help = true;
        return true;
      case 'r':
        try {
          args.runs = std::stoi(optarg);
          if (args.runs <= 0) throw std::invalid_argument("runs <= 0");
        } catch (...) {
          std::cerr << "ERROR: invalid --runs value: " << optarg << std::endl;
          return false;
        }
        break;

      case 'b':
        try {
          args.batchSize = std::stoi(optarg);
          if (args.batchSize <= 0) throw std::invalid_argument("batchSize <= 0");
        } catch (...) {
          std::cerr << "ERROR: invalid --batchSize value: " << optarg << std::endl;
          return false;
        }
        break;

      case 'm':
        try {
          args.multiInstanceNum = std::stoi(optarg);
          if (args.multiInstanceNum <= 0) throw std::invalid_argument("multiInstanceNum <= 0");
        } catch (...) {
          std::cerr << "ERROR: invalid --multiInstanceNum value: " << optarg << std::endl;
          return false;
        }
        break;

      case 'e':
        if (std::string(optarg) == "true" || std::string(optarg) == "1") args.enableBatchH2D = true;
        else if (std::string(optarg) == "false" || std::string(optarg) == "0") args.enableBatchH2D = false;
        else {
          std::cerr << "ERROR: invalid --enableBatchH2D value: " << optarg << std::endl;
          return false;
        }
        break;

      case 'a':
        args.aiCoreNum = optarg;
        break;

      default:
        return false;
    }
  }

  std::cout << "recomand config: runs=" << args.runs
      << ", batchSize=" << args.batchSize
      << ", multiInstanceNum=" << args.multiInstanceNum
      << ", enableBatchH2D=" << (args.enableBatchH2D ? "true" : "false")
      << ", aiCoreNum=" << (args.aiCoreNum.empty() ? "\"\"" : args.aiCoreNum)
      << std::endl;

  return true;
}

// 构建DCN_v2.pb模型配置
ModelConfig BuildDCNV2Config(const std::string &model_path, const std::string &model_type, const int32_t batchSize) {
  static const std::vector<std::string> modelInputs = {
      "Input_21", "Input_24", "Input_4", "Input_6", "Input_12", "Input_19",
      "Input_3", "Input_22", "Input_23", "Input_13", "Input_18", "Input_10",
      "Input_1", "Input_14", "Input_26", "Input_8", "Input_2", "Input_15",
      "Input_16", "Input_17", "Input_20", "Input_7", "Input", "Input_11", "Input_5",
      "Input_25", "Input_9"
  };

  std::vector<TensorSpec> inputs;
  inputs.reserve(modelInputs.size());
  auto makeIntInput = [&](const std::string &name) {
    return TensorSpec{name, ge::DT_INT32, {{batchSize}, {batchSize}},
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}}
    };
  };

  const std::string specialInput = "Input";
  for (const auto &name : modelInputs) {
    if (name == specialInput) {
      inputs.emplace_back(TensorSpec{"Input", ge::DT_FLOAT, {{batchSize, 8}, {batchSize, 8}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}}
      });
      continue;
    }
    inputs.emplace_back(makeIntInput(name));
  }

  std::vector<TensorSpec> outputs = {
      {"Identity:0", ge::DT_FLOAT, {{batchSize}, {batchSize}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}}
  };

  // 构建DCN_v2模型解析参数
  std::stringstream ss;
  int input_size = 27;
  for (int i = 1; i < input_size; ++i) ss << "Input_" << i << ":" << batchSize << ";";
  ss << "Input:" << batchSize << ",8";

  std::map<ge::AscendString, ge::AscendString> parser = {
      {ge::AscendString(ge::ir_option::OUT_NODES),
       ge::AscendString("Identity:0")},
      {ge::AscendString(ge::ir_option::INPUT_SHAPE),
       ge::AscendString(ss.str().c_str())}
  };

  ModelConfig cfg;
  cfg.model_path = model_path;
  cfg.model_type = model_type;
  cfg.input_specs = std::move(inputs);
  cfg.output_specs = std::move(outputs);
  cfg.parser_params = std::move(parser);
  cfg.batch_size = batchSize;
  return cfg;
}

ge::Status RunInference(const ModelConfig &cfg, int num_runs, int32_t multiInstanceNum,
                        bool enableBatchH2D, const std::string &aiCoreNum) {
  auto model_inference = gerec::ModelInference::Builder(cfg.model_path, cfg.model_type)
                         .InputBatchCopy(enableBatchH2D)
                         .AiCoreNum(aiCoreNum)
                         .MultiInstanceNum(multiInstanceNum)
                         .GraphParserParams(cfg.parser_params)
                         .Build();
  if (model_inference->Init() != ge::SUCCESS) {
    std::cerr << "Init ModelInference failed" << std::endl;
    return ge::FAILED;
  }

  // 构造输入输出
  std::vector<std::shared_ptr<std::vector<gert::Tensor>>> all_inputs;
  std::vector<std::shared_ptr<std::vector<gert::Tensor>>> all_outputs;
  if (PrepareHostData(cfg, num_runs, all_inputs, all_outputs) != ge::SUCCESS) {
    for (const auto &t : all_inputs) {
      FreeHostTensors(t);
    }
    for (const auto &t : all_outputs) {
      FreeHostTensors(t);
    }
    return ge::FAILED;
  }

  std::atomic<int> success_count{0};
  std::atomic<long long> total_exec_us{0};

  auto global_start = std::chrono::high_resolution_clock::now();

  auto callback = [&](std::shared_ptr<std::vector<gert::Tensor>> outputs,
                      std::shared_ptr<std::vector<gert::Tensor>> inputs, bool status, long long exec_us) {
    if (status) {
      success_count.fetch_add(1, std::memory_order_relaxed);
      total_exec_us.fetch_add(exec_us, std::memory_order_relaxed);
    }
    FreeHostTensors(outputs);
    FreeHostTensors(inputs);
  };
  for (int i = 0; i < num_runs; ++i) {
    if (model_inference->RunGraphAsync(all_inputs[i], all_outputs[i], callback) != ge::SUCCESS) {
      std::cerr << "RunGraphAsync failed at " << i << std::endl;
      return ge::FAILED;
    }
  }

  model_inference.reset(); // 主动析构，等待所有任务执行完

  auto global_end = std::chrono::high_resolution_clock::now();
  double elapsed_sec = std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "Run " << num_runs << " inferences in " << elapsed_sec << " seconds. Success count: " << success_count.
      load() << std::endl;
  std::cout << "Throughput: " << (num_runs * cfg.batch_size / elapsed_sec) << " inferences/sec" << std::endl;

  double avg_ms = static_cast<double>(total_exec_us.load()) / success_count.load() / 1000.0;
  std::cout << "Average execution latency: " << avg_ms << " ms" << std::endl;

  return ge::SUCCESS;
}

int main(int argc, char *argv[]) {
  Args args;
  if (!ParseArgs(args, argc, argv)) {
    std::cerr << "ERROR: invalid arguments" << std::endl;
    return EXIT_FAILURE;
  }
  if (args.help) {
    PrintHelpInfo();
    return EXIT_SUCCESS;
  }

  aclError aerr = aclInit(nullptr);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << "Failed to init ACL, error=" << aerr << std::endl;
    return EXIT_FAILURE;
  }
  aerr = aclrtSetDevice(0);
  if (aerr != ACL_ERROR_NONE) {
    std::cerr << "aclrtSetDevice failed, ret=" << aerr << std::endl;
    aclFinalize();
    return EXIT_FAILURE;
  }

  const std::string model_path = "../data/DCN_v2.pb";
  const std::string model_type = "TensorFlow";
  ModelConfig cfg = BuildDCNV2Config(model_path, model_type, args.batchSize);
  ge::Status status = RunInference(cfg, args.runs, args.multiInstanceNum, args.enableBatchH2D, args.aiCoreNum);
  aclFinalize();
  if (status != ge::SUCCESS) {
    std::cerr << "RunInference failed" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
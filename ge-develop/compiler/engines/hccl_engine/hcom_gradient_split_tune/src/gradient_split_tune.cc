/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gradient_split_tune.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <climits>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include "hcom_gradient_split_tune.h"
#include "tune_log.h"
#include "platform/platform_info.h"
#include "mmpa_api.h"

namespace {
const std::string TRACE_OP_TYPE = "TensorRedirect";
const float MAX_SD = 3.0;
const int MAX_BUF_LEN = 1024;
std::string globalGraphId_;
}  // namespace

TuneResult_t CheckTimeSd(const std::vector<std::vector<uint64_t>> &time, std::vector<struct BPTimeInfo> &bpTimeInfo) {
  uint32_t i, j;
  double sd = 0.0;

  if (time.size() == 0 || time[0].size() == 0 || bpTimeInfo.size() != time[0].size()) {
    TUNE_ERROR("time.size:%u <= 1 or time[0].size:%u or bpTimeInfo.size:%u != time[0].size, invlaid", time.size(),
               time[0].size(), bpTimeInfo.size());
    return TUNE_E_PARA;
  }

  for (i = 1; i < time.size(); i++) {
    if (time[i].size() != time[0].size()) {
      TUNE_ERROR("time[%d].size[%llu] != time[0].size[%llu]", i, time[i].size(), time[0].size());
      return TUNE_E_PARA;
    }
  }

  if (time.size() == 1) { /* only one set of data, no need to calculate standard deviation */
    return TUNE_SUCCESS;
  }

  for (j = 1; j < time[0].size(); j++) { /* the first column is start time, excluding */
    for (i = 0; i < time.size(); i++) {
      sd += (time[i][j] - bpTimeInfo[j].time) * (time[i][j] - bpTimeInfo[j].time);
    }
    sd /= (time.size() - 1);
    sd = sqrt(sd);
    if (sd > MAX_SD) {
      TUNE_WARN("the j:%d opName:%s standard deviation:%d is too big", j, bpTimeInfo[j].opName.c_str(), sd);
    }
  }

  return TUNE_SUCCESS;
}

static TuneResult_t CheckTaskTBEList(const std::vector<std::vector<struct ProfilingMetric>> &taskTBEList, uint32_t &row,
                                     uint32_t &col) {
  if (taskTBEList.size() == 0) {
    TUNE_ERROR("taskTBEList.size is 0, invlaid");
    return TUNE_E_PARA;
  }

  for (uint32_t k = 0; k < taskTBEList.size(); k++) {
    if (taskTBEList[k].size() != taskTBEList[0].size()) {
      TUNE_ERROR("taskTBEList[%u].size[%u] is not same to the taskTBEList[0].size[%u]", k, taskTBEList[k].size(),
                 taskTBEList[0].size());
      return TUNE_E_PARA;
    }
  }

  row = taskTBEList.size();
  col = taskTBEList[0].size();
  return TUNE_SUCCESS;
}

static TuneResult_t GetTBETime(const std::vector<std::vector<struct ProfilingMetric>> &taskTBEList,
                               std::vector<std::vector<uint64_t>> &time) {
  TuneResult_t ret;
  uint32_t i, j;
  uint32_t row, col;

  ret = CheckTaskTBEList(taskTBEList, row, col);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("the taskTBEList is invalid, ret:%d", ret);
    return ret;
  }

  for (i = 0; i < row; i++) {
    for (j = 0; j < col; j++) {
      if (j == 0) {
        time[i][j] = taskTBEList[i][j].taskStartTime;
        continue;
      }
      if (taskTBEList[i][j].taskStartTime < taskTBEList[i][j - 1].taskEndTime) {
        TUNE_ERROR("taskStartTime[%llu] < the pre taskStartTime[%llu], invliad", taskTBEList[i][j].taskStartTime,
                   taskTBEList[i][j - 1].taskEndTime);
        return TUNE_E_PARA;
      }
      time[i][j] = taskTBEList[i][j].taskStartTime - taskTBEList[i][j - 1].taskEndTime;
    }
  }

  return TUNE_SUCCESS;
}

static TuneResult_t GetAndCheckTBETime(const std::vector<std::vector<struct ProfilingMetric>> taskTBEList,
                                       std::vector<struct BPTimeInfo> &bpTimeInfo) {
  TuneResult_t ret;
  uint32_t row, col;
  struct BPTimeInfo bpTimeTmp;

  ret = CheckTaskTBEList(taskTBEList, row, col);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("the taskTBEList is invalid, ret:%d", ret);
    return ret;
  }

  std::vector<std::vector<uint64_t>> time(row, std::vector<uint64_t>(col, 0));

  ret = GetTBETime(taskTBEList, time);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("GetTBETime failed, ret:%d", ret);
    return ret;
  }
  TUNE_INFO("Current GDAT Run Loop[%u]", row);
  const uint32_t invalidLoopNum = 3;
  if (row <= invalidLoopNum) {
    TUNE_ERROR("Invalid Loop Num[%u] is more than All Loop Num[%u]", invalidLoopNum, row);
    return TUNE_E_PARA;
  }
  for (uint32_t j = 0; j < col; j++) {
    std::vector<uint64_t> rowTime(row, 0);
    for (uint32_t i = 0; i < row; i++) {
      if (taskTBEList[i][j].opName != taskTBEList[0][j].opName) {
        TUNE_ERROR("taskTBEList[%u][%u].opName[%s] is not same to the taskTBEList[0][%u].opName[%s]", i, j,
                   taskTBEList[i][j].opName.c_str(), j, taskTBEList[0][j].opName.c_str());
        return TUNE_E_PARA;
      }
      rowTime[i] = time[i][j];
      TUNE_DEBUG("GDAT Time[%llu] Index[%u] Op[%s]", rowTime[i], i, taskTBEList[0][j].opName.c_str());
    }
    std::sort(rowTime.begin(), rowTime.end());
    uint64_t sum = 0;
    uint64_t times = 0;

    for (uint32_t i = 0; i < row - invalidLoopNum; i++) {
      sum += rowTime[i];
      times++;
    }
    bpTimeTmp.opName = taskTBEList[0][j].opName;
    bpTimeTmp.time = static_cast<uint64_t>(static_cast<double>(sum) / static_cast<double>(times));
    TUNE_DEBUG("Valid GDAT Loops[%llu] Op[%s] time[%llu]", times, bpTimeTmp.opName.c_str(), bpTimeTmp.time);
    bpTimeInfo.push_back(bpTimeTmp);
  }

  ret = CheckTimeSd(time, bpTimeInfo);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("CheckTimeSd failed, ret:%d", ret);
    return ret;
  }
  return TUNE_SUCCESS;
}

TuneResult_t GetBPTimeFromProfiling(const std::vector<std::vector<struct ProfilingMetric>> &taskProfilingList,
                                    std::vector<struct BPTimeInfo> &bpTimeInfo) {
  TuneResult_t ret;

  std::vector<std::vector<struct ProfilingMetric>> taskTBEList;

  if (taskProfilingList.size() == 0 || taskProfilingList[0].size() == 0) {
    TUNE_ERROR("taskProfilingList is invlaid para");
    return TUNE_E_PARA;
  }

  for (uint32_t k = 0; k < taskProfilingList.size(); k++) {
    if (taskProfilingList[k].size() != taskProfilingList[0].size()) {
      TUNE_INFO("taskProfilingList[%llu].size[%llu] is not same to the taskProfilingList[0].size[%llu]", k,
                taskProfilingList[k].size(), taskProfilingList[0].size());
    }
  }

  for (uint32_t i = 0; i < taskProfilingList.size(); i++) {
    std::vector<struct ProfilingMetric> taskTBENode;
    for (uint32_t j = 0; j < taskProfilingList[i].size(); j++) {
      if (taskProfilingList[0][0].modelName != taskProfilingList[i][j].modelName) {
        TUNE_ERROR("the %d_th modelName[%s] is not same to the 0_th modelName[%s]", i,
                   taskProfilingList[i][j].modelName.c_str(), taskProfilingList[0][0].modelName.c_str());
        return TUNE_E_PARA;
      }
      if ((taskProfilingList[i][j].opType == TRACE_OP_TYPE)) {
        taskTBENode.push_back(taskProfilingList[i][j]);
      }
    }
    if (taskTBENode.size() == 0) {
      TUNE_ERROR("cant not find TensorRedirect, error!");
      return TUNE_E_DATA_NOT_MATCH;
    }
    taskTBEList.push_back(taskTBENode);
  }

  ret = GetAndCheckTBETime(taskTBEList, bpTimeInfo);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("GetTBETime failed, ret:%d", ret);
    return ret;
  }

  return TUNE_SUCCESS;
}

static void GetIndexInfo(const std::vector<std::string> &vec, IndexInfo &indexInfo, uint32_t &maxIndex) {
  maxIndex = 0;

  for (uint32_t j = 0; j < vec.size(); j++) {
    if (vec[j] == "gradient_size(byte)") {
      indexInfo.gradientSizeIndex = j;
      maxIndex = j > maxIndex ? j : maxIndex;
    }
    if (vec[j] == "data_type") {
      indexInfo.dataTypeIndex = j;
      maxIndex = j > maxIndex ? j : maxIndex;
    }
    if (vec[j] == "graph_id") {
      indexInfo.graphIdIndex = j;
      maxIndex = j > maxIndex ? j : maxIndex;
    }
    if (vec[j] == "trace_node_name") {
      indexInfo.traceNodeNameIndex = j;
      maxIndex = j > maxIndex ? j : maxIndex;
    }
    if (vec[j] == "group_name") {
      indexInfo.groupNameIndex = j;
      maxIndex = j > maxIndex ? j : maxIndex;
    }
  }
  TUNE_INFO("gradientSizeIndex:%d, dataTypeIndex:%d, graphIdIndex:%d, traceNodeNameIndex:%d",
            indexInfo.gradientSizeIndex, indexInfo.dataTypeIndex, indexInfo.graphIdIndex, indexInfo.traceNodeNameIndex);
}

TuneResult_t FileStreamRead(const std::string filePath, std::vector<struct GradientNode> &graNode) {
  uint32_t i = 0;
  uint32_t maxIndex = 0;
  IndexInfo indexInfo = {};
  struct GradientNode graTmp;
  std::ifstream fileStream(filePath.c_str());
  if (fileStream.is_open()) {
    while (fileStream.peek() != EOF) {
      char buf[MAX_BUF_LEN];
      fileStream.getline(buf, MAX_BUF_LEN);

      std::string tmp = buf;
      std::vector<std::string> vec;
      std::string nodeInfo;
      std::stringstream ss;

      ss << tmp;
      while (getline(ss, nodeInfo, ',')) {
        vec.push_back(nodeInfo);
      }
      if (i == 0) {
        GetIndexInfo(vec, indexInfo, maxIndex);
      }

      if ((i > 0) && (maxIndex < vec.size()) && (vec[indexInfo.graphIdIndex] == globalGraphId_)) {
        graTmp.traceNodeName = vec[indexInfo.traceNodeNameIndex];
        graTmp.gradientSize = atol(vec[indexInfo.gradientSizeIndex].c_str());
        graTmp.dataType = vec[indexInfo.dataTypeIndex];
        graTmp.graphId = atoi(vec[indexInfo.graphIdIndex].c_str());
        graTmp.groupName = vec[indexInfo.groupNameIndex];
        graNode.push_back(graTmp);
      }
      i++;
    }
    fileStream.close();
  } else {
    TUNE_ERROR("open file:%s failed", filePath.c_str());
    return TUNE_E_OPEN_FILE_FAILURE;
  }

  return TUNE_SUCCESS;
}

TuneResult_t GetGradientInfo(const std::vector<struct GradientNode> &graNode,
                             const std::vector<struct BPTimeInfo> &bpTimeInfo,
                             std::vector<struct GradientInfo> &graInfo) {
  if (graNode.size() != bpTimeInfo.size()) {
    TUNE_ERROR("GraNode size[%llu] and BpTimeInfo size[%llu] are not match", graNode.size(), bpTimeInfo.size());
    return TUNE_E_DATA_NOT_MATCH;
  }

  std::unordered_map<std::string, GradientNode> gradientNodeMap;
  for (uint32_t k = 0; k < graNode.size(); k++) {
    gradientNodeMap.insert(std::pair<std::string, GradientNode>(graNode[k].traceNodeName, graNode[k]));
  }

  for (uint32_t k = 0; k < bpTimeInfo.size(); k++) {
    struct GradientInfo graInfoTmp;
    graInfoTmp.time = bpTimeInfo[k].time;
    graInfoTmp.opName = bpTimeInfo[k].opName;
    graInfoTmp.index = k;
    auto it = gradientNodeMap.find(graInfoTmp.opName);
    if (it != gradientNodeMap.end()) {
      graInfoTmp.gradientSize = it->second.gradientSize;
      graInfoTmp.dataType = it->second.dataType;
      graInfoTmp.graphId = it->second.graphId;
      graInfoTmp.groupName = it->second.groupName;
    } else {
      TUNE_ERROR("Can't find opName[%s] in Gradient Node", graInfoTmp.opName.c_str());
      return TUNE_E_PARA;
    }
    graInfo.push_back(graInfoTmp);
  }

  for (const auto &it : graInfo) {
    TUNE_INFO(
        "[GradientInfo] opName[%s] groupName[%s] dataType[%s] time[%llu] gradientSize[%llu] graphId[%u] index[%u]",
        it.opName.c_str(), it.groupName.c_str(), it.dataType.c_str(), it.time, it.gradientSize, it.graphId, it.index);
  }

  return TUNE_SUCCESS;
}

TuneResult_t GetGradientNode(const std::vector<std::vector<struct ProfilingMetric>> &taskProfilingList,
                             std::string workPath, std::vector<struct GradientInfo> &graInfo) {
  TuneResult_t ret;
  char realWorkPath[PATH_MAX] = {0};  // lint !e813
  std::vector<struct BPTimeInfo> bpTimeInfo;
  std::vector<struct GradientNode> graNode;
  std::string filePath;

  if ((workPath.size() > PATH_MAX) || (realpath(workPath.c_str(), realWorkPath) == nullptr)) {
    TUNE_ERROR("path_len > (PATH_MAX) or workPath[%s] is invalid", workPath.c_str());
    return TUNE_E_PARA;
  }

  ret = GetBPTimeFromProfiling(taskProfilingList, bpTimeInfo);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("GetBPTimeFromProfiling failed, ret:%d", ret);
    return TUNE_E_PARA;
  }
  filePath = std::string(realWorkPath) + "/" + "gradient_summary.csv";
  ret = FileStreamRead(filePath, graNode);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("FileStreamRead failed, ret:%d", ret);
    return TUNE_E_PARA;
  }
  ret = GetGradientInfo(graNode, bpTimeInfo, graInfo);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("GetGradientInfo failed, ret:%d", ret);
    return TUNE_E_PARA;
  }
  return TUNE_SUCCESS;
}

TuneResult_t GetSocVersion(std::string &configVersion) {
  uint32_t ret = 0;
  fe::PlatformInfoManager &configInst = fe::PlatformInfoManager::Instance();
  fe::PlatformInfo autoPlatInfo;
  fe::OptionalInfo autoOptinalInfo;
  ret = configInst.GetPlatformInfoWithOutSocVersion(autoPlatInfo, autoOptinalInfo);
  if (ret == 0) {
    configVersion = autoOptinalInfo.soc_version;
    TUNE_INFO("[GradientFusionTune]get soc version success, [%s], return [%d]", configVersion.c_str(), ret);
  } else {
    TUNE_ERROR("[GradientFusionTune]get soc version failed.");
  }
  return TUNE_SUCCESS;
}

TuneResult_t GetTuneFusionPath(std::string &fusionPath) {
  TuneResult_t ret;
  std::string autoFileName;
  std::string autoConfigVersion;
  ret = GetSocVersion(autoConfigVersion);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("[GradientFusionTune]GetSocVersion failed, ret %d", ret);
    return TUNE_E_PARA;
  }
  autoFileName = autoConfigVersion + "_gradient_fusion.json";

  std::string autoLibPath;
  char *getEnvPath = nullptr;
  MM_SYS_GET_ENV(MM_ENV_TUNE_BANK_PATH, getEnvPath);
  if (getEnvPath != nullptr) {
    autoLibPath = getEnvPath;
    fusionPath = autoLibPath + "/" + autoFileName;
  } else {
    TUNE_WARN("TUNE ENV:TUNE_BANK_PATH is not set, use Default Path.");
    char *getDefPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HOME, getDefPath);
    if (getDefPath != nullptr) {
      autoLibPath = getDefPath;
      autoLibPath = autoLibPath + "/Ascend/latest/data/aoe/custom/graph/" + autoConfigVersion;
      fusionPath = autoLibPath + "/" + autoFileName;
    } else {
      TUNE_ERROR("[GradientFusionTune]find fusion library path failed.");
      return TUNE_E_OPEN_FILE_FAILURE;
    }
  }
  char realFile[PATH_MAX] = {0};
  if (realpath(fusionPath.c_str(), realFile) == nullptr) {
    TUNE_ERROR("[GradientFusionTune][fusionPath]path %s is not a valid real path", fusionPath.c_str());
    return TUNE_E_PARA;
  }

  return TUNE_SUCCESS;
}

TuneResult_t HcomGradientFusionTune(const std::vector<std::vector<struct ProfilingMetric>> &taskProfilingList,
                                    std::string workPath) {
  TUNE_INFO("Start gradient auto tuning.");
  for (size_t i = 0; i < taskProfilingList.size(); i++) {
    for (size_t j = 0; j < taskProfilingList[0].size(); j++) {
      TUNE_DEBUG(
          "[GradientFusionTune][ProfilingInfo] Row[%zu] Col[%zu] modelName[%s] opType[%s] \
                       opName[%s] taskStart time[%llu] taskEnd time[%llu]",
          i, j, taskProfilingList[i][j].modelName.c_str(), taskProfilingList[i][j].opType.c_str(),
          taskProfilingList[i][j].opName.c_str(), taskProfilingList[i][j].taskStartTime,
          taskProfilingList[i][j].taskEndTime);
    }
  }
  globalGraphId_ = taskProfilingList[0][0].modelName;

  TuneResult_t ret;
  std::vector<struct GradientInfo> graInfo;
  std::vector<uint64_t> gradientTimeTr;
  std::vector<uint64_t> gradientSizeTr;

  // 获取value中用到的值
  ret = GetGradientNode(taskProfilingList, workPath, graInfo);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("[GradientFusionTune]GetGradientNode fail, ret %d", ret);
    return TUNE_E_PARA;
  }

  // 获取梯度数据量和梯度计算耗时
  int graInfoSize = graInfo.size();
  for (auto i = 0; i < graInfoSize; i++) {
    gradientTimeTr.push_back(graInfo[i].time);
    gradientSizeTr.push_back(graInfo[i].gradientSize);
  }

  // 获取知识库的位置
  std::string configPath;
  ret = GetTuneFusionPath(configPath);
  if (ret != TUNE_SUCCESS) {
    TUNE_ERROR("[GradientFusionTune]GetTuneFusionPath fail, ret %d", ret);
    return TUNE_E_PARA;
  }

  std::fstream jFile;
  jFile.open(configPath, std::ios::in);
  nlohmann::json modelRoot;

  try {
    jFile >> modelRoot;  // 将文件内容读取到json对象内
  } catch (...) {
    TUNE_ERROR("[Read][File]load file[%s] to json fail. please check json file!", configPath.c_str());
    jFile.close();
    return TUNE_E_OPEN_FILE_FAILURE;
  }
  jFile.close();

  // default value
  std::vector<uint64_t> defaultvalue{0, 0, 0, 0, 0};
  int modelNum = modelRoot.size();

  for (auto i = 0; i < modelNum; i++) {
    if (modelRoot[i]["modelvalue"]["gradientTime"] == defaultvalue ||
        modelRoot[i]["modelvalue"]["gradientSize"] == defaultvalue) {
      modelRoot[i]["modelvalue"]["gradientTime"] = gradientTimeTr;
      modelRoot[i]["modelvalue"]["gradientSize"] = gradientSizeTr;
    }
  }

  jFile.open(configPath, std::ios::out | std::ios::trunc);
  jFile << modelRoot;
  jFile.close();

  TUNE_INFO("[GradientFusionTune]Fusion tune run success.");

  return TUNE_SUCCESS;
}

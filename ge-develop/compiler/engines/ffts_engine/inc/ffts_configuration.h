/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_INC_FFTS_CONFIGURATION_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_INC_FFTS_CONFIGURATION_H_

#include <map>
#include <string>
#include <mutex>
#include <vector>
#include "inc/ffts_error_codes.h"
namespace ffts {
/** @brief Configuration.
* Used to manage all the configuration data within the fusion engine module. */
class Configuration {
 public:
  Configuration(const Configuration &) = delete;
  Configuration &operator=(const Configuration &) = delete;

  static Configuration &Instance();

  /**
   * Initialize the content_map
   * Read the content from the config file, the save the data into content_map
   * @return Whether the object has been initialized successfully.
   */
  Status Initialize();

  Status Finalize();

  std::string GetFftsOptimizeEnable();
  
  uint32_t GetFftsMergeLimitHBMRatio();

  void SetFftsOptimizeEnable();

  size_t GetInnerCmoNodeDistTh() const;

  size_t GetWriteBackDataSizeLowerTh() const;

  size_t GetWriteBackDataSizeHigherTh() const;

  size_t GetInnerPrefetchDataSizeLowerTh() const;

  size_t GetInnerPrefetchDataSizeUpperTh() const;

  size_t GetOuterPrefetchDataSizeLowerTh() const;

  size_t GetOuterPrefetchDataSizeUpperTh() const;

  size_t GetPrefetchConsumerCnt() const;

  size_t GetPrefetchOstNum() const;

  size_t GetCmaintOstNum() const;

  size_t GetAicPrefetchLower() const;

  size_t GetAicPrefetchUpper() const;

  size_t GetAivPrefetchLower() const;

  size_t GetAivPrefetchUpper() const;

  size_t GetDataSplitUnit() const;
 private:
  Configuration();
  ~Configuration();
  static const std::string config_file_relative_path_;
  static const std::string config_ops_relative_path_;
  bool is_init_;
  std::map<std::string, std::string> content_map_;
  std::string lib_path_;
  std::mutex mode_lock_;

  std::string runmode_;
  uint32_t hbm_ratio_;
  size_t inner_cmo_node_distance_threshold_;

  size_t inner_write_back_data_size_lower_threshold_;

  size_t inner_write_back_data_size_higher_threshold_;

  size_t inner_prefetch_data_size_lower_threshold_;

  size_t inner_prefetch_data_size_upper_threshold_;

  size_t outer_prefetch_data_size_lower_threshold_;

  size_t outer_prefetch_data_size_upper_threshold_;

  size_t prefetch_consumer_cnt_;

  size_t prefetch_ost_num_;
  size_t cmaint_ost_num_;

  size_t aic_prefetch_lower_;
  size_t aic_prefetch_upper_;

  size_t aiv_prefetch_lower_;
  size_t aiv_prefetch_upper_;

  size_t data_split_unit_;

  Status InitSingleConfig(const std::string &key, size_t &value) const;
  /**
   * Get the real Path of current so lib
   */
  Status InitLibPath();

  /**
   * Read the content of configuration file(FE_CONFIG_FILE_PATH)
   * Save the data into content_map
   * @return Whether the config file has been loaded successfully.
   */
  Status LoadConfigFile();

  Status InitCMOThreshold();
  /**
   * Get the value from the content_map if the content_map contains the input key.
   * @param key config key
   * @param return_value output value. if the value has been found,
   *                    return_value will refer to this value.
   * @return Whether the vale has been found.
   *         Status SUCCESS:found, FAILED:not found
   */
  Status GetStringValue(const std::string &key, std::string &return_value) const;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_INC_FFTS_CONFIGURATION_H_

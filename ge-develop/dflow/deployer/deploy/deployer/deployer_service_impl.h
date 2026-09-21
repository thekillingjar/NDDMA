/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_DAEMON_DEPLOYER_SERVICE_IMPL_H_
#define AIR_RUNTIME_DEPLOY_DAEMON_DEPLOYER_SERVICE_IMPL_H_

#include <map>
#include <string>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "common/config/json_parser.h"
#include "deploy/deployer/deploy_context.h"
#include "proto/deployer.pb.h"

namespace ge {
class DeployerServiceImpl {
 public:
  static DeployerServiceImpl &GetInstance();
  /*
   *  @ingroup ge
   *  @brief   deployer service interface
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return  SUCCESS or FAILED
   */
  Status Process(DeployContext &context,
                 const deployer::DeployerRequest &request,
                 deployer::DeployerResponse &response);

  /*
   *  @ingroup ge
   *  @brief   deployer service interface
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return  SUCCESS or FAILED
   */
  using ProcessFunc = std::function<void(DeployContext &context,
                                         const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response)>;

  /*
  *  @ingroup ge
  *  @brief   register deployer server api
  *  @param   [in]  DeployerRequestType
  *  @param   [in]  const ProcessFunc &
  *  @return: None
   *
  */
  void RegisterReqProcessor(deployer::DeployerRequestType type, const ProcessFunc &fn);

  static void UpdateDeployPlanProcess(DeployContext &context,
                                      const deployer::DeployerRequest &request,
                                      deployer::DeployerResponse &response);

  static void FlowRoutePlanProcess(DeployContext &context,
                                   const deployer::DeployerRequest &request,
                                   deployer::DeployerResponse &response);

  /*
   *  @ingroup ge
   *  @brief   download device maintenance config
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return: None
   */
  static void DownloadDevMaintenanceCfgProcess(DeployContext &context,
                                               const deployer::DeployerRequest &request,
                                               deployer::DeployerResponse &response);
  /*
   *  @ingroup ge
   *  @brief   download model
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return: None
   */
  static void DownloadModelProcess(DeployContext &context,
                                   const deployer::DeployerRequest &request,
                                   deployer::DeployerResponse &response);

  /*
   *  @ingroup ge
   *  @brief   load model
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return: None
   */
  static void LoadModelProcess(DeployContext &context,
                               const deployer::DeployerRequest &request,
                               deployer::DeployerResponse &response);

  /*
   *  @ingroup ge
   *  @brief   unload model
   *  @param   [in]  context        grpc context
   *  @param   [in]  request        service request
   *  @param   [out] response       service response
   *  @return: None
   */
  static void UnloadModelProcess(DeployContext &context,
                                 const deployer::DeployerRequest &request,
                                 deployer::DeployerResponse &response);

  static void MultiVarManagerInfoProcess(DeployContext &context,
                                         const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response);

  static void SharedContentProcess(DeployContext &context,
                                   const deployer::DeployerRequest &request,
                                   deployer::DeployerResponse &response);

  static void TransferFileProcess(DeployContext &context,
                                  const deployer::DeployerRequest &request,
                                  deployer::DeployerResponse &response);

  static void InitProcessResourceProcess(DeployContext &context, const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response);

  static void HeartbeatProcess(DeployContext &context,
                               const deployer::DeployerRequest &request,
                               deployer::DeployerResponse &response);

  static void DataGwSchedInfo(DeployContext &context,
                              const deployer::DeployerRequest &request,
                              deployer::DeployerResponse &response);
  static void ClearModelRunningData(DeployContext &context,
                                    const deployer::DeployerRequest &request,
                                    deployer::DeployerResponse &response);
  static void DataFlowExceptionNotifyProcess(DeployContext &context, const deployer::DeployerRequest &request,
                                             deployer::DeployerResponse &response);

  static void UpdateProfilingInfoProcess(DeployContext &context,
                                         const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response);

 private:
  std::map<deployer::DeployerRequestType, ProcessFunc> process_fns_;
  std::mutex mu_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_DAEMON_DEPLOYER_SERVICE_IMPL_H_

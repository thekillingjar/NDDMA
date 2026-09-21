/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <fstream>
#define protected public
#define private public
#include "inc/ffts_log.h"
#include "inc/ffts_utils.h"
#include "platform/platform_info.h"
#undef private
#undef protected

using namespace std;

namespace ffts {
void init_platform_info()
{
    FFTS_LOGI("Begin to init PlatformInfo.");
    Dl_info dl_info;
    if (dladdr((void*) init_platform_info, &dl_info) == 0)
    {
        FFTS_LOGE("Failed to read the so file path when initialize the PlatformInfo!");
        return;
    }
    else
    {
        string so_path = dl_info.dli_fname;
        FFTS_LOGI("PlatformInfo so path=%s", so_path.c_str());

        string so_file_path = RealPath(so_path);
        string real_dir_file_path = so_file_path.substr(0, so_file_path.rfind('/') + 1) + "../ut/obj/data/platform_config";
        FFTS_LOGI("realDirFilePath is %s.", real_dir_file_path.c_str());
        string command = "mkdir -p " + real_dir_file_path;
        system((char*) command.c_str());
        FFTS_LOGI("The PlatformInfo file directory has been created.");
        string config_file_path = real_dir_file_path + "/Ascend310.ini";
        FFTS_LOGI("Begin to write into the PlatformInfo file %s", config_file_path.c_str());
        ofstream ofs;
        ofs.open(config_file_path, ios::out);
        if (ofs)
        {
            FFTS_LOGI("The PlatformInfo file has been opened!");
        }

        ofs << "[version]" << endl;
        ofs << "SoC_version=Ascend310" << endl;
        ofs << "AIC_version=AIC-M-100" << endl;
        ofs << "CCEC_AIC_version=-dav-m100" << endl;
        ofs << "CCEC_AIV_version=-dav-m200-vec" << endl;
        ofs << "Compiler_aicpu_support_os=true" << endl;
        ofs << "[SoCInfo]" << endl;
        ofs << "aiCoreCnt=2" << endl;
        ofs << "vectorCoreCnt=0" << endl;
        ofs << "aiCpuCnt=4" << endl;
        ofs << "memoryType=0" << endl;
        ofs << "memorySize=65536" << endl;
        ofs << "l2Type=1" << endl;
        ofs << "l2Size=8388608" << endl;
        ofs << "l2PageNum=64" << endl;
        ofs << "[AICoreSpec]" << endl;
        ofs << "cubeFreq=680" << endl;
        ofs << "cubeMSize=16" << endl;
        ofs << "cubeNSize=16" << endl;
        ofs << "cubeKSize=16" << endl;
        ofs << "vecCalcSize=256" << endl;
        ofs << "l0ASize=65536" << endl;
        ofs << "l0BSize=65536" << endl;
        ofs << "l0CSize=262144" << endl;
        ofs << "l1Size=1048576" << endl;
        ofs << "ubSize=253952" << endl;
        ofs << "smaskBuffer=256" << endl;
        ofs << "ubblockSize=32" << endl;
        ofs << "ubbankSize=4096" << endl;
        ofs << "ubbankNum=64" << endl;
        ofs << "ubburstInOneBlock=32" << endl;
        ofs << "ubbankGroupNum=16" << endl;
        ofs << "[AICoreMemoryRates]" << endl;
        ofs << "ddrRate=67" << endl;
        ofs << "l2Rate=192" << endl;
        ofs << "l2ReadRate=128" << endl;
        ofs << "l2WriteRate=64" << endl;
        ofs << "l1ToL0ARate=512" << endl;
        ofs << "l1ToL0BRate=256" << endl;
        ofs << "l1ToUBRate=128" << endl;
        ofs << "l0CToUBRate=128" << endl;
        ofs << "ubToL2Rate=1234" << endl;
        ofs << "ubToDdrRate=21474836480/680000000" << endl;
        ofs << "ubToL1Rate=512" << endl;
        ofs << "[AICoreintrinsicDtypeMap]" << endl;
        ofs << "0=Intrinsic_vrec|float16,float32" << endl;
        ofs << "1=Intrinsic_vadd|float16,float32,int32 " << endl;
        ofs << "2=Intrinsic_vadds|float16,float32" << endl;
        ofs << "3=Intrinsic_vrsqrt|float16,float32" << endl;
        ofs << "4=Intrinsic_vsub|float16,float32,int32" << endl;
        ofs << "5=Intrinsic_vmul|float16,float32,int32" << endl;
        ofs << "6=Intrinsic_vmax|float16,float32,int32" << endl;
        ofs << "7=Intrinsic_vmaxs|float16,float32,int32" << endl;
        ofs << "8=Intrinsic_vmin|float16,float32,int32" << endl;
        ofs << "[VectorCoreSpec]" << endl;
        ofs << "vecCalcSize=1" << endl;
        ofs << "smaskBuffer=0" << endl;
        ofs << "ubSize=2" << endl;
        ofs << "ubblockSize=3" << endl;
        ofs << "ubbankSize=4" << endl;
        ofs << "ubbankNum=5" << endl;
        ofs << "ubburstInOneBlock=6" << endl;
        ofs << "ubbankGroupNum=7" << endl;

        ofs << "[VectorCoreMemoryRates]" << endl;
        ofs << "ddrRate=1" << endl;
        ofs << "l2Rate=2" << endl;
        ofs << "l2ReadRate=3" << endl;
        ofs << "l2WriteRate=4" << endl;
        ofs << "ubToL2Rate=5" << endl;
        ofs << "ubToDdrRate=6" << endl;

        ofs << "[VectorCoreintrinsicDtypeMap]" << endl;
        ofs << "0=Intrinsic_vrec|float16,float32" << endl;
        ofs << "1=Intrinsic_vadd|float16,float32,int32 " << endl;
        ofs.close();
        FFTS_LOGI("Finish writing the PlatformInfo file %s", config_file_path.c_str());
    }
}
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    ffts::init_platform_info();
    int ret = RUN_ALL_TESTS();
    return ret;
}


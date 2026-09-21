#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)

cd ${SCRIPT_DIR}

# print usage message
usage() {
    echo "Usage:"
    echo "  sh test_ascir.sh [-h | --help] [--mode=<mode>] [--path=<PATH>] [--case=<case_name>] [--prof] [--model]"
    echo ""
    echo "Options:"
    echo "    -h, --help        Print usage"
    echo "    --mode=<mode>"
    echo "                  test mode."
    echo "                   0: default, test case: run input_ascir.py to generate code then compile, launch kernel, check output"
    echo "                   1: compile and launch, need kernel and tiling code, then compile and launch kernel, check output"
    echo "                   2: only compile, need kernel and tiling code, then compile"
    echo "    --path=<PATH>"
    echo "                  input path, test case path or code path"
    echo "                   E.g.: ./testcase/two_add ; /npu/testcase/kernel_meta_156083/te_ascbackend_0051f9c1_ukasmy_"
    echo "    --case=<PATH>"
    echo "                  test case name, existing test case dir name(in ./testcase)"
    echo "                   E.g.: two_add"
    echo "    --prof"
    echo "    --model"
    echo ""
}

# parse and set options
checkopts() {
    # Process the options
    parsed_args=$(getopt -a -o h -l help,prof,model,mode:,path:,case: -- "$@") || {
        usage
        exit 1
    }

    eval set -- "$parsed_args"
    # 初始化变量
    MODE=0
    ENABLE_PROF=0
    ENABLE_MODEL=0
    CASE=
    TEST_CASE_PATH=


    while true; do
        case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --path)
            TEST_CASE_PATH="$(realpath $2)"
            shift 2
            ;;
        --case)
            CASE="$2"
            shift 2
            ;;
        --prof)
            ENABLE_PROF=1
            shift
            ;;
        --model)
            ENABLE_MODEL=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Undefined option: $1"
            usage
            exit 1
            ;;
        esac
    done

    # 检查--case和--path是否互斥且至少提供一个
    if [ -n "$CASE" ] && [ -n "$TEST_CASE_PATH" ]; then
        echo "Error: --case and --path are mutually exclusive. Provide only one."
        usage
        exit 1
    elif [ -z "$CASE" ] && [ -z "$TEST_CASE_PATH" ]; then
        echo "Error: Either --case or --path must be provided."
        usage
        exit 1
    fi

    if [ "$ENABLE_PROF" == "1" ] && [ "$ENABLE_MODEL" == "1" ]; then
        echo "Error: --prof and --model are mutually exclusive. Provide only one."
        usage
        exit 1
    fi

    if ! [[ "$MODE" =~ ^[0-9]+$ ]]; then
        echo "Error: --mode must be a number."
        usage
        exit 1
    fi
}

run_test_case() {
    cd ${SCRIPT_DIR}
    cp ${org_graph_file} ${SCRIPT_DIR}/py/input_ascir.py.bak
    cp ${graph_file} ${org_graph_file}

    echo "you test configfile is $config_file"
    echo "you test inputfile is $input_file"
    echo "you test graphfile is $graph_file"

    # 0. get graph_name from json
    json=$(cat $config_file)
    graph_name=$(python3 -c "import json,sys;obj=json.load(sys.stdin);print(obj['kernel_config']['graph_name'])" <<< "$json")
    output_type=$(python3 -c "import json,sys;obj=json.load(sys.stdin);print(obj['kernel_config']['output_data_types'])" <<< "$json")
    echo "graph_name:${graph_name}"
    echo "output_type:${output_type}"

    # 1. modify py/test_ascir.py to replace ascir, and then codegen
    echo "################# begin to run autofuse and codegen."
    rm -rf *.so
    python3 py/test_ascir.py --mode=0 --graph_name=${graph_name} --output_path=${SCRIPT_DIR}
    if [ "$?" != 0 ]; then
        echo "################# fail to run autofuse and codegen."
        mv ${SCRIPT_DIR}/py/input_ascir.py.bak ${org_graph_file}
        exit 1
    fi
    mv ${SCRIPT_DIR}/py/input_ascir.py.bak ${org_graph_file}
    echo "################# end to run autofuse and codegen."

    # 2. compile tool
    echo "################# begin to compile tool."
    cd src
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    echo "################# end to compile tool."

    # 3. generate input
    echo "################# begin to generate input."
    cd ${SCRIPT_DIR}/input/
    python3 ${input_file}
    if [ "$?" != 0 ]; then
        echo "################# fail to generate input."
        exit 1
    fi
    echo "################# end to generate input."

    # 4. launch kernel
    echo "################# begin to launch kernel."
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    cd ${SCRIPT_DIR}
    rm -rf out/Output_*.bin
    if [ "$ENABLE_PROF" == "1" ]; then
        msprof --application="src/build/bin/fused_graph_test ${config_file}" --task-time=on --ai-core=on --output=out
    elif [ "$ENABLE_MODEL" == "1" ]; then
        msopprof simulator --application="src/build/bin/fused_graph_test ${config_file}" --output=out
    else
        src/build/bin/fused_graph_test ${config_file}
    fi
    if [ "$?" != 0 ]; then
        echo "################# fail to launch kernel."
        exit 1
    fi
    echo "################# end to launch kernel."

    # 5. compare
    echo "################# output to ${SCRIPT_DIR}/out, begin to compare"
    output_type_array=(${output_type//,/ })
    python3 out/verify_result.py out/Output_0.bin out/golden.bin ${testcase} ${output_type_array[0]}
    if [ -f "out/Output_1.bin" ]; then
        python3 out/verify_result.py out/Output_1.bin out/golden_1.bin ${testcase} ${output_type_array[1]}
    fi
    if [ -f "out/Output_2.bin" ]; then
        python3 out/verify_result.py out/Output_2.bin out/golden_2.bin ${testcase} ${output_type_array[2]}
    fi
    if [ "$ENABLE_PROF" == "1" ]; then
        python3 py/perf_summary.py
    fi
    current_time=$(date "+%Y-%m-%d_%H_%M_%S")
    mkdir -p "kernel_code"
    mv tmp_ascir kernel_code/tmp_ascir_${testcase}_$current_time
}

test_mode_0() {
    clear
    rm -rf out/*.bin out/P*
    if [ -n "$CASE" ]; then
        echo "you select testcase: $CASE"
        org_graph_file="${SCRIPT_DIR}/py/input_ascir.py"
        config_file="${SCRIPT_DIR}/testcase/${CASE}/ascir.json"
        input_file="${SCRIPT_DIR}/testcase/${CASE}/gen_input.py"
        graph_file="${SCRIPT_DIR}/testcase/${CASE}/input_ascir.py"
        testcase="${CASE}"
        run_test_case
        cp ${SCRIPT_DIR}/testcase/${CASE}/ascir.json ${SCRIPT_DIR}/config/
        cp ${SCRIPT_DIR}/testcase/${CASE}/gen_input.py ${SCRIPT_DIR}/input/
    elif [ -n "$TEST_CASE_PATH" ]; then
        echo "testcase path: $TEST_CASE_PATH"
        org_graph_file="${SCRIPT_DIR}/py/input_ascir.py"
        config_file="${TEST_CASE_PATH}/ascir.json"
        input_file="${TEST_CASE_PATH}/gen_input.py"
        graph_file="${TEST_CASE_PATH}/input_ascir.py"
        testcase=$(basename $TEST_CASE_PATH)
        run_test_case
        cp ${TEST_CASE_PATH}/ascir.json ${SCRIPT_DIR}/config/
        cp ${TEST_CASE_PATH}/gen_input.py ${SCRIPT_DIR}/input/
    fi
}

test_mode_1() {
    cd ${SCRIPT_DIR}
    config_file="${SCRIPT_DIR}/config/ascir.json"
    input_file="${SCRIPT_DIR}/input/gen_input.py"
    echo "you test configfile is $config_file"
    echo "you test inputfile is $input_file"

    # 0. get graph_name from json
    json=$(cat $config_file)
    graph_name=$(python3 -c "import json,sys;obj=json.load(sys.stdin);print(obj['kernel_config']['graph_name'])" <<< "$json")
    output_type=$(python3 -c "import json,sys;obj=json.load(sys.stdin);print(obj['kernel_config']['output_data_types'])" <<< "$json")
    echo "graph_name:${graph_name}"
    echo "output_type:${output_type}"

    rm -rf *.so
    python3 py/test_ascir.py --mode=1 --graph_name=${graph_name} --output_path=${SCRIPT_DIR} --code_path=${TEST_CASE_PATH}
    if [ "$?" != 0 ]; then
        echo "################# fail to compile."
        exit 1
    fi

    # 2. compile tool
    echo "################# begin to compile tool."
    cd src
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    echo "################# end to compile tool."

    # 3. generate input
    echo "################# begin to generate input."
    cd ${SCRIPT_DIR}/input/
    python3 ${input_file}
    if [ "$?" != 0 ]; then
        echo "################# fail to generate input."
        exit 1
    fi
    echo "################# end to generate input."

    # 4. launch kernel
    echo "################# begin to launch kernel."
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    cd ${SCRIPT_DIR}
    rm -rf out/Output_*.bin
    if [ "$ENABLE_PROF" == "1" ]; then
        msprof --application="src/build/bin/fused_graph_test ${config_file}" --output=out
    elif [ "$ENABLE_MODEL" == "1" ]; then
        msopprof simulator --application="src/build/bin/fused_graph_test ${config_file}" --output=out
    else
        src/build/bin/fused_graph_test ${config_file}
    fi
    if [ "$?" != 0 ]; then
        echo "################# fail to launch kernel."
        exit 1
    fi
    echo "################# end to launch kernel."

    # 5. compare
    echo "################# output to ${SCRIPT_DIR}/out, begin to compare"
    output_type_array=(${output_type//,/ })
    python3 out/verify_result.py out/Output_0.bin out/golden.bin case_nogen ${output_type_array[0]}
    if [ -f "out/Output_1.bin" ]; then
        python3 out/verify_result.py out/Output_1.bin out/golden_1.bin case_nogen ${output_type_array[1]}
    fi
    if [ -f "out/Output_2.bin" ]; then
        python3 out/verify_result.py out/Output_2.bin out/golden_2.bin case_nogen ${output_type_array[2]}
    fi
    if [ "$ENABLE_PROF" == "1" ]; then
        python3 py/perf_summary.py
    fi
}

test_mode_2() {
    cd ${SCRIPT_DIR}
    rm -rf *.so
    python3 py/test_ascir.py --mode=1 --output_path=${SCRIPT_DIR} --code_path=${TEST_CASE_PATH}
    echo "################# compile finish."
}

main() {
    cd "${BASEPATH}"
    checkopts "$@"

    case ${MODE} in
        0)
            echo "test mode 0"
            test_mode_0
            ;;
        1)
            echo "test mode 1"
            test_mode_1
            ;;
        2)
            echo "test mode 2"
            test_mode_2
            ;;
        *)
            echo "use default mode 0"
            test_mode_0
            ;;
    esac

    echo "---------------- test finished ----------------"
}


main "$@"
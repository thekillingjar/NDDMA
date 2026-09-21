#include <cmath>
#include <random>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/logical.h"

using namespace AscendC;

template <typename InT>
struct TensorLogicalAndInputParam {
    uint8_t *y{};
    uint8_t *exp{};
    InT *src0{};
    InT *src1{};
    int32_t size{0};
};

class TestApiLogicalAnd :public testing::Test {
    protected:
    template <typename InT>
    static void InvokeKernel(TensorLogicalAndInputParam<InT> &param) {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
        tpipe.InitBuffer(x1buf, sizeof(InT) * param.size);
        tpipe.InitBuffer(x2buf, sizeof(InT) * param.size);
        tpipe.InitBuffer(ybuf, sizeof(uint8_t) * param.size);
        LocalTensor<InT> l_x1 = x1buf.Get<InT>();
        LocalTensor<InT> l_x2 = x2buf.Get<InT>();
        LocalTensor<uint8_t> l_y = ybuf.Get<uint8_t>();

        GmToUb(l_x1, param.src0, param.size);
        GmToUb(l_x2, param.src1, param.size);
        LogicalAndExtend<InT>(l_y, l_x1, l_x2, param.size);
        UbToGm(param.y, l_y, param.size);
    }

    template <typename InT>
    static void CreateTensorInput(TensorLogicalAndInputParam<InT> &param) {
        // 构造测试输入和预期结果
        param.y = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));
        param.exp = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));
        param.src0 = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));
        param.src1 = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));

        std::mt19937 eng(1);
        
        auto distr = []() {
              if constexpr (std::is_integral_v<InT>) {
                  return std::uniform_int_distribution<int32_t>(-10, 10);
              } else {
                  return std::uniform_real_distribution<float>(-10.0f, 10.0f);
              }
          }();
        // std::uniform_int_distribution distr(0, input_range);  // Define the range
        // std::uniform_real_distribution<float> distr(-10.0f, static_cast<float>(input_range));

        for (int i = 0; i < param.size; i++) {
            InT input0 = static_cast<InT>(distr(eng));
            InT input1 = static_cast<InT>(distr(eng));
            param.src0[i] = input0;
            param.src1[i] = input1;
            param.exp[i] = static_cast<uint8_t>((param.src0[i] == static_cast<InT>(0) ? false : true) &&
                           (param.src1[i] == static_cast<InT>(0) ? false : true));
        }
    }

    static uint32_t Valid(uint8_t *y, uint8_t *exp, size_t comp_size) {
        uint32_t diff_count = 0;
        for (uint32_t i = 0; i < comp_size; i++) {
            if (y[i] != exp[i]) {
                diff_count++;
            }
        }
        return diff_count;
    }

    template <typename InT>
    static void LogicalAndTest(const int32_t size) {
        TensorLogicalAndInputParam<InT> param{};
        param.size = size;
        CreateTensorInput(param);

        // 构造Api调用函数
        auto kernel = [&param] { InvokeKernel(param); };

        // 调用kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(kernel, 1);

        // 验证结果
        uint32_t diff_count = Valid(param.y, param.exp, param.size);
        EXPECT_EQ(diff_count, 0);
    }
};

TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Half) {
    LogicalAndTest<half>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Float) {
    LogicalAndTest<float>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Uint8) {
    LogicalAndTest<uint8_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Int16) {
    LogicalAndTest<int16_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Int32) {
    LogicalAndTest<int32_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Int8) {
    LogicalAndTest<int8_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Int64) {
    LogicalAndTest<int64_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Uint32) {
    LogicalAndTest<uint32_t>(256);
}
TEST_F(TestApiLogicalAnd, LogicalAnd_Test_Bf16) {
    LogicalAndTest<bfloat16_t>(256);
}
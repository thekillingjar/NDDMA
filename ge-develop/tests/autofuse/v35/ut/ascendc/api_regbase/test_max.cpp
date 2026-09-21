#include <cmath>
#include <random>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"

using namespace AscendC;

template <typename InT>
struct TensorMaxInputParam {
    InT *y{};
    InT *exp{};
    InT *src0{};
    InT *src1{};
    int32_t size{0};
};

class TestApiMax :public testing::Test {
    protected:
    template <typename InT>
    static void InvokeKernel(TensorMaxInputParam<InT> &param) {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
        tpipe.InitBuffer(x1buf, sizeof(InT) * param.size);
        tpipe.InitBuffer(x2buf, sizeof(InT) * param.size);
        tpipe.InitBuffer(ybuf, sizeof(InT) * param.size);
        LocalTensor<InT> l_x1 = x1buf.Get<InT>();
        LocalTensor<InT> l_x2 = x2buf.Get<InT>();
        LocalTensor<InT> l_y = ybuf.Get<InT>();

        GmToUb(l_x1, param.src0, param.size);
        GmToUb(l_x2, param.src1, param.size);
        AscendC::Max<InT>(l_y, l_x1, l_x2, param.size);
        UbToGm(param.y, l_y, param.size);
    }

    template <typename InT>
    static void CreateTensorInput(TensorMaxInputParam<InT> &param) {
        // 构造测试输入和预期结果
        param.y = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));
        param.exp = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));
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
            param.exp[i] = std::max(param.src0[i], param.src1[i]);
        }
    }

    template <typename InT>
    static uint32_t Valid(InT *y, InT *exp, size_t comp_size) {
        uint32_t diff_count = 0;
        for (uint32_t i = 0; i < comp_size; i++) {
            if (y[i] != exp[i]) {
                diff_count++;
            }
        }
        return diff_count;
    }

    template <typename InT>
    static void MaxTest(const int32_t size) {
        TensorMaxInputParam<InT> param{};
        param.size = size;
        CreateTensorInput(param);

        // 构造Api调用函数
        auto kernel = [&param] { InvokeKernel(param); };

        // 调用kernel
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(kernel, 1);

        // 验证结果
        uint32_t diff_count = Valid<InT>(param.y, param.exp, param.size);
        EXPECT_EQ(diff_count, 0);
    }
};

TEST_F(TestApiMax, Max_Test_Int16) {
    MaxTest<int16_t>(256);
}
TEST_F(TestApiMax, Max_Test_Int32) {
    MaxTest<int32_t>(256);
}
TEST_F(TestApiMax, Max_Test_Half) {
    MaxTest<half>(256);
}
TEST_F(TestApiMax, Max_Test_Float) {
    MaxTest<float>(256);
}
TEST_F(TestApiMax, Max_Test_Uint8) {
    MaxTest<uint8_t>(256);
}
TEST_F(TestApiMax, Max_Test_Int8) {
    MaxTest<int8_t>(256);
}
TEST_F(TestApiMax, Max_Test_Int64) {
    MaxTest<int64_t>(256);
}
TEST_F(TestApiMax, Max_Test_Bf16) {
    MaxTest<bfloat16_t>(256);
}
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  // init the logging
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  printf("finish ut\n");

  return ret;
}

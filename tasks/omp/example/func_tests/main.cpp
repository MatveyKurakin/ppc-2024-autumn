// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "omp/example/include/ops_omp.hpp"

TEST(Parallel_Operations_OpenMP, Test_const_0) {
  kurakin_m_monte_carlo_omp::Integral integral{
      .func_ = [](std::vector<double> x) { return 0.; }, .bounds_ = {{-10, 10}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_omp::TestOMPTaskParallel testOmpTaskParallel(taskDataSeq);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  ASSERT_EQ(0, res[0]);
}

TEST(Parallel_Operations_OpenMP, Test_const_10) {
  kurakin_m_monte_carlo_omp::Integral integral{
      .func_ = [](std::vector<double> x) { return 10.; }, .bounds_ = {{-10, 10}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_omp::TestOMPTaskParallel testOmpTaskParallel(taskDataSeq);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  ASSERT_EQ(200, res[0]);
}

TEST(Parallel_Operations_OpenMP, Test_dimension_1) {
  kurakin_m_monte_carlo_omp::Integral integral{
      .func_ = [](std::vector<double> x) { return std::sin(x[0]); }, .bounds_ = {{-10, 10}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_omp::TestOMPTaskParallel testOmpTaskParallel(taskDataSeq);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  ASSERT_NEAR(0, res[0], 0.3);
}

TEST(Parallel_Operations_OpenMP, Test_dimension_2) {
  kurakin_m_monte_carlo_omp::Integral integral{
      .func_ = [](std::vector<double> x) { return std::log(x[0] + x[1]) * cos(x[0] * x[1]); },
      .bounds_ = {{1, 2}, {1, 3}},
      .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_omp::TestOMPTaskParallel testOmpTaskParallel(taskDataSeq);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  ASSERT_NEAR(-1.36, res[0], 0.3);
}

TEST(Parallel_Operations_OpenMP, Test_dimension_3) {
  kurakin_m_monte_carlo_omp::Integral integral{.func_ =
                                                  [](std::vector<double> x) {
                                                    return std::sin(x[0]) * std::pow(x[1], 2) /
                                                           std::pow((1 + std::pow(x[2], 2)), 0.5);
                                                  },
                                              .bounds_ = {{3, 4}, {0, 5}, {-7, 10}},
                                              .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_omp::TestOMPTaskParallel testOmpTaskParallel(taskDataSeq);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  ASSERT_NEAR(-79.07, res[0], 0.3);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/example/include/ops_seq.hpp"

TEST(Sequential, Test_validation) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{1, -1}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, Test_const) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ = [](std::vector<double> x) { return 10.; }, .bounds_ = {{-1, 1}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(20, res[0], 0.1);
}

TEST(Sequential, Test_dimension_1) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ = [](std::vector<double> x) { return std::sin(x[0]); }, .bounds_ = {{0, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(0.4597, res[0], 0.01);
}

TEST(Sequential, Test_dimension_2) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ = [](std::vector<double> x) { return std::log(x[0] + x[1]) * cos(x[0] * x[1]); },
      .bounds_ = {{1, 2}, {2, 3}},
      .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(-0.7585, res[0], 0.01);
}

TEST(Sequential, Test_dimension_3) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ =
          [](std::vector<double> x) { return std::sin(x[0]) * std::pow(x[1], 2) / std::sqrt((1 + std::pow(x[2], 2))); },
      .bounds_ = {{3, 4}, {0, 1}, {-7, -6}},
      .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  kurakin_m_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(-0.0171, res[0], 0.01);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/example/include/ops_seq.hpp"

TEST(sequential_example_perf_test, test_task_run) {
  kurakin_m_monte_carlo_seq::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kurakin_m_monte_carlo_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(0, res[0], 0.01);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

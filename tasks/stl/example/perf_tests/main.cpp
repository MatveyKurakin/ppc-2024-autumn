// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "stl/example/include/ops_stl.hpp"

TEST(stl_example_perf_test_const, test_stl_task_run) {
  kurakin_m_monte_carlo_stl::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSTL->inputs_count.emplace_back(size_t(1));
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSTL->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSTL = std::make_shared<kurakin_m_monte_carlo_stl::TestSTLTaskParallel>(taskDataSTL);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSTL);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(0, res[0], 0.01);
}

TEST(stl_example_perf_test_const, test_seq_task_run) {
  kurakin_m_monte_carlo_stl::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSeq = std::make_shared<kurakin_m_monte_carlo_stl::TestTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(0, res[0], 0.01);
}

TEST(stl_example_perf_test_dimension_1, test_stl_task_run) {
  kurakin_m_monte_carlo_stl::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSTL->inputs_count.emplace_back(size_t(1));
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSTL->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSTL = std::make_shared<kurakin_m_monte_carlo_stl::TestSTLTaskParallel>(taskDataSTL);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSTL);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(0, res[0], 0.01);
}

TEST(stl_example_perf_test_dimension_1, test_seq_task_run) {
  kurakin_m_monte_carlo_stl::Integral integral{
      .func_ = [](std::vector<double> x) { return x[0]; }, .bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
  taskDataSeq->inputs_count.emplace_back(size_t(1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSeq = std::make_shared<kurakin_m_monte_carlo_stl::TestTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR(0, res[0], 0.01);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
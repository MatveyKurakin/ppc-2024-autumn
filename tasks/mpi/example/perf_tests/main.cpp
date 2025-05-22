// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/example/include/ops_mpi.hpp"

TEST(mpi_omp_example_perf_test_const, test_mpi_omp_task_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiOmpTaskParallel = std::make_shared<kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel>(
      taskDataPar, [](std::vector<double> x) { return x[0]; });
  ASSERT_EQ(testMpiOmpTaskParallel->validation(), true);
  testMpiOmpTaskParallel->pre_processing();
  testMpiOmpTaskParallel->run();
  testMpiOmpTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiOmpTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(0, res[0], 0.1);
  }
}

TEST(mpi_omp_example_perf_test_const, test_seq_task_run) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{-1, 1}}, .iterations_ = 1000000};
    std::vector<double> res(1, 0);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&integral));
    taskDataSeq->inputs_count.emplace_back(size_t(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataSeq->outputs_count.emplace_back(res.size());

    auto testTaskSeq = std::make_shared<kurakin_m_monte_carlo_mpi_omp::TestTaskSequential>(
        taskDataSeq, [](std::vector<double> x) { return x[0]; });
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

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
    ASSERT_NEAR(0, res[0], 0.1);
  }
}

TEST(mpi_omp_example_perf_test_dimension_1, test_mpi_omp_task_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{-1, 1}, {-1, 1}, {-1, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiOmpTaskParallel = std::make_shared<kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel>(
      taskDataPar, [](std::vector<double> x) { return x[0] + x[1] + x[2]; });
  ASSERT_EQ(testMpiOmpTaskParallel->validation(), true);
  testMpiOmpTaskParallel->pre_processing();
  testMpiOmpTaskParallel->run();
  testMpiOmpTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiOmpTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(0, res[0], 0.1);
  }
}

TEST(mpi_omp_example_perf_test_dimension_1, test_seq_task_run) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> res(1, 0);

    kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{-1, 1}, {-1, 1}, {-1, 1}}, .iterations_ = 1000000};

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&integral));
    taskDataSeq->inputs_count.emplace_back(size_t(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataSeq->outputs_count.emplace_back(res.size());

    auto testTaskSeq = std::make_shared<kurakin_m_monte_carlo_mpi_omp::TestTaskSequential>(
        taskDataSeq, [](std::vector<double> x) { return x[0] + x[1] + x[2]; });
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

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
    ASSERT_NEAR(0, res[0], 0.1);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0 && (argc < 2 || argv[1] != std::string("--full-workers-log"))) {
    class WorkersTestPrinter : public ::testing::EmptyTestEventListener {
     public:
      WorkersTestPrinter(std::unique_ptr<TestEventListener>&& base, int rank) : base_(std::move(base)), rank_(rank) {}

      void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (test_info.result()->Passed()) {
          return;
        }
        print_process_rank();
        base_->OnTestEnd(test_info);
      }

      void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        print_process_rank();
        base_->OnTestPartResult(test_part_result);
      }

     private:
      void print_process_rank() const { printf(" [  PROCESS %d  ] ", rank_); }

      std::unique_ptr<TestEventListener> base_;
      int rank_;
    };
    listeners.Append(new WorkersTestPrinter(
        std::unique_ptr<::testing::TestEventListener>(listeners.Release(listeners.default_result_printer())),
        world.rank()));
  }
  struct BufferGarbageDetector : public ::testing::EmptyTestEventListener {
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
      world.barrier();
      if (const auto status = world.iprobe(boost::mpi::any_source, boost::mpi::any_tag)) {
        fprintf(stderr, "[  PROCESS %d  ] [  FAILED  ] %s.%s: MPI buffer is cluttered, unread message tag is %d\n",
                world.rank(), test_info.test_suite_name(), test_info.name(), status->tag());
        exit(2);
      }
      world.barrier();
    }

    boost::mpi::communicator world;
  };
  listeners.Append(new BufferGarbageDetector);
  return RUN_ALL_TESTS();
}

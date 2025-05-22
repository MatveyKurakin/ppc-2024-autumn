// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "mpi/example/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI_OpenMP, Test_validation) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{1, -1}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());

    kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel testMPIOmpTaskParallel(
        taskDataPar, [](std::vector<double> x) { return x[0]; });
    ASSERT_EQ(testMPIOmpTaskParallel.validation(), false);
  }
}

TEST(Parallel_Operations_MPI_OpenMP, Test_const) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{-1, 1}}, .iterations_ = 100000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel testMPIOmpTaskParallel(
      taskDataPar, [](std::vector<double> x) { return 10. + (0 * x[0]); });
  ASSERT_EQ(testMPIOmpTaskParallel.validation(), true);
  testMPIOmpTaskParallel.pre_processing();
  testMPIOmpTaskParallel.run();
  testMPIOmpTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(20, res[0], 0.1);
  }
}

TEST(Parallel_Operations_MPI_OpenMP, Test_dimension_1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{0, 1}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel testMPIOmpTaskParallel(
      taskDataPar, [](std::vector<double> x) { return std::sin(x[0]); });
  ASSERT_EQ(testMPIOmpTaskParallel.validation(), true);
  testMPIOmpTaskParallel.pre_processing();
  testMPIOmpTaskParallel.run();
  testMPIOmpTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(0.4597, res[0], 0.01);
  }
}

TEST(Parallel_Operations_MPI_OpenMP, Test_dimension_2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{1, 2}, {2, 3}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel testMPIOmpTaskParallel(
      taskDataPar, [](std::vector<double> x) { return std::log(x[0] + x[1]) * cos(x[0] * x[1]); });
  ASSERT_EQ(testMPIOmpTaskParallel.validation(), true);
  testMPIOmpTaskParallel.pre_processing();
  testMPIOmpTaskParallel.run();
  testMPIOmpTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(-0.7585, res[0], 0.01);
  }
}

TEST(Parallel_Operations_MPI_OpenMP, Test_dimension_3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  kurakin_m_monte_carlo_mpi_omp::Integral integral{.bounds_ = {{3, 4}, {0, 1}, {-7, -6}}, .iterations_ = 1000000};
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&integral));
    taskDataPar->inputs_count.emplace_back(size_t(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel testMPIOmpTaskParallel(taskDataPar, [](std::vector<double> x) {
    return std::sin(x[0]) * std::pow(x[1], 2) / std::sqrt((1 + std::pow(x[2], 2)));
  });
  ASSERT_EQ(testMPIOmpTaskParallel.validation(), true);
  testMPIOmpTaskParallel.pre_processing();
  testMPIOmpTaskParallel.run();
  testMPIOmpTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(-0.0171, res[0], 0.01);
  }
}

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  auto &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0 && (argc < 2 || argv[1] != std::string("--full-workers-log"))) {
    class WorkersTestPrinter : public ::testing::EmptyTestEventListener {
     public:
      WorkersTestPrinter(std::unique_ptr<TestEventListener> &&base, int rank) : base_(std::move(base)), rank_(rank) {}

      void OnTestEnd(const ::testing::TestInfo &test_info) override {
        if (test_info.result()->Passed()) {
          return;
        }
        print_process_rank();
        base_->OnTestEnd(test_info);
      }

      void OnTestPartResult(const ::testing::TestPartResult &test_part_result) override {
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
    void OnTestEnd(const ::testing::TestInfo &test_info) override {
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

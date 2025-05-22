// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_monte_carlo_mpi_omp {

struct Integral {
  // double (*func_)(std::vector<double> x);
  std::vector<std::pair<double, double>> bounds_;
  size_t iterations_;
};

class TestMPIOMPTaskParallel : public ppc::core::Task {
 public:
  explicit TestMPIOMPTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData, double (*func)(std::vector<double> x))
      : Task(std::move(taskData)), func_(func) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world_;
  double (*func_)(std::vector<double> x);
  double res_{};
};
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData, double (*func)(std::vector<double> x))
      : Task(std::move(taskData)), func_(func) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double (*func_)(std::vector<double> x);
  double res_{};
};

}  // namespace kurakin_m_monte_carlo_mpi_omp
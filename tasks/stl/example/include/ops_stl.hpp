// Copyright 2023 Nesterov Alexander
#ifndef TASKS_EXAMPLES_TEST_STD_OPS_STD_H_
#define TASKS_EXAMPLES_TEST_STD_OPS_STD_H_

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_monte_carlo_stl {

struct Integral {
  double (*func_)(std::vector<double> x);
  std::vector<std::pair<double, double>> bounds_;
  size_t iterations_;
};

class TestSTLTaskParallel : public ppc::core::Task {
 public:
  explicit TestSTLTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double res{};
};
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double res{};
};

}  // namespace kurakin_m_monte_carlo_stl

#endif  // TASKS_EXAMPLES_TEST_STD_OPS_STD_H_

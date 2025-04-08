// Copyright 2023 Nesterov Alexander
#ifndef TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_
#define TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_

#include <tbb/tbb.h>

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_monte_carlo_tbb {

struct Integral {
  double (*func_)(std::vector<double> x);
  std::vector<std::pair<double, double>> bounds_;
  size_t iterations_;
};

class MonteCarloMethods {
  Integral integral_;
  double sum_;

 public:
  MonteCarloMethods(Integral integral) : integral_(integral), sum_(0.0f) {}
  MonteCarloMethods(const MonteCarloMethods& other, tbb::split) : integral_(other.integral_), sum_(0.0f) {}
  void operator()(const tbb::blocked_range<size_t>& r);
  void join(const MonteCarloMethods& other);
  double get_sum();
};

class TestTBBTaskParallel : public ppc::core::Task {
 public:
  explicit TestTBBTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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

}  // namespace kurakin_m_monte_carlo_tbb

#endif  // TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_

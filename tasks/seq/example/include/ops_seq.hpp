// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_monte_carlo_seq {

struct Integral {
  double (*func_)(std::vector<double> x);
  std::vector<std::pair<double, double>> bounds_;
  size_t iterations_;
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

}  // namespace kurakin_m_monte_carlo_seq
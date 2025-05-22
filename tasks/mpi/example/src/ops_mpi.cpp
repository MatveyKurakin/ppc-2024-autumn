// Copyright 2023 Nesterov Alexander
#include "mpi/example/include/ops_mpi.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

bool kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel::validation() {
  internal_order_test();
  bool is_valid = true;
  if (world_.rank() == 0) {
    auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
    is_valid = std::ranges::all_of(integral.bounds_.cbegin(), integral.bounds_.cend(),
                                   [](const auto& bounds) { return bounds.first < bounds.second; });
  }
  return is_valid;
}

bool kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel::run() {
  internal_order_test();

  std::vector<std::pair<double, double>> bounds;
  size_t iterations = 1;
  size_t iteration_proc;
  if (world_.rank() == 0) {
    Integral integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
    bounds = integral.bounds_;
    iterations = integral.iterations_;
    iteration_proc = (iterations + world_.size() - 1) / world_.size();
  }
  boost::mpi::broadcast(world_, bounds, 0);
  boost::mpi::broadcast(world_, iteration_proc, 0);

  double sum = 0.0;
  double local_sum = 0.0;

  double section = 1.0;
  if (world_.rank() == 0) {
    for (const auto& b : bounds) {
      section *= b.second - b.first;
    }
  }
  const int count_proc = 3;

#pragma omp parallel shared(bounds) num_threads(count_proc)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> x(bounds.size());
#pragma omp for reduction(+ : local_sum)
    for (int k = 0; k < iteration_proc; ++k) {
      for (int i = 0; i < bounds.size(); ++i) {
        x[i] = std::uniform_real_distribution<double>(bounds[i].first, bounds[i].second)(gen);
      }
      local_sum += func_(x);
    }
  }
  boost::mpi::reduce(world_, local_sum, sum, std::plus<double>(), 0);
  if (world_.rank() == 0) {
    res_ = (sum * section) / static_cast<double>(iterations);
  }
  return true;
}

bool kurakin_m_monte_carlo_mpi_omp::TestMPIOMPTaskParallel::post_processing() {
  internal_order_test();
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}

bool kurakin_m_monte_carlo_mpi_omp::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_mpi_omp::TestTaskSequential::validation() {
  internal_order_test();
  bool ret = false;
  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  if (std::ranges::all_of(integral.bounds_.cbegin(), integral.bounds_.cend(),
                          [](std::pair<double, double> bounds) { return bounds.first < bounds.second; }))
    ret = true;
  return ret;
}

bool kurakin_m_monte_carlo_mpi_omp::TestTaskSequential::run() {
  internal_order_test();

  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  res_ = 0.0;

  double section = 1.0;
  for (const auto& bounds : integral.bounds_) {
    section *= bounds.second - bounds.first;
  }

  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<double> x(integral.bounds_.size());
  for (size_t k = 0; k < integral.iterations_; ++k) {
    for (size_t i = 0; i < integral.bounds_.size(); ++i) {
      x[i] = std::uniform_real_distribution<double>(integral.bounds_[i].first, integral.bounds_[i].second)(gen);
    }
    res_ += func_(x);
  }

  res_ *= section / double(integral.iterations_);

  return true;
}

bool kurakin_m_monte_carlo_mpi_omp::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;
  return true;
}

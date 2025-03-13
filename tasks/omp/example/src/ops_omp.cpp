// Copyright 2023 Nesterov Alexander
#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool kurakin_m_monte_carlo_omp::TestOMPTaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_omp::TestOMPTaskParallel::validation() {
  internal_order_test();
  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  for (const auto& bounds : integral.bounds_) {
    if (bounds.first > bounds.second) {
      return false;
    }
  }
  return true;
}

bool kurakin_m_monte_carlo_omp::TestOMPTaskParallel::run() {
  internal_order_test();

  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  double sum = 0.0;

  double section = 1.0;
  for (const auto& bounds : integral.bounds_) {
    section *= bounds.second - bounds.first;
  }

  std::random_device dev;
  std::mt19937 gen(dev());
#pragma omp parallel firstprivate(gen) shared(integral) num_threads(4)
  {
    std::vector<double> x(integral.bounds_.size());
#pragma omp for reduction(+ : sum)
    for (int k = 0; k < integral.iterations_; ++k) {
      for (int i = 0; i < integral.bounds_.size(); ++i) {
        x[i] = std::uniform_real_distribution<double>(integral.bounds_[i].first, integral.bounds_[i].second)(gen);
      }
      sum += integral.func_(x);
    }
  }

  res = (sum * section) / double(integral.iterations_);

  return true;
}

bool kurakin_m_monte_carlo_omp::TestOMPTaskParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool kurakin_m_monte_carlo_omp::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_omp::TestTaskSequential::validation() {
  internal_order_test();
  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  for (const auto& bounds : integral.bounds_) {
    if (bounds.first > bounds.second) {
      return false;
    }
  }
  return true;
}

bool kurakin_m_monte_carlo_omp::TestTaskSequential::run() {
  internal_order_test();

  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  res = 0.0;

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
    res += integral.func_(x);
  }

  res *= section / double(integral.iterations_);

  return true;
}

bool kurakin_m_monte_carlo_omp::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

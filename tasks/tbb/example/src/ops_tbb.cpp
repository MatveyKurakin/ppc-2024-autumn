// Copyright 2023 Nesterov Alexander
#include "tbb/example/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void kurakin_m_monte_carlo_tbb::MonteCarloMethods::operator()(const tbb::blocked_range<size_t>& r) {
  int begin = r.begin(), end = r.end();

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> x(integral_.bounds_.size());
  for (int k = begin; k != end; ++k) {
    for (int i = 0; i < integral_.bounds_.size(); ++i) {
      x[i] = std::uniform_real_distribution<double>(integral_.bounds_[i].first, integral_.bounds_[i].second)(gen);
    }
    sum_ += integral_.func_(x);
  }
}

void kurakin_m_monte_carlo_tbb::MonteCarloMethods::join(const MonteCarloMethods& other) { sum_ += other.sum_; }

double kurakin_m_monte_carlo_tbb::MonteCarloMethods::get_sum() { return sum_; }

bool kurakin_m_monte_carlo_tbb::TestTBBTaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTBBTaskParallel::validation() {
  internal_order_test();
  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  for (const auto& bounds : integral.bounds_) {
    if (bounds.first > bounds.second) {
      return false;
    }
  }
  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTBBTaskParallel::run() {
  internal_order_test();

  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  double sum = 0.0;

  double section = 1.0;
  for (const auto& bounds : integral.bounds_) {
    section *= bounds.second - bounds.first;
  }

  MonteCarloMethods monteCarloMethodsObject(integral);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(static_cast<size_t>(0), static_cast<size_t>(integral.iterations_)),
                       monteCarloMethodsObject);
  sum = monteCarloMethodsObject.get_sum();
  res = (sum * section) / double(integral.iterations_);

  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTBBTaskParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTaskSequential::validation() {
  internal_order_test();
  auto integral = *reinterpret_cast<Integral*>(taskData->inputs[0]);
  for (const auto& bounds : integral.bounds_) {
    if (bounds.first > bounds.second) {
      return false;
    }
  }
  return true;
}

bool kurakin_m_monte_carlo_tbb::TestTaskSequential::run() {
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

bool kurakin_m_monte_carlo_tbb::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

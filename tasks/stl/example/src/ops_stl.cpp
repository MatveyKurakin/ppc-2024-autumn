// Copyright 2023 Nesterov Alexander
#include "stl/example/include/ops_stl.hpp"

#include <algorithm>
#include <functional>
#include <future>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

void MonteCarloMethods(double (*func)(std::vector<double> x), std::vector<std::pair<double, double>> &bounds,
                       size_t iterations, std::promise<double> &&pr) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> x(bounds.size());
  double sum = 0.0;
  for (int k = 0; k != iterations; ++k) {
    for (int i = 0; i < bounds.size(); ++i) {
      x[i] = std::uniform_real_distribution<double>(bounds[i].first, bounds[i].second)(gen);
    }
    sum += func(x);
  }
  pr.set_value(sum);
}

bool kurakin_m_monte_carlo_stl::TestSTLTaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_stl::TestSTLTaskParallel::validation() {
  internal_order_test();
  bool ret = false;
  auto integral = *reinterpret_cast<Integral *>(taskData->inputs[0]);
  if (std::ranges::all_of(integral.bounds_.cbegin(), integral.bounds_.cend(),
                          [](std::pair<double, double> bounds) { return bounds.first < bounds.second; }))
    ret = true;
  return ret;
}

bool kurakin_m_monte_carlo_stl::TestSTLTaskParallel::run() {
  internal_order_test();
  auto integral = *reinterpret_cast<Integral *>(taskData->inputs[0]);
  double sum = 0.0;

  double section = 1.0;
  for (const auto &bounds : integral.bounds_) {
    section *= bounds.second - bounds.first;
  }

  const int nthreads = 4;  // std::thread::hardware_concurrency();

  size_t iteration_thread = (integral.iterations_ + nthreads - 1) / nthreads;

  std::vector<std::promise<double>> promises(nthreads);
  std::vector<std::future<double>> futures;
  futures.reserve(nthreads);
  std::vector<std::thread> threads;
  threads.reserve(nthreads);

  for (auto &p : promises) {
    futures.push_back(p.get_future());
  }

  for (std::size_t i = 0; i < nthreads; i++) {
    threads.emplace_back(MonteCarloMethods, integral.func_, std::ref(integral.bounds_), iteration_thread,
                         std::move(promises[i]));
  }
  for (auto &th : threads) {
    th.join();
  }
  for (auto &f : futures) {
    sum += f.get();
  }

  res = (sum * section) / double(integral.iterations_);

  return true;
}

bool kurakin_m_monte_carlo_stl::TestSTLTaskParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<double *>(taskData->outputs[0])[0] = res;
  return true;
}

bool kurakin_m_monte_carlo_stl::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_monte_carlo_stl::TestTaskSequential::validation() {
  internal_order_test();
  bool ret = false;
  auto integral = *reinterpret_cast<Integral *>(taskData->inputs[0]);
  if (std::ranges::all_of(integral.bounds_.cbegin(), integral.bounds_.cend(),
                          [](std::pair<double, double> bounds) { return bounds.first < bounds.second; }))
    ret = true;
  return ret;
}

bool kurakin_m_monte_carlo_stl::TestTaskSequential::run() {
  internal_order_test();

  auto integral = *reinterpret_cast<Integral *>(taskData->inputs[0]);
  res = 0.0;

  double section = 1.0;
  for (const auto &bounds : integral.bounds_) {
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

bool kurakin_m_monte_carlo_stl::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double *>(taskData->outputs[0])[0] = res;
  return true;
}

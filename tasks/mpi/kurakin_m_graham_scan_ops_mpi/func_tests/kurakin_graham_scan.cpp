#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/kurakin_m_graham_scan_ops_mpi/include/kurakin_graham_scan_ops_mpi.hpp"

namespace kurakin_m_graham_scan_mpi {

bool inOneLine(std::vector<double> &res, double new_point_x, double new_point_y) {
  for (int j = 2; j < res.size(); j += 2) {
    if (abs((new_point_x - res[0]) * (res[j + 1] - res[1]) - (new_point_y - res[1]) * (res[j] - res[0])) < 1e-4) {
      return true;
    }
  }
  return false;
}

double getRandomDouble(double start = 0.0, double end = 100.0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  double res =
      (double)(gen() % ((long long)(end * 10000) - (long long)(start * 10000)) + (long long)(start * 10000)) / 10000.0;
  return res;
}

int getRandomInt(int start = 0, int end = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int res = gen() % (end - start) + start;
  return res;
}

void getShuffle(std::vector<double> &vec) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::shuffle(vec.begin(), vec.end(), gen);
}

void getRandomVectorForGrahamScan(std::vector<double> &res, int count_point, int size) {
  res = std::vector<double>();
  res = {getRandomDouble(-1000.0, 0.0), getRandomDouble(-1000.0, 0.0), getRandomDouble(0.0, 1000.0),
         getRandomDouble(0.0, 1000.0)};
  for (int i = 4; i < count_point * 2; i += 2) {
    double new_point_x = getRandomDouble(-1000.0, 1000.0);
    double new_point_y = getRandomDouble(-1000.0, 1000.0);
    while (inOneLine(res, new_point_x, new_point_y)) {
      new_point_x += getRandomDouble(-100.0, 100.0);
      new_point_y += getRandomDouble(-100.0, 100.0);
    }
    res.push_back(new_point_x);
    res.push_back(new_point_y);
  }
  getShuffle(res);
}

}  // namespace kurakin_m_graham_scan_mpi

TEST(kurakin_m_graham_scan_mpi, Test_shell_rhomb) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 4;

  if (world.rank() == 0) {
    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_square) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 4;

  if (world.rank() == 0) {
    points = {2.0, 2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {2.0, 2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_rhomb_with_inside_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 17;

  if (world.rank() == 0) {
    points = {0.3, -0.25, 1.0, 0.0,   2.0, 0.0,  0.3, 0.25, 0.0,  -2.0, 0.0, -1.0, 0.25, -0.3, -0.25, -0.3, 0.0,
              1.0, 0.0,   2.0, -0.25, 0.3, 0.25, 0.3, -0.3, 0.25, -1.0, 0.0, -2.0, 0.0,  -0.3, -0.25, 0.1,  0.1};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {0.3, -0.25, 1.0, 0.0,   2.0, 0.0,  0.3, 0.25, 0.0,  -2.0, 0.0, -1.0, 0.25, -0.3, -0.25, -0.3, 0.0,
              1.0, 0.0,   2.0, -0.25, 0.3, 0.25, 0.3, -0.3, 0.25, -1.0, 0.0, -2.0, 0.0,  -0.3, -0.25, 0.1,  0.1};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_square_with_inside_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 17;

  if (world.rank() == 0) {
    points = {-2.0, -2.0, -1.0, -1.0, -0.5, -1.0, -1.0, -0.5, 2.0, -2.0, 0.5, -1.0, 1.0, -1.0, 1.0, -0.5, 2.0,
              2.0,  1.0,  1.0,  0.5,  1.0,  1.0,  0.5,  -2.0, 2.0, -0.5, 1.0, -1.0, 1.0, -1.0, 0.5, 0.1,  0.1};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {-2.0, -2.0, -1.0, -1.0, -0.5, -1.0, -1.0, -0.5, 2.0, -2.0, 0.5, -1.0, 1.0, -1.0, 1.0, -0.5, 2.0,
              2.0,  1.0,  1.0,  0.5,  1.0,  1.0,  0.5,  -2.0, 2.0, -0.5, 1.0, -1.0, 1.0, -1.0, 0.5, 0.1,  0.1};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 4;
    std::vector<double> ans = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_count_3) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 3;

  if (world.rank() == 0) {
    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 3;
    std::vector<double> ans = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_count_5) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 5;

  if (world.rank() == 0) {
    points = {2.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 2.0, -2.0, 0.0};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {2.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 2.0, -2.0, 0.0};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 5;
    std::vector<double> ans = {1.0, -1.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0, -1.0, -1.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_count_7) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 7;

  if (world.rank() == 0) {
    points = {2.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 1.5, 1.5, -1.5, 1.5};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    points = {2.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 1.5, 1.5, -1.5, 1.5};

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    int ans_size = 7;
    std::vector<double> ans = {1.0, -1.0, 2.0, 0.0, 1.5, 1.5, 0.0, 2.0, -1.5, 1.5, -2.0, 0.0, -1.0, -1.0};

    ASSERT_EQ(scan_size_par, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_par[i], ans[i]);
      ASSERT_EQ(scan_points_par[i + 1], ans[i + 1]);
    }

    ASSERT_EQ(scan_size_seq, ans_size);
    for (int i = 0; i < ans_size * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], ans[i]);
      ASSERT_EQ(scan_points_seq[i + 1], ans[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_random_count_17) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = 17;

  if (world.rank() == 0) {
    kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(points, count_point, world.size());

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(scan_size_seq, scan_size_par);
    for (int i = 0; i < scan_size_seq * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], scan_points_par[i]);
      ASSERT_EQ(scan_points_seq[i + 1], scan_points_par[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_random_count_pow_3_n) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = std::pow(3, world.size());

  if (world.rank() == 0) {
    kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(points, count_point, world.size());

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(scan_size_seq, scan_size_par);
    for (int i = 0; i < scan_size_seq * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], scan_points_par[i]);
      ASSERT_EQ(scan_points_seq[i + 1], scan_points_par[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_random_count_pow_5_n) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = std::pow(5, world.size());

  if (world.rank() == 0) {
    kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(points, count_point, world.size());

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(scan_size_seq, scan_size_par);
    for (int i = 0; i < scan_size_seq * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], scan_points_par[i]);
      ASSERT_EQ(scan_points_seq[i + 1], scan_points_par[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_shell_random_count_pow_10_n) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  count_point = std::pow(10, world.size());

  if (world.rank() == 0) {
    kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(points, count_point, world.size());

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());
  }

  kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    int scan_size_seq;
    std::vector<double> scan_points_seq(count_point * 2);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataSeq->inputs_count.emplace_back(points.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_seq));
    taskDataSeq->outputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_seq.data()));
    taskDataSeq->outputs_count.emplace_back(scan_points_seq.size());

    kurakin_m_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(scan_size_seq, scan_size_par);
    for (int i = 0; i < scan_size_seq * 2; i += 2) {
      ASSERT_EQ(scan_points_seq[i], scan_points_par[i]);
      ASSERT_EQ(scan_points_seq[i + 1], scan_points_par[i + 1]);
    }
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_count_points) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 2;
    points = {2.0, 2.0, 1.0, 1.0};

    scan_points_par = std::vector<double>(count_point * 2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_inputs_point) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    points = {2.0, 2.0, 1.0, 1.0, -2.0, 2.0, 1.0};

    scan_points_par = std::vector<double>(count_point * 2, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_outputs_point) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

    scan_points_par = std::vector<double>(count_point * 2 - 1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_inputs_count) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;

    scan_points_par = std::vector<double>(count_point * 2 - 1, 0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points_par.data()));
    taskDataPar->outputs_count.emplace_back(scan_points_par.size());

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_graham_scan_mpi, Test_validation_outputs_count) {
  boost::mpi::communicator world;

  int count_point;
  std::vector<double> points;

  int scan_size_par;
  std::vector<double> scan_points_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_point = 4;
    points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

    scan_points_par = std::vector<double>(count_point * 2 - 1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
    taskDataPar->inputs_count.emplace_back(points.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size_par));
    taskDataPar->outputs_count.emplace_back((size_t)1);

    kurakin_m_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
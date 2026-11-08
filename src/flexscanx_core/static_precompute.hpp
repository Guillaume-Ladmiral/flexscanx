#pragma once
#include <limits>
#include <memory>
#include <vector>

namespace flexscanx_core {

struct CSR {
  std::vector<int> row_ptr; // size N+1
  std::vector<int> col_idx; // size E*2 for undirected
};

enum class KNNMode {
  K_FIXED = 0, // fixed top-K
  K_REFILL = 1 // refill to K from farther neighbors when some are excluded (guardrailed)
};

struct Precompute {
  int N{};
  int K{};            // KNN size per center
  double radius{};    // cluster radius bound (Euclidean); inf => disabled
  double radius_sq{}; // squared radius for fast comparisons

  std::vector<double> x; // coords
  std::vector<double> y;
  CSR csr;

  // Per-center KNN universe (indices and distances)
  std::vector<std::vector<int>> knn_idx;    // top-K indices (sorted)
  std::vector<std::vector<float>> knn_dist; // top-K squared distances (sorted, float32)

  // Optional extended neighbor lists for refill mode (distance-sorted)
  // Contains the first K + refill_extra neighbors only (not all N), to avoid O(N^2) builds.
  std::vector<std::vector<int>> knn_idx_refill;

  // Per-center flag: 1 when the K-th and (K+1)-th nearest neighbours are exactly
  // equidistant and inside the radius, so the K-limited universe boundary falls
  // inside a tied distance group and is resolved by area index (FXA-187 / D4).
  // Relabelling the input rows can then change which neighbour enters the
  // universe; the Python layer warns when any center is flagged.
  std::vector<char> knn_tie_at_boundary;

  int knn_tie_boundary_count() const {
    int count = 0;
    for (char flag : knn_tie_at_boundary)
      count += (flag != 0);
    return count;
  }

  KNNMode knn_mode{KNNMode::K_FIXED};
  bool refill_enabled{false}; // true if knn_idx_refill is built and usable
  // Extended neighbors for refill: benchmarks on moderate-degree graphs (avg_degree≈6)
  // show 75 extra neighbors provide ~95% coverage after typical exclusions (≤10 nodes).
  // Higher values yield diminishing returns (<1% improvement) while increasing memory.
  int refill_extra{75};
};

std::unique_ptr<Precompute> build_static(const std::vector<double> &x, const std::vector<double> &y,
                                         const CSR &csr, int K, double radius, KNNMode knn_mode,
                                         bool force_refill);

double euclid(double x1, double y1, double x2, double y2);
double euclid2(double x1, double y1, double x2, double y2);

} // namespace flexscanx_core

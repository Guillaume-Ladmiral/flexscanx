#include "static_precompute.hpp"
#include "algorithms/scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

namespace flexscanx_core {

double euclid(double x1, double y1, double x2, double y2) {
  const long double dx = static_cast<long double>(x1) - static_cast<long double>(x2);
  const long double dy = static_cast<long double>(y1) - static_cast<long double>(y2);
  return static_cast<double>(std::hypot(dx, dy));
}

double euclid2(double x1, double y1, double x2, double y2) {
  const long double dx = static_cast<long double>(x1) - static_cast<long double>(x2);
  const long double dy = static_cast<long double>(y1) - static_cast<long double>(y2);
  return static_cast<double>(dx * dx + dy * dy);
}

// Simple 2D KD-tree for KNN (header-only replacement for nanoflann-like behavior)
namespace {
struct KDNode {
  int idx{-1};
  int left{-1};
  int right{-1};
  int axis{0}; // 0=x, 1=y
  double split{0.0};
};

struct KDTree2D {
  const std::vector<double> &x;
  const std::vector<double> &y;
  std::vector<int> order;    // indices into points
  std::vector<KDNode> nodes; // nodes pool
  int root{-1};

  KDTree2D(const std::vector<double> &x_, const std::vector<double> &y_) : x(x_), y(y_) {
    const int N = (int)x.size();
    order.resize(N);
    for (int i = 0; i < N; ++i)
      order[i] = i;
    nodes.reserve(N);
    root = build(0, N, /*depth=*/0);
  }

  int build(int l, int r, int depth) {
    if (l >= r)
      return -1;
    int axis = depth & 1; // alternate axis
    int m = (l + r) >> 1;
    auto comp = [&](int a, int b) {
      if (axis == 0) {
        if (x[a] < x[b])
          return true;
        if (x[a] > x[b])
          return false;
        return a < b; // stable tie-break by index
      } else {
        if (y[a] < y[b])
          return true;
        if (y[a] > y[b])
          return false;
        return a < b;
      }
    };
    std::nth_element(order.begin() + l, order.begin() + m, order.begin() + r, comp);
    int node_id = (int)nodes.size();
    nodes.push_back({});
    KDNode &nd = nodes.back();
    nd.idx = order[m];
    nd.axis = axis;
    nd.split = (axis == 0 ? x[nd.idx] : y[nd.idx]);
    nd.left = build(l, m, depth + 1);
    nd.right = build(m + 1, r, depth + 1);
    return node_id;
  }

  inline long double distance(int i, int j) const {
    const long double dx = static_cast<long double>(x[i]) - static_cast<long double>(x[j]);
    const long double dy = static_cast<long double>(y[i]) - static_cast<long double>(y[j]);
    return std::hypot(dx, dy);
  }

  // Query K nearest neighbors of point c (by index) including itself.
  void knn(int c, int K, std::vector<std::pair<long double, int>> &out_pairs) const {
    out_pairs.clear();
    out_pairs.reserve(K);
    using Key = std::pair<long double, int>; // (distance, idx) for lexicographic ordering
    auto worse = [](const Key &a, const Key &b) { return a < b; };
    std::priority_queue<Key, std::vector<Key>, decltype(worse)> heap(worse);

    auto try_push = [&](long double distance_value, int idx) {
      Key cand(distance_value, idx);
      if ((int)heap.size() < K) {
        heap.push(cand);
      } else {
        const Key &worst = heap.top();
        if (cand < worst) {
          heap.pop();
          heap.push(cand);
        }
      }
    };

    std::function<void(int)> dfs = [&](int node_id) {
      if (node_id < 0)
        return;
      const KDNode &nd = nodes[node_id];
      const int i = nd.idx;
      const long double distance_value = distance(c, i);
      try_push(distance_value, i);
      const long double val =
          (nd.axis == 0 ? static_cast<long double>(x[c]) - static_cast<long double>(nd.split)
                        : static_cast<long double>(y[c]) - static_cast<long double>(nd.split));
      int near = (val <= 0.0 ? nd.left : nd.right);
      int far = (val <= 0.0 ? nd.right : nd.left);
      dfs(near);
      const bool need_more = (int)heap.size() < K;
      const long double worst_distance =
          need_more ? std::numeric_limits<long double>::infinity() : heap.top().first;
      if (need_more || std::abs(val) <= worst_distance) {
        dfs(far);
      }
    };

    dfs(root);
    // Extract sorted ascending by (distance, index).
    std::vector<Key> tmp;
    tmp.reserve(heap.size());
    while (!heap.empty()) {
      tmp.push_back(heap.top());
      heap.pop();
    }
    std::sort(tmp.begin(), tmp.end(), [](const Key &a, const Key &b) {
      if (a.first < b.first)
        return true;
      if (a.first > b.first)
        return false;
      return a.second < b.second;
    });
    out_pairs.clear();
    out_pairs.reserve(tmp.size());
    for (auto &kv : tmp)
      out_pairs.emplace_back(kv.first, kv.second);
  }
};
} // namespace

std::unique_ptr<Precompute> build_static(const std::vector<double> &x, const std::vector<double> &y,
                                         const CSR &csr, int K, double radius, KNNMode knn_mode,
                                         bool force_refill) {
  // Guardrail threshold: refill mode trades O(N*K_ext) build time for O(K_ext) query speed.
  // Benchmarks show <2% speedup above N=5000 due to cache pressure and memory bandwidth limits.
  // Memory impact: refill stores (K+75)*N extra integers vs K*N for K_FIXED mode.
  constexpr int KNN_REFILL_MAX_N = 5000;
  if (y.size() != x.size()) {
    throw std::runtime_error("x and y must have the same length");
  }
  if (x.empty()) {
    throw std::invalid_argument("coords must contain at least one point");
  }
  if (x.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("coords contains too many points");
  }
  const int N = static_cast<int>(x.size());
  if (K < 1 || K > N) {
    throw std::invalid_argument("K must be in [1, N]");
  }
  if (!(radius >= 0.0)) {
    throw std::invalid_argument("radius must be finite and non-negative or +inf");
  }
  for (int i = 0; i < N; ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
      throw std::invalid_argument("coordinates must be finite");
    }
  }
  auto pre = std::make_unique<Precompute>();
  pre->N = N;
  pre->K = K;
  pre->radius = radius;
  pre->radius_sq =
      std::isfinite(radius) ? (radius * radius) : std::numeric_limits<double>::infinity();
  pre->x = x;
  pre->y = y;
  pre->csr = csr;
  pre->knn_mode = knn_mode;
  pre->refill_enabled =
      (knn_mode == KNNMode::K_REFILL && (force_refill || pre->N <= KNN_REFILL_MAX_N));
  // CSR basic validation: row_ptr size and col_idx bounds
  if ((int)pre->csr.row_ptr.size() != pre->N + 1) {
    throw std::runtime_error("csr.row_ptr must have size N+1");
  }
  if (pre->csr.row_ptr.empty() || pre->csr.row_ptr[0] != 0) {
    throw std::runtime_error("csr.row_ptr must start at 0");
  }
  if (!std::is_sorted(pre->csr.row_ptr.begin(), pre->csr.row_ptr.end())) {
    throw std::runtime_error("csr.row_ptr must be non-decreasing");
  }
  if (pre->csr.row_ptr.back() != (int)pre->csr.col_idx.size()) {
    throw std::runtime_error("csr.row_ptr[N] must equal length of col_idx");
  }
  for (int v : pre->csr.col_idx) {
    if (v < 0 || v >= pre->N) {
      throw std::runtime_error("csr.col_idx contains out-of-range vertex index");
    }
  }

  pre->knn_idx.resize(pre->N);
  pre->knn_dist.resize(pre->N);
  pre->knn_tie_at_boundary.assign(pre->N, 0);
  if (pre->refill_enabled) {
    pre->knn_idx_refill.resize(pre->N);
  }

  // Build KD-tree once over all points
  KDTree2D kdt(pre->x, pre->y);

  // Multi-threaded per-center KNN queries
  const int K_eff = pre->K;
  int hw = (int)std::thread::hardware_concurrency();
  if (hw <= 0)
    hw = 1;
  int T = std::min(std::max(1, hw), N);
  // Split into T contiguous chunks
  auto worker = [&](int t, const std::atomic<bool> &cancel) {
    int chunk = (N + T - 1) / T;
    int start = t * chunk;
    int end = std::min(N, start + chunk);
    if (start >= end)
      return;
    std::vector<std::pair<long double, int>> topk;
    topk.reserve(K_eff);
    std::vector<std::pair<long double, int>> topk_ext;
    topk_ext.reserve(K_eff + pre->refill_extra);
    std::vector<std::pair<long double, int>> topk_plus;
    topk_plus.reserve(K_eff + 1);
    // The K-limited universe boundary falls inside a tied distance group when
    // the K-th and (K+1)-th nearest neighbours are exactly equidistant.  The
    // index tie-break then decides which one is admitted, so the admissible
    // window set depends on the caller's row order (FXA-187).  A boundary
    // beyond the radius cannot matter: both tied neighbours are excluded.
    auto flag_tie_at_boundary = [&](int c,
                                    const std::vector<std::pair<long double, int>> &neighbors) {
      // neighbors: center first, then distance-sorted; index K_eff is the first
      // neighbour outside the universe.
      if ((int)neighbors.size() <= K_eff || K_eff < 1)
        return;
      const long double boundary = neighbors[K_eff - 1].first;
      const long double next = neighbors[K_eff].first;
      if (boundary != next)
        return;
      if (pre->radius < std::numeric_limits<double>::infinity() &&
          boundary > static_cast<long double>(pre->radius))
        return;
      pre->knn_tie_at_boundary[c] = 1;
    };
    auto center_first = [&](int c, int limit, std::vector<std::pair<long double, int>> &neighbors) {
      auto it = std::find_if(neighbors.begin(), neighbors.end(),
                             [&](const auto &p) { return p.second == c; });
      if (it != neighbors.end())
        neighbors.erase(it);
      neighbors.insert(neighbors.begin(), std::make_pair(0.0L, c));
      if ((int)neighbors.size() > limit)
        neighbors.resize(limit);
    };
    for (int c = start; c < end; ++c) {
      if (cancel.load(std::memory_order_acquire))
        return;
      // KNN via KD-tree (Euclidean distances held in long double to avoid overflow).
      kdt.knn(c, K_eff, topk);
      center_first(c, K_eff, topk);
      // Save top-K in distance order; retain the historical squared-distance cache.
      pre->knn_idx[c].clear();
      pre->knn_dist[c].clear();
      pre->knn_idx[c].reserve((int)topk.size());
      pre->knn_dist[c].reserve((int)topk.size());
      for (auto &p : topk) {
        pre->knn_idx[c].push_back(p.second);
        pre->knn_dist[c].push_back(static_cast<float>(p.first * p.first));
      }
      if (pre->refill_enabled) {
        // Build only K + refill_extra neighbors for refill mode (avoid full N)
        int Kplus = std::min(N, K_eff + std::max(0, pre->refill_extra));
        kdt.knn(c, Kplus, topk_ext);
        center_first(c, Kplus, topk_ext);
        pre->knn_idx_refill[c].clear();
        pre->knn_idx_refill[c].reserve((int)topk_ext.size());
        for (auto &p : topk_ext)
          pre->knn_idx_refill[c].push_back(p.second);
        flag_tie_at_boundary(c, topk_ext);
      } else if (K_eff < N) {
        kdt.knn(c, K_eff + 1, topk_plus);
        center_first(c, K_eff + 1, topk_plus);
        flag_tie_at_boundary(c, topk_plus);
      }
    }
  };
  Scheduler::for_each_worker(T, worker);

  return pre;
}

} // namespace flexscanx_core

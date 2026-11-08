#include "bindings_support.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flexscanx_pybind {

void validate_buildstatic_csr(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                              int N) {
  if ((int)row_ptr.size() != N + 1) {
    throw std::runtime_error("adjacency row_ptr must have length N+1 (" + std::to_string(N + 1) +
                             "), got " + std::to_string(row_ptr.size()));
  }
  if (row_ptr.empty() || row_ptr[0] != 0)
    throw std::runtime_error("adjacency row_ptr must start at 0");
  for (int i = 1; i < (int)row_ptr.size(); ++i) {
    if (row_ptr[i] < row_ptr[i - 1])
      throw std::runtime_error("adjacency row_ptr must be non-decreasing");
  }
  if (row_ptr.back() != (int)col_idx.size()) {
    throw std::runtime_error("adjacency CSR is invalid: row_ptr[N] must equal len(col_idx)");
  }

  // Build sorted per-row adjacency lists — O(E log E) total, O(E) memory.
  // Sorted order enables O(log E) symmetry checks via binary_search, and makes
  // self-loop / duplicate detection trivial during the initial sort pass.
  std::vector<std::vector<int>> sorted_adj(N);
  for (int i = 0; i < N; ++i) {
    int s = row_ptr[i], e = row_ptr[i + 1];
    sorted_adj[i].assign(col_idx.begin() + s, col_idx.begin() + e);
    std::sort(sorted_adj[i].begin(), sorted_adj[i].end());
    for (int k = 0; k < (int)sorted_adj[i].size(); ++k) {
      int j = sorted_adj[i][k];
      if (j < 0 || j >= N)
        throw std::runtime_error("adjacency col_idx contains out-of-range vertex indices");
      if (j == i)
        throw std::runtime_error("adjacency contains self-loop at node " + std::to_string(i));
      if (k > 0 && sorted_adj[i][k - 1] == j)
        throw std::runtime_error("adjacency row " + std::to_string(i) +
                                 " contains duplicate neighbors");
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j : sorted_adj[i]) {
      if (!std::binary_search(sorted_adj[j].begin(), sorted_adj[j].end(), i)) {
        throw std::runtime_error(
            "adjacency must be symmetric for an undirected graph: missing reverse edge " +
            std::to_string(j) + "->" + std::to_string(i) + " for edge " + std::to_string(i) + "->" +
            std::to_string(j));
      }
    }
  }
}

double fs_trigamma(double x) {
  double result = 0.0;
  while (x < 6.0) {
    result += 1.0 / (x * x);
    x += 1.0;
  }
  double inv = 1.0 / x;
  double inv2 = inv * inv;
  double inv3 = inv2 * inv;
  double inv5 = inv3 * inv2;
  double inv7 = inv5 * inv2;
  result += inv + 0.5 * inv2 + (1.0 / 6.0) * inv3 - (1.0 / 30.0) * inv5 + (1.0 / 42.0) * inv7;
  return result;
}

void capsule_destruct(PyObject *capsule) {
  auto *h = reinterpret_cast<PreHandle *>(PyCapsule_GetPointer(capsule, "flexscanx_pre"));
  delete h;
}

} // namespace flexscanx_pybind

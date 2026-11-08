#pragma once
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

namespace flexscanx_core {

// Build a center list from either a provided order or 0..N-1, optionally sorted by a key lambda.
// Key lambda returns a tuple comparable with std::tuple lexicographic order.

template <class KeyFn>
inline std::vector<int> build_centers(int N, const std::vector<int> *center_order, KeyFn key_fn) {
  std::vector<int> centers;
  centers.reserve(N);
  if (center_order) {
    for (int c : *center_order)
      if (c >= 0 && c < N)
        centers.push_back(c);
  } else {
    centers.resize(N);
    std::iota(centers.begin(), centers.end(), 0);
  }
  // Stable sort by provided key
  std::stable_sort(centers.begin(), centers.end(),
                   [&](int a, int b) { return key_fn(a) < key_fn(b); });
  return centers;
}

// Overload without sorting (preserve given order or 0..N-1)
inline std::vector<int> build_centers(int N, const std::vector<int> *center_order) {
  std::vector<int> centers;
  centers.reserve(N);
  if (center_order) {
    for (int c : *center_order)
      if (c >= 0 && c < N)
        centers.push_back(c);
  } else {
    centers.resize(N);
    std::iota(centers.begin(), centers.end(), 0);
  }
  return centers;
}

} // namespace flexscanx_core

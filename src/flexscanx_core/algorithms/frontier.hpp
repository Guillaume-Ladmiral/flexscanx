#pragma once
#include <algorithm>
#include <limits>
#include <vector>

namespace flexscanx_core {

// Frontier generation-marker helper to avoid O(N) clears.
struct FrontierMarker {
  std::vector<int> gen;
  int cur{1};

  void ensure(int N) {
    if ((int)gen.size() < N)
      gen.assign(N, 0);
    // bump generation; on wrap, reset table to avoid stale matches
    if (cur == std::numeric_limits<int>::max()) {
      std::fill(gen.begin(), gen.end(), 0);
      cur = 1;
    } else {
      cur += 1;
    }
  }
  inline bool is_marked(int v) const {
    return gen[v] == cur;
  }
  inline void mark(int v) {
    gen[v] = cur;
  }
};

} // namespace flexscanx_core

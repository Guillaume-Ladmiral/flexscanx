#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace flexscanx_core {

// Simple scheduler: iterate centers sequentially or with an atomic index across threads.
struct Scheduler {
private:
  static inline void join_all(std::vector<std::thread> &threads) noexcept {
    for (auto &thread : threads) {
      if (thread.joinable())
        thread.join();
    }
  }

public:
  // A single caller-supplied job budget governs a worker family.  Hardware and
  // available tasks can only reduce that budget, never override it.
  static inline int bounded_worker_count(int requested_workers, int tasks,
                                         int hardware_workers) noexcept {
    const int requested = std::max(1, requested_workers);
    const int available_hardware = std::max(1, hardware_workers);
    const int available_tasks = std::max(1, tasks);
    return std::min({requested, available_hardware, available_tasks});
  }

  // Run a fixed worker family with one shared cancellation flag. Exceptions
  // never cross a std::thread boundary: the first is retained, all started
  // threads are joined, then that exception is rethrown on the caller thread.
  // The construction guard supplies the same cleanup if emplace_back throws.
  // Worker 0 runs on the calling thread (the other T-1 workers are spawned), so
  // a cooperative interrupt poll inside fn reaches the thread that owns the
  // Python call (FXA-188 / D6).  Exceptions from any worker, the caller's own
  // included, are recorded, all threads are cancelled and joined, and the first
  // exception is rethrown here.
  template <class Fn> static inline void for_each_worker(int workers, Fn &&fn) {
    const int T = std::max(1, workers);
    std::atomic<bool> cancel{false};
    if (T == 1) {
      fn(0, cancel);
      return;
    }

    std::exception_ptr worker_error;
    std::mutex error_mtx;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(T - 1));
    auto record_failure = [&]() {
      std::lock_guard<std::mutex> lock(error_mtx);
      if (!worker_error)
        worker_error = std::current_exception();
      cancel.store(true, std::memory_order_release);
    };
    try {
      for (int t = 1; t < T; ++t) {
        threads.emplace_back([&, t]() {
          try {
            fn(t, cancel);
          } catch (...) {
            record_failure();
          }
        });
      }
    } catch (...) {
      cancel.store(true, std::memory_order_release);
      join_all(threads);
      throw;
    }
    try {
      fn(0, cancel);
    } catch (...) {
      record_failure();
    }
    join_all(threads);
    if (worker_error)
      std::rethrow_exception(worker_error);
  }

  // Center loops.  ``poll`` (optional) is invoked on the calling thread before
  // each center it processes; the calling thread participates as one of the
  // workers so that a cooperative interrupt (FXA-188 / D6) can stop the whole
  // family through the shared cancellation flag.
  template <class Fn, class Poll = std::nullptr_t>
  static inline void for_each_center_atomic(const std::vector<int> &centers, int jobs, Fn &&fn,
                                            Poll &&poll = nullptr) {
    int T = std::max(1, std::min((int)centers.size(), jobs));
    if (T <= 1) {
      for (int c : centers) {
        invoke_poll(poll);
        fn(c);
      }
      return;
    }
    std::atomic<int> idx{0};
    std::atomic<bool> cancel{false};
    std::exception_ptr worker_error;
    std::mutex error_mtx;
    std::vector<std::thread> threads;
    threads.reserve(T - 1);
    auto record_failure = [&]() {
      std::lock_guard<std::mutex> lock(error_mtx);
      if (!worker_error)
        worker_error = std::current_exception();
      cancel.store(true, std::memory_order_release);
    };
    auto worker = [&](bool is_caller) {
      while (!cancel.load(std::memory_order_acquire)) {
        try {
          if (is_caller)
            invoke_poll(poll);
        } catch (...) {
          record_failure();
          return;
        }
        const int i = idx.fetch_add(1, std::memory_order_relaxed);
        if (i >= static_cast<int>(centers.size()))
          return;
        try {
          fn(centers[i]);
        } catch (...) {
          record_failure();
          return;
        }
      }
    };
    try {
      for (int t = 1; t < T; ++t) {
        threads.emplace_back([&]() { worker(false); });
      }
    } catch (...) {
      cancel.store(true, std::memory_order_release);
      join_all(threads);
      throw;
    }
    worker(true);
    join_all(threads);
    if (worker_error)
      std::rethrow_exception(worker_error);
  }

  template <class Fn, class Poll = std::nullptr_t>
  static inline void for_each_center_worksteal(const std::vector<int> &centers, int jobs, Fn &&fn,
                                               Poll &&poll = nullptr) {
    int T = std::max(1, std::min((int)centers.size(), jobs));
    if (T <= 1) {
      for (int c : centers) {
        invoke_poll(poll);
        fn(c);
      }
      return;
    }
    std::vector<std::deque<int>> queues(T);
    for (size_t i = 0; i < centers.size(); ++i) {
      queues[i % T].push_back(centers[i]);
    }
    std::vector<std::mutex> qmtx(T);
    std::atomic<int> remaining((int)centers.size());
    std::atomic<bool> cancel{false};
    std::exception_ptr worker_error;
    std::mutex error_mtx;
    std::vector<std::thread> threads;
    threads.reserve(T - 1);
    auto record_failure = [&]() {
      std::lock_guard<std::mutex> lock(error_mtx);
      if (!worker_error)
        worker_error = std::current_exception();
      cancel.store(true, std::memory_order_release);
    };
    auto worker = [&](int t, bool is_caller) {
      int c = -1;
      while (!cancel.load(std::memory_order_acquire)) {
        try {
          if (is_caller)
            invoke_poll(poll);
        } catch (...) {
          record_failure();
          return;
        }
        // own queue
        {
          std::lock_guard<std::mutex> lk(qmtx[t]);
          if (!queues[t].empty()) {
            c = queues[t].back(); // LIFO: most-recently-pushed item is likely in cache
            queues[t].pop_back();
            remaining.fetch_sub(1, std::memory_order_relaxed);
          } else {
            c = -1;
          }
        }
        if (c != -1) {
          try {
            fn(c);
          } catch (...) {
            record_failure();
            return;
          }
          continue;
        }
        // steal
        bool stole = false;
        for (int v = 0; v < T; ++v) {
          if (v == t)
            continue;
          std::lock_guard<std::mutex> lk(qmtx[v]);
          if (!queues[v].empty()) {
            c = queues[v].front(); // FIFO steal: take oldest work, minimises contention with owner
            queues[v].pop_front();
            remaining.fetch_sub(1, std::memory_order_relaxed);
            stole = true;
            break;
          }
        }
        if (stole) {
          try {
            fn(c);
          } catch (...) {
            record_failure();
            return;
          }
          continue;
        }
        if (remaining.load(std::memory_order_relaxed) <= 0)
          return;
        std::this_thread::yield();
      }
    };
    try {
      for (int t = 1; t < T; ++t) {
        threads.emplace_back([&, t]() { worker(t, false); });
      }
    } catch (...) {
      cancel.store(true, std::memory_order_release);
      join_all(threads);
      throw;
    }
    worker(0, true);
    join_all(threads);
    if (worker_error)
      std::rethrow_exception(worker_error);
  }

private:
  static inline void invoke_poll(std::nullptr_t) noexcept {}
  template <class Poll> static inline void invoke_poll(Poll &&poll) {
    if (poll)
      poll();
  }
};

} // namespace flexscanx_core

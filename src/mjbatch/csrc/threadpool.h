// SPDX-License-Identifier: Apache-2.0

// Persistent thread pool: a blocking parallel-for with sticky slices.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#else
#include <cerrno>
#endif

// Pins the calling thread to cpu. Returns 0 on success, an errno-style code
// otherwise; only Linux supports pinning.
inline int PinThisThreadToCpu(int cpu) {
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
  return ENOTSUP;
#endif
}

// Runs fn(worker, i) for i in [0, n) and blocks until done. Item i belongs to
// the slice of worker i * T / n, which keeps a sim on the same core across
// calls; a worker that finishes its slice claims from the others, so no worker
// waits on the slowest. Only one Run may be active at a time; fn must not throw.
//
// When cpu_ids is non-empty it must have exactly nthreads entries and worker i
// pins itself to cpu_ids[i] before entering the work loop (Linux only). The
// constructor then blocks until every worker has reported, so PinError is
// final once it returns.
class ThreadPool {
 public:
  explicit ThreadPool(int nthreads, std::vector<int> cpu_ids = {})
      : nthreads_(nthreads), next_(new std::atomic<int>[nthreads]),
        cpu_ids_(std::move(cpu_ids)) {
    for (int t = 0; t < nthreads; ++t) {
      threads_.emplace_back([this, t] { Worker(t); });
    }
    // With pins requested, block until every worker applied its affinity so a
    // failure is final (and readable via PinError) before the owner proceeds.
    if (!cpu_ids_.empty()) {
      while (started_.load(std::memory_order_acquire) < nthreads_) {
        std::this_thread::yield();
      }
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  int size() const { return nthreads_; }

  // Zero when every requested pin succeeded, otherwise the error code from the
  // first failing pin attempt (ENOTSUP off Linux).
  int PinError() const { return pin_error_.load(std::memory_order_relaxed); }

  void Run(int n, std::function<void(int, int)> fn) {
    std::unique_lock<std::mutex> lock(mu_);
    fn_ = std::move(fn);
    n_ = n;
    for (int t = 0; t < nthreads_; ++t) next_[t].store(Start(t), std::memory_order_relaxed);
    active_ = nthreads_;
    ++epoch_;
    wake_.notify_all();
    done_.wait(lock, [this] { return active_ == 0; });
    fn_ = nullptr;
  }

 private:
  int Start(int t) const { return static_cast<int>(static_cast<int64_t>(t) * n_ / nthreads_); }

  void Worker(int worker) {
    if (!cpu_ids_.empty()) {
      int err = PinThisThreadToCpu(cpu_ids_[worker]);
      if (err != 0) {
        int expected = 0;
        pin_error_.compare_exchange_strong(expected, err, std::memory_order_relaxed);
      }
      started_.fetch_add(1, std::memory_order_release);
    }
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
      wake_.wait(lock, [this, &seen] { return stop_ || epoch_ != seen; });
      if (stop_) return;
      seen = epoch_;
      const std::function<void(int, int)>* fn = &fn_;
      lock.unlock();
      for (int k = 0; k < nthreads_; ++k) {
        const int t = (worker + k) % nthreads_;
        const int end = Start(t + 1);
        for (int i = next_[t].fetch_add(1, std::memory_order_relaxed); i < end;
             i = next_[t].fetch_add(1, std::memory_order_relaxed)) {
          (*fn)(worker, i);
        }
      }
      lock.lock();
      if (--active_ == 0) done_.notify_one();
    }
  }

  const int nthreads_;
  std::unique_ptr<std::atomic<int>[]> next_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::function<void(int, int)> fn_;
  int n_ = 0;
  int active_ = 0;
  uint64_t epoch_ = 0;
  bool stop_ = false;
  // CPU affinity state (cold path only, final once the constructor returns).
  std::vector<int> cpu_ids_;
  std::atomic<int> started_{0};
  std::atomic<int> pin_error_{0};
};

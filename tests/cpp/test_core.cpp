#include "flexscanx_core/algorithms/scan_flexible.hpp"
#include "flexscanx_core/algorithms/scheduler.hpp"
#include "flexscanx_core/llr_binomial.hpp"
#include "flexscanx_core/llr_poisson.hpp"
#include "flexscanx_core/middle_p_binomial.hpp"
#include "flexscanx_core/middle_p_poisson.hpp"
#include "flexscanx_core/monte_carlo.hpp"
#include "flexscanx_core/scanner_circular_binomial.hpp"
#include "flexscanx_core/scanner_circular_poisson.hpp"
#include "flexscanx_core/scanner_flexible_binomial.hpp"
#include "flexscanx_core/scanner_flexible_poisson.hpp"
#include "flexscanx_core/static_precompute.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using namespace flexscanx_core;

struct Tester {
  int fails = 0;
  int oks = 0;
  void expect_true(bool cond, const std::string &name) {
    if (!cond) {
      std::cout << "FAIL: " << name << "\n";
      ++fails;
    } else {
      std::cout << "OK:   " << name << "\n";
      ++oks;
    }
  }
  void expect_near(double a, double b, double eps, const std::string &name) {
    if (!(std::isfinite(a) && std::isfinite(b))) {
      std::cout << "FAIL: " << name << " (non-finite)\n";
      ++fails;
      return;
    }
    if (std::fabs(a - b) > eps) {
      std::cout << "FAIL: " << name << " (|" << a << "-" << b << "|>" << eps << ")\n";
      ++fails;
    } else {
      std::cout << "OK:   " << name << "\n";
      ++oks;
    }
  }
  template <class Fn> void expect_throws(Fn &&fn, const std::string &name) {
    try {
      fn();
      std::cout << "FAIL: " << name << " (did not throw)\n";
      ++fails;
    } catch (const std::exception &) {
      std::cout << "OK:   " << name << "\n";
      ++oks;
    }
  }
};

int main() {
  Tester T;

  // FXA-127: 100-decimal mpmath fixtures for the likelihood-ratio identities.
  // These high-count, near-null cases amplify cancellation in c1 + c2 - c3;
  // they must retain the long-double Poisson/KL formulation's positive signal.
  {
    T.expect_near(llr_poisson_original_hot(1000001.0, 1000000.0, 2000001.0, 2000000.0),
                  2.4999987500007291662e-7, 1e-18,
                  "Poisson near-null LLR matches arbitrary-precision fixture");
    T.expect_near(llr_binomial_original_hot(500001.0, 1000000.0, 1000001.0, 2000000.0),
                  1.0000000000011666667e-6, 1e-17,
                  "Binomial near-null LLR matches arbitrary-precision fixture");
  }

  // FXA-099/142: null generators keep their valid contracts, while a Poisson
  // mean outside the int-result generator domain fails before it can stall.
  {
    constexpr double unsupported_mean = static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
    T.expect_true(!poisson_mean_supported(unsupported_mean),
                  "Poisson generator marks means above int domain unsupported");
    T.expect_throws([&] { (void)poisson_counts_crn({unsupported_mean}, 1234ULL, 0); },
                    "Poisson CRN generator rejects means above int domain");

    const auto first = poisson_counts_crn({0.0, -1.0, 3.0}, 1234ULL, 7);
    const auto second = poisson_counts_crn({0.0, -1.0, 3.0}, 1234ULL, 7);
    T.expect_true(first == second && first.size() == 3 && first[0] == 0 && first[1] == 0 &&
                      first[2] >= 0,
                  "Poisson CRN null generator is deterministic and handles zero means");

    const auto multinomial = multinomial_counts_crn({1.0, 2.0, 3.0}, 30, 5678ULL, 2);
    T.expect_true(multinomial.size() == 3 && multinomial[0] + multinomial[1] + multinomial[2] == 30,
                  "multinomial CRN null generator preserves requested total");
  }

  // Native worker failures must return to the initiating thread after every
  // started worker is joined.  A second pass proves the scheduler remains
  // usable rather than terminating or retaining poisoned global state.
  for (const bool worksteal : {false, true}) {
    const std::vector<int> centers{0, 1, 2, 3};
    bool propagated = false;
    try {
      auto throw_on_worker = [](int) { throw std::runtime_error("worker failure"); };
      if (worksteal)
        Scheduler::for_each_center_worksteal(centers, 2, throw_on_worker);
      else
        Scheduler::for_each_center_atomic(centers, 2, throw_on_worker);
    } catch (const std::runtime_error &) {
      propagated = true;
    }
    T.expect_true(propagated, worksteal ? "work-steal worker exception propagates after joining"
                                        : "atomic worker exception propagates after joining");

    std::atomic<int> completed{0};
    auto count_center = [&](int) { completed.fetch_add(1, std::memory_order_relaxed); };
    if (worksteal)
      Scheduler::for_each_center_worksteal(centers, 2, count_center);
    else
      Scheduler::for_each_center_atomic(centers, 2, count_center);
    T.expect_true(completed.load(std::memory_order_relaxed) == static_cast<int>(centers.size()),
                  worksteal ? "work-steal scheduler remains usable after worker exception"
                            : "atomic scheduler remains usable after worker exception");
  }

  // The reusable worker-pool seam is used by KNN precomputation and parallel
  // Monte Carlo execution. It must join all started workers, rethrow the first
  // worker exception to the caller, and remain usable afterwards.
  {
    bool propagated = false;
    try {
      Scheduler::for_each_worker(2, [](int, const std::atomic<bool> &) {
        throw std::runtime_error("worker family failure");
      });
    } catch (const std::runtime_error &) {
      propagated = true;
    }
    T.expect_true(propagated, "shared worker-pool exception propagates after joining");

    std::atomic<int> completed{0};
    Scheduler::for_each_worker(2, [&](int, const std::atomic<bool> &cancel) {
      if (!cancel.load(std::memory_order_acquire))
        completed.fetch_add(1, std::memory_order_relaxed);
    });
    T.expect_true(completed.load(std::memory_order_relaxed) == 2,
                  "shared worker-pool remains usable after worker exception");
  }

  // FXA-146: a caller's center-job budget is also the total worker budget for
  // parallel Monte Carlo; task count and available hardware only lower it.
  {
    T.expect_true(Scheduler::bounded_worker_count(/*requested=*/1, /*tasks=*/199,
                                                  /*hardware=*/48) == 1,
                  "MC worker budget honors one requested center job");
    T.expect_true(Scheduler::bounded_worker_count(/*requested=*/4, /*tasks=*/199,
                                                  /*hardware=*/48) == 4,
                  "MC worker budget honors requested center jobs");
    T.expect_true(Scheduler::bounded_worker_count(/*requested=*/8, /*tasks=*/3,
                                                  /*hardware=*/48) == 3,
                  "MC worker budget never exceeds replicate count");
  }

  // The internal Xoshiro generator must satisfy the C++17
  // UniformRandomBitGenerator named requirements used by std distributions.
  {
    using Generator = detail_crn::Xoshiro256ss;
    static_assert(std::is_same_v<Generator::result_type, uint64_t>);
    static_assert(std::is_same_v<decltype(std::declval<Generator &>()()), Generator::result_type>);
    static_assert(std::is_same_v<decltype(Generator::min()), Generator::result_type>);
    static_assert(std::is_same_v<decltype(Generator::max()), Generator::result_type>);
    static_assert(Generator::min() < Generator::max());

    Generator rng;
    rng.seed(20260718ULL);
    std::uniform_int_distribution<int> dist(2, 7);
    bool all_in_range = true;
    for (int i = 0; i < 128; ++i) {
      const int draw = dist(rng);
      all_in_range = all_in_range && draw >= 2 && draw <= 7;
    }
    T.expect_true(all_in_range, "Xoshiro satisfies std distribution generator contract");
  }

  // Poisson middle-p basic properties
  {
    double mu = 3.0;
    // x<0 upper should be 1.0
    T.expect_near(poisson_middle_p_upper(-1, mu), 1.0, 1e-15, "poisson upper x<0 = 1");
    // Identity: upper(ax) + lower(ax) == 1 for ax >= 0
    int ax = 3;
    double up = poisson_middle_p_upper(ax, mu);
    double low = poisson_middle_p_lower(ax, mu);
    T.expect_near(up + low, 1.0, 1e-12, "poisson midp identity up+low=1");
  }

  // FXA-128: 80-decimal incomplete-gamma/PMF fixture at the central
  // high-mean regime where the legacy gamma continued fraction drifts.
  {
    constexpr double mean = 5.0e7;
    constexpr int k = 50000000;
    constexpr double lower_reference = 0.50000940315967669055;
    T.expect_near(poisson_middle_p_lower(k, mean), lower_reference, 1e-9,
                  "Poisson central mid-p matches arbitrary-precision fixture below 1e8");
    T.expect_near(poisson_middle_p_lower(2000000, 2.0e6), 0.50004701579249080639, 1e-9,
                  "Poisson central mid-p meets the smooth handoff error bound");
  }

  // Binomial middle-p basic properties
  {
    int n = 10;
    double p = 0.3;
    // x<0 upper should be 1.0
    T.expect_near(binomial_middle_p_upper(-1, n, p), 1.0, 1e-15, "binomial upper x<0 = 1");
    int ax = 4;
    double up = binomial_middle_p_upper(ax, n, p);
    double low = binomial_middle_p_lower(ax, n, p);
    T.expect_near(up + low, 1.0, 1e-12, "binomial midp identity up+low=1");
  }

  // Poisson LLR behavior
  {
    double nZ = 10, mZ = 5, nG = 100, mG = 100; // rIn=2, rOut~0.947
    double Lhot = llr_poisson_original_hot(nZ, mZ, nG, mG);
    double Lcold = llr_poisson_original_cold(nZ, mZ, nG, mG);
    T.expect_true(Lhot > 0.0, "llr_poisson hot positive when rIn>rOut");
    T.expect_true(Lcold == 0.0, "llr_poisson cold zero when not rIn<rOut");
    // invalid denom
    T.expect_true(llr_poisson_original_hot(5, 0, 10, 10) == 0.0, "llr_poisson hot mZ<=0 -> 0");
  }

  // Binomial LLR behavior
  {
    double nZ = 7, mZ = 10, nG = 100, mG = 200; // pIn=0.7, pOut~0.493
    double Lhot = llr_binomial_original_hot(nZ, mZ, nG, mG);
    double Lcold = llr_binomial_original_cold(nZ, mZ, nG, mG);
    T.expect_true(Lhot > 0.0, "llr_binomial hot positive when pIn>pOut");
    T.expect_true(Lcold == 0.0, "llr_binomial cold zero when not pIn<pOut");
    T.expect_true(llr_binomial_original_hot(5, 0, 10, 10) == 0.0, "llr_binomial hot mZ<=0 -> 0");
  }

  // Additional mid-p corner cases
  {
    double mu = 1.7;
    double pmf0 = std::exp(-mu);
    T.expect_near(poisson_middle_p_lower(0, mu), 0.5 * pmf0, 1e-12,
                  "poisson midp lower at 0 equals 0.5*pmf(0)");
    T.expect_near(poisson_middle_p_lower(-1, mu), 0.0, 1e-15, "poisson midp lower x<0 = 0");
    for (int k = 0; k <= 8; ++k) {
      double up = poisson_middle_p_upper(k, mu);
      double low = poisson_middle_p_lower(k, mu);
      T.expect_near(up + low, 1.0, 1e-12, "poisson midp identity up+low=1 over k range");
    }
    T.expect_near(poisson_middle_p_upper(800, 800.0), 0.4976499772108532, 1e-10,
                  "poisson midp upper stable at large equal mean");
    T.expect_near(poisson_middle_p_lower(800, 800.0), 0.5023500227891468, 1e-10,
                  "poisson midp lower stable at large equal mean");
  }
  // Large-n underflow regime (audit CC-3): (1-p)^n underflows in double for
  // n*|log(1-p)| > ~708; the CDF must survive via the log-space path.
  {
    // n=2000, p=0.5, k=1000 is the exact distribution centre: midp_upper = 0.5.
    T.expect_near(binomial_middle_p_upper(1000, 2000, 0.5), 0.5, 1e-9,
                  "binomial midp upper at large-n centre equals 0.5");
    // Moderate tail values vs scipy.stats.binom references — exactly the
    // ralpha-comparable regime that the underflow corrupted (was ~1.0 on the
    // p<=0.5 path and 0.5*pmf on the complement path).
    T.expect_near(binomial_middle_p_upper(1020, 2000, 0.5), 0.18560846802989878, 1e-9,
                  "binomial midp upper n=2000 k=1020 matches scipy");
    T.expect_near(binomial_middle_p_upper(1040, 2000, 0.5), 0.03684195555268773, 1e-9,
                  "binomial midp upper n=2000 k=1040 matches scipy");
    // Far tail: limited by 1-F cancellation to ~1e-16 resolution, but must be
    // tiny — not clamped to 1.0 (the pre-fix p<=0.5 failure mode) — and the
    // upper/lower identity must hold there too (catches a 0.0-collapse).
    double far = binomial_middle_p_upper(1500, 2000, 0.5);
    T.expect_true(far >= 0.0 && far < 1e-12,
                  "binomial midp upper far tail tiny at n=2000 (no underflow collapse)");
    T.expect_near(far + binomial_middle_p_lower(1500, 2000, 0.5), 1.0, 1e-9,
                  "binomial midp identity up+low=1 at far tail n=2000");
    // Identity still holds in the underflow regime.
    double up2k = binomial_middle_p_upper(1100, 2000, 0.5);
    double low2k = binomial_middle_p_lower(1100, 2000, 0.5);
    T.expect_near(up2k + low2k, 1.0, 1e-9, "binomial midp identity up+low=1 at n=2000");
  }

  {
    int n = 10;
    double p = 0.3;
    double p0 = std::pow(1.0 - p, n);
    T.expect_near(binomial_middle_p_lower(0, n, p), 0.5 * p0, 1e-12,
                  "binomial midp lower at 0 equals 0.5*(1-p)^n");
    T.expect_near(binomial_middle_p_lower(-1, n, p), 0.0, 1e-15, "binomial midp lower x<0 = 0");
    for (int k = 0; k <= n; ++k) {
      double up = binomial_middle_p_upper(k, n, p);
      double low = binomial_middle_p_lower(k, n, p);
      T.expect_near(up + low, 1.0, 1e-12, "binomial midp identity up+low=1 over k range");
    }
  }

  // Restricted center gating tests (circular Poisson)
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/3, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    // HOT gating: make only center 0 potentially admissible when ralpha >= ~0.003
    // Observed vs expected
    std::vector<int> obs_hot{5, 0, 0};
    std::vector<double> exp_hot{1.0, 1.0, 1.0};

    ScanSettings stH;
    stH.clustersize = 3;
    stH.ralpha = 0.001; // too strict: all centers fail
    auto selH0 =
        scan_circular_poisson(*pre, obs_hot, exp_hot, stH,
                              /*restricted=*/true, /*hot=*/true,
                              /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(selH0.area.empty() && selH0.stats == 0.0,
                  "restricted HOT centers all inadmissible -> empty");

    stH.ralpha = 0.01; // allow center 0
    auto selH1 =
        scan_circular_poisson(*pre, obs_hot, exp_hot, stH,
                              /*restricted=*/true, /*hot=*/true,
                              /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selH1.area.empty() && selH1.area[0] == 0 && selH1.stats > 0.0,
                  "restricted HOT admissible center yields non-empty cluster");
  }

  // Original stattype: circular Poisson HOT, flexible Binomial HOT/COLD
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/3, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    std::vector<int> obsP{10, 0, 0};
    std::vector<double> expP{1.0, 1.0, 1.0};
    ScanSettings stO;
    stO.clustersize = 3;
    auto selPO =
        scan_circular_poisson(*pre, obsP, expP, stO,
                              /*restricted=*/false, /*hot=*/true,
                              /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selPO.area.empty() && selPO.area[0] == 0 && selPO.stats > 0.0,
                  "original circular Poisson HOT yields selection");

    std::vector<int> obsB1{80, 0, 0};
    std::vector<double> pop{100.0, 100.0, 100.0};
    auto selBFH =
        scan_flexible_hot_original_binomial(*pre, obsB1, pop, stO,
                                            /*excluded=*/nullptr, /*center_order=*/nullptr);
    T.expect_true(!selBFH.area.empty() && selBFH.stats > 0.0,
                  "original flexible Binomial HOT yields selection");

    std::vector<int> obsB2{0, 80, 80};
    auto selBFC =
        scan_flexible_cold_original_binomial(*pre, obsB2, pop, stO,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr);
    T.expect_true(!selBFC.area.empty() && selBFC.stats > 0.0,
                  "original flexible Binomial COLD yields selection");
  }

  {
    using namespace flexscanx_core;
    // COLD gating: center 0 has deficit; others have surfeit
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/3, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    std::vector<int> obs_cold{0, 5, 5};
    std::vector<double> exp_cold{1.0, 1.0, 1.0};

    ScanSettings stC;
    stC.clustersize = 3;
    stC.ralpha = 0.2; // admits center 0 for cold
    auto selC =
        scan_circular_poisson(*pre, obs_cold, exp_cold, stC,
                              /*restricted=*/true, /*hot=*/false,
                              /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selC.area.empty() && selC.area[0] == 0 && selC.stats > 0.0,
                  "restricted COLD center gating and prefix stop");
  }

  // Binomial circular restricted gating (HOT and COLD)
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/3, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    // HOT: center 0 very high vs pTot
    std::vector<int> obs_hot{80, 0, 0};
    std::vector<double> pop{100.0, 100.0, 100.0};
    double nG = 80.0, mG = 300.0;
    double p0 = nG / mG;
    double ph0 = binomial_middle_p_upper(obs_hot[0], (int)std::llround(pop[0]), p0);

    ScanSettings stBH;
    stBH.clustersize = 3;
    stBH.ralpha = ph0 / 10.0; // too strict => center 0 fails
    auto selBH0 =
        scan_circular_binomial(*pre, obs_hot, pop, stBH,
                               /*restricted=*/true, /*hot=*/true,
                               /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(selBH0.area.empty(), "binomial circular HOT too strict ralpha -> empty");

    stBH.ralpha = ph0 * 10.0; // allow center 0
    auto selBH1 =
        scan_circular_binomial(*pre, obs_hot, pop, stBH,
                               /*restricted=*/true, /*hot=*/true,
                               /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selBH1.area.empty() && selBH1.area[0] == 0,
                  "binomial circular HOT admissible center");
  }
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/3, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    // COLD: center 0 very low vs pTot
    std::vector<int> obs_coldB{0, 80, 80};
    std::vector<double> pop{100.0, 100.0, 100.0};
    double nG = 160.0, mG = 300.0;
    double p0 = nG / mG;
    double pc0 = binomial_middle_p_lower(obs_coldB[0], (int)std::llround(pop[0]), p0);

    ScanSettings stBC;
    stBC.clustersize = 3;
    stBC.ralpha = pc0 / 10.0; // too strict => center 0 fails
    auto selBC0 =
        scan_circular_binomial(*pre, obs_coldB, pop, stBC,
                               /*restricted=*/true, /*hot=*/false,
                               /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(selBC0.area.empty(), "binomial circular COLD too strict ralpha -> empty");

    stBC.ralpha = pc0 * 10.0; // allow center 0
    auto selBC1 =
        scan_circular_binomial(*pre, obs_coldB, pop, stBC,
                               /*restricted=*/true, /*hot=*/false,
                               /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selBC1.area.empty() && selBC1.area[0] == 0,
                  "binomial circular COLD admissible center");
  }

  // Secondary enumeration (manual exclude) using flexible Poisson restricted HOT
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 2.0, 4.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0, 0};
    csr.col_idx.clear(); // no adjacency -> singletons
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{20, 15, 0};
    std::vector<double> exp{1.0, 1.0, 1.0};
    ScanSettings stS;
    stS.clustersize = 2;
    stS.ralpha = 0.2;

    auto sel1 = scan_flexible_hot_restricted_poisson(*pre, obs, exp, stS,
                                                     /*excluded=*/nullptr, /*center_order=*/nullptr,
                                                     /*gating=*/nullptr);
    T.expect_true(!sel1.area.empty() && sel1.stats > 0.0, "secondary sel1 valid");
    std::vector<char> excl(pre->N, 0);
    for (int v : sel1.area)
      excl[v] = 1;
    auto sel2 = scan_flexible_hot_restricted_poisson(*pre, obs, exp, stS,
                                                     /*excluded=*/&excl, /*center_order=*/nullptr,
                                                     /*gating=*/nullptr);
    bool disjoint = true;
    for (int v : sel2.area)
      if (v < (int)excl.size() && excl[v])
        disjoint = false;
    T.expect_true(!sel2.area.empty() && sel2.stats > 0.0 && disjoint,
                  "secondary sel2 disjoint and valid");
  }

  // Flexible Poisson restricted gating (HOT and COLD)
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0}, y{0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{10, 0};
    std::vector<double> exp{1.0, 1.0};
    double ph0 = poisson_middle_p_upper(obs[0], exp[0]);

    ScanSettings stFH;
    stFH.clustersize = 2;
    stFH.ralpha = ph0 / 10.0; // too strict => center 0 fails
    auto selFH0 =
        scan_flexible_hot_restricted_poisson(*pre, obs, exp, stFH,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr,
                                             /*gating=*/nullptr);
    T.expect_true(selFH0.area.empty(), "flexible Poisson HOT too strict ralpha -> empty");

    stFH.ralpha = ph0 * 10.0; // allow center 0
    auto selFH1 =
        scan_flexible_hot_restricted_poisson(*pre, obs, exp, stFH,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr,
                                             /*gating=*/nullptr);
    T.expect_true(!selFH1.area.empty() && selFH1.area[0] == 0,
                  "flexible Poisson HOT admissible center");
  }
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 1, 3, 4};
    csr.col_idx = {1, 0, 2, 1};
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{900, 800, 800};
    std::vector<double> exp{800.0, 800.0, 800.0};

    ScanSettings stFH;
    stFH.clustersize = 2;
    stFH.ralpha = 0.20; // historical repro: this used to collapse to empty under underflow
    auto selFH1 =
        scan_flexible_hot_restricted_poisson(*pre, obs, exp, stFH,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr,
                                             /*gating=*/nullptr);
    bool contains0 = false;
    for (int v : selFH1.area)
      contains0 = contains0 || (v == 0);
    T.expect_true(!selFH1.area.empty() && contains0 && selFH1.stats > 0.0,
                  "flexible Poisson HOT high-count restricted remains admissible");
  }
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0}, y{0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{0, 10};
    std::vector<double> exp{1.0, 1.0};
    double pc0 = poisson_middle_p_lower(obs[0], exp[0]);

    ScanSettings stFC;
    stFC.clustersize = 2;
    stFC.ralpha = pc0 / 10.0; // too strict => center 0 fails
    auto selFC0 =
        scan_flexible_cold_restricted_poisson(*pre, obs, exp, stFC,
                                              /*excluded=*/nullptr, /*center_order=*/nullptr,
                                              /*gating=*/nullptr);
    T.expect_true(selFC0.area.empty(), "flexible Poisson COLD too strict ralpha -> empty");

    stFC.ralpha = pc0 * 10.0; // allow center 0
    auto selFC1 =
        scan_flexible_cold_restricted_poisson(*pre, obs, exp, stFC,
                                              /*excluded=*/nullptr, /*center_order=*/nullptr,
                                              /*gating=*/nullptr);
    T.expect_true(!selFC1.area.empty() && selFC1.area[0] == 0,
                  "flexible Poisson COLD admissible center");
  }

  // K=1 property test: best cluster equals best single node (Original) — Poisson
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0, 2.0}, y{0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 1, 2, 3};
    csr.col_idx = {0, 1, 2}; // self-loops for completeness
    auto pre = build_static(x, y, csr, /*K=*/1, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{5, 10, 7};
    std::vector<double> exp{1.0, 1.0, 1.0};
    ScanSettings stK;
    stK.clustersize = 3; // K=1 limits to center only regardless

    // Best single node by LLR
    double nG = 0.0, mG = 0.0;
    for (int i = 0; i < 3; ++i) {
      nG += obs[i];
      mG += exp[i];
    }
    int argmax = 0;
    double best = -1e99;
    for (int i = 0; i < 3; ++i) {
      double v = llr_poisson_original_hot(obs[i], exp[i], nG, mG);
      if (v > best) {
        best = v;
        argmax = i;
      }
    }

    auto sel = scan_flexible_hot_original_poisson(*pre, obs, exp, stK,
                                                  /*excluded=*/nullptr,
                                                  /*center_order=*/nullptr);
    T.expect_true(sel.area.size() == 1 && sel.area[0] == argmax &&
                      std::fabs(sel.stats - best) < 1e-12,
                  "K=1 Poisson flexible original equals best single LLR");
  }

  // Root frontiers share the reusable DFS pool, so repeated single-node scans
  // must reach a stable retained-vector count after warm-up.
  {
    std::vector<double> x{0.0, 1.0, 2.0, 3.0}, y{0.0, 0.0, 0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 1, 3, 5, 6};
    csr.col_idx = {1, 0, 2, 1, 3, 2};
    auto pre = build_static(x, y, csr, /*K=*/4, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);
    std::vector<int> obs{9, 1, 1, 1};
    std::vector<double> exp{1.0, 1.0, 1.0, 1.0};
    ScanSettings st;
    st.clustersize = 1;
    st.n_jobs_centers = 1;

    const std::size_t retained_before = detail::flexible_frontier_pool().size();
    (void)scan_flexible_hot_original_poisson(*pre, obs, exp, st,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr);
    const std::size_t retained_after_first = detail::flexible_frontier_pool().size();
    (void)scan_flexible_hot_original_poisson(*pre, obs, exp, st,
                                             /*excluded=*/nullptr, /*center_order=*/nullptr);
    const std::size_t retained_after_second = detail::flexible_frontier_pool().size();

    T.expect_true(retained_after_first <= retained_before + 1 &&
                      retained_after_second == retained_after_first,
                  "flexible root frontiers reach a bounded thread-local pool after warm-up");
  }

  // Flexible Binomial restricted gating (HOT and COLD)
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0}, y{0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    std::vector<int> obsB{80, 0};
    std::vector<double> pop{100.0, 100.0};
    double nG = 80.0, mG = 200.0;
    double p0 = nG / mG;
    double ph0 = binomial_middle_p_upper(obsB[0], (int)std::llround(pop[0]), p0);

    ScanSettings stFBH;
    stFBH.clustersize = 2;
    stFBH.ralpha = ph0 / 10.0; // too strict => center 0 fails
    auto selFBH0 = scan_flexible_hot_restricted_binomial(
        *pre, obsB, pop, stFBH,
        /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(selFBH0.area.empty(), "flexible Binomial HOT too strict ralpha -> empty");

    stFBH.ralpha = ph0 * 10.0; // allow center 0
    auto selFBH1 = scan_flexible_hot_restricted_binomial(
        *pre, obsB, pop, stFBH,
        /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selFBH1.area.empty() && selFBH1.area[0] == 0,
                  "flexible Binomial HOT admissible center");
  }
  {
    using namespace flexscanx_core;
    std::vector<double> x{0.0, 1.0}, y{0.0, 0.0};
    CSR csr;
    csr.row_ptr = {0, 0, 0};
    csr.col_idx.clear();
    auto pre = build_static(x, y, csr, /*K=*/2, std::numeric_limits<double>::infinity(),
                            KNNMode::K_FIXED, /*force_refill=*/false);

    std::vector<int> obsB{0, 80};
    std::vector<double> pop{100.0, 100.0};
    double nG = 80.0, mG = 200.0;
    double p0 = nG / mG;
    double pc0 = binomial_middle_p_lower(obsB[0], (int)std::llround(pop[0]), p0);

    ScanSettings stFBC;
    stFBC.clustersize = 2;
    stFBC.ralpha = pc0 / 10.0; // too strict => center 0 fails
    auto selFBC0 = scan_flexible_cold_restricted_binomial(
        *pre, obsB, pop, stFBC,
        /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(selFBC0.area.empty(), "flexible Binomial COLD too strict ralpha -> empty");

    stFBC.ralpha = pc0 * 10.0; // allow center 0
    auto selFBC1 = scan_flexible_cold_restricted_binomial(
        *pre, obsB, pop, stFBC,
        /*excluded=*/nullptr, /*center_order=*/nullptr, /*gating=*/nullptr);
    T.expect_true(!selFBC1.area.empty() && selFBC1.area[0] == 0,
                  "flexible Binomial COLD admissible center");
  }

  // FXA-187 / D4: tie at the K-limited universe boundary is detected per center.
  {
    using namespace flexscanx_core;
    // 3x3 unit lattice, rows 0..8 = (x=i%3, y=i/3).
    std::vector<double> x, y;
    for (int i = 0; i < 9; ++i) {
      x.push_back(static_cast<double>(i % 3));
      y.push_back(static_cast<double>(i / 3));
    }
    CSR csr;
    csr.row_ptr.assign(10, 0);
    csr.col_idx.clear();
    for (KNNMode mode : {KNNMode::K_FIXED, KNNMode::K_REFILL}) {
      // K=5: corner and edge centers have a tie at the boundary (distance 2 or
      // sqrt(2) shared by the 4th and 5th nearest); the middle center's four
      // orthogonal neighbours fill K-1 exactly, so 8 of 9 centers are flagged.
      auto pre5 = build_static(x, y, csr, /*K=*/5, std::numeric_limits<double>::infinity(), mode,
                               /*force_refill=*/false);
      T.expect_true(pre5->knn_tie_boundary_count() == 8,
                    "3x3 lattice K=5 flags 8 tie-boundary centers");
      T.expect_true(pre5->knn_tie_at_boundary[4] == 0, "3x3 lattice K=5 middle center not flagged");
      // K=N: no neighbour is excluded, so no boundary and no tie.
      auto pre9 = build_static(x, y, csr, /*K=*/9, std::numeric_limits<double>::infinity(), mode,
                               /*force_refill=*/false);
      T.expect_true(pre9->knn_tie_boundary_count() == 0, "3x3 lattice K=N flags nothing");
      // A radius below the tied boundary distance excludes both tied neighbours.
      auto pre5r = build_static(x, y, csr, /*K=*/5, /*radius=*/1.2, mode, /*force_refill=*/false);
      T.expect_true(pre5r->knn_tie_at_boundary[0] == 0,
                    "corner tie at distance 2 beyond radius 1.2 is not flagged");
    }
    // Strictly tie-free coordinates never flag.
    std::vector<double> xj, yj;
    for (int i = 0; i < 9; ++i) {
      xj.push_back(static_cast<double>(i % 3) + 1e-3 * (i + 1));
      yj.push_back(static_cast<double>(i / 3) + 7e-4 * (i * i + 1));
    }
    auto prej = build_static(xj, yj, csr, /*K=*/5, std::numeric_limits<double>::infinity(),
                             KNNMode::K_FIXED, /*force_refill=*/false);
    T.expect_true(prej->knn_tie_boundary_count() == 0, "jittered lattice flags no tie boundary");
  }

  // FXA-188 / D6: the schedulers poll on the calling thread and unwind cleanly
  // when the poll throws; worker 0 of a worker family runs on the caller.
  {
    using namespace flexscanx_core;
    std::vector<int> centers(64);
    for (int i = 0; i < 64; ++i)
      centers[i] = i;
    const std::thread::id caller = std::this_thread::get_id();
    for (int backend = 0; backend < 2; ++backend) {
      std::atomic<int> processed{0};
      std::atomic<int> polls{0};
      std::atomic<int> polls_off_thread{0};
      auto fn = [&](int) { processed.fetch_add(1, std::memory_order_relaxed); };
      auto counting_poll = [&]() {
        polls.fetch_add(1, std::memory_order_relaxed);
        if (std::this_thread::get_id() != caller)
          polls_off_thread.fetch_add(1, std::memory_order_relaxed);
      };
      std::function<void()> poll_fn = counting_poll;
      if (backend == 0)
        Scheduler::for_each_center_atomic(centers, 4, fn, poll_fn);
      else
        Scheduler::for_each_center_worksteal(centers, 4, fn, poll_fn);
      T.expect_true(processed.load() == 64, "scheduler with poll processes every center");
      T.expect_true(polls.load() >= 1 && polls_off_thread.load() == 0,
                    "scheduler polls only on the calling thread");

      std::atomic<int> processed2{0};
      std::atomic<int> polls2{0};
      auto fn2 = [&](int) {
        processed2.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      };
      std::function<void()> throwing_poll = [&]() {
        if (polls2.fetch_add(1, std::memory_order_relaxed) == 2)
          throw std::runtime_error("interrupt");
      };
      T.expect_throws(
          [&]() {
            if (backend == 0)
              Scheduler::for_each_center_atomic(centers, 4, fn2, throwing_poll);
            else
              Scheduler::for_each_center_worksteal(centers, 4, fn2, throwing_poll);
          },
          "throwing poll unwinds through the scheduler");
      T.expect_true(processed2.load() < 64, "throwing poll cancels the remaining centers");
      // The scheduler stays usable afterwards (all threads were joined).
      std::atomic<int> processed3{0};
      auto fn3 = [&](int) { processed3.fetch_add(1, std::memory_order_relaxed); };
      if (backend == 0)
        Scheduler::for_each_center_atomic(centers, 4, fn3);
      else
        Scheduler::for_each_center_worksteal(centers, 4, fn3);
      T.expect_true(processed3.load() == 64, "scheduler usable after a poll interrupt");
      // Empty poll (default) still processes everything on the T==1 path.
      std::atomic<int> processed4{0};
      auto fn4 = [&](int) { processed4.fetch_add(1, std::memory_order_relaxed); };
      if (backend == 0)
        Scheduler::for_each_center_atomic(centers, 1, fn4);
      else
        Scheduler::for_each_center_worksteal(centers, 1, fn4);
      T.expect_true(processed4.load() == 64, "single-thread path unchanged without a poll");
    }
    std::thread::id worker0_thread;
    std::atomic<int> ran{0};
    Scheduler::for_each_worker(4, [&](int t, const std::atomic<bool> &) {
      if (t == 0)
        worker0_thread = std::this_thread::get_id();
      ran.fetch_add(1, std::memory_order_relaxed);
    });
    T.expect_true(ran.load() == 4 && worker0_thread == caller,
                  "worker family runs worker 0 on the calling thread");
    T.expect_throws(
        [&]() {
          Scheduler::for_each_worker(3, [&](int t, const std::atomic<bool> &) {
            if (t == 0)
              throw std::runtime_error("caller worker failure");
          });
        },
        "caller-thread worker exception is rethrown after joining");
  }

  std::cout << "\nSummary: " << T.oks << " OK, " << T.fails << " FAIL\n";
  return T.fails == 0 ? 0 : 1;
}

/**
 * pi_cpu.cpp - CPU/OpenMP backend for pi integration.
 *
 * Midpoint-rule integration of 4/(1+x²) over [0,1] with Kahan summation,
 * parallelized via OpenMP.  Supports core affinity pinning with --cores.
 */

#include "backends.h"

#include <cstdlib>
#include <algorithm>
#include <thread>
#include <gmp.h>

#ifdef HAS_OPENMP
#  include <omp.h>
#endif

#ifdef __linux__
#  include <sched.h>
#  include <fstream>
#  include <sstream>
#endif

#ifdef _WIN32
#  include <windows.h>
#endif

// ===================================================================
//  CPU info
// ===================================================================

int cpu_core_count() {
#ifdef HAS_OPENMP
    return omp_get_max_threads();
#else
    return (int)std::thread::hardware_concurrency();
#endif
}

std::string cpu_model_name() {
#ifdef __linux__
    std::ifstream f("/proc/cpuinfo");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string name = line.substr(pos + 1);
                    // Trim leading whitespace
                    size_t start = name.find_first_not_of(" \t");
                    if (start != std::string::npos) name = name.substr(start);
                    return name;
                }
            }
        }
    }
    return "Unknown CPU";
#elif defined(_WIN32)
    // Read from registry or just return generic
    return "x86_64 CPU";
#else
    return "Unknown CPU";
#endif
}

// ===================================================================
//  CPU integration with OpenMP
// ===================================================================

ComputeResult cpu_integrate(long long N, const std::vector<int>& cores) {
#ifdef HAS_OPENMP
    int num_threads;

    if (!cores.empty()) {
        num_threads = (int)cores.size();
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
    }

    double step = 1.0 / (double)N;

    // Split computation into chunks for progress reporting.
    // Each chunk runs a parallel region; overhead is negligible for large chunks.
    const int NUM_CHUNKS = 100;
    long long chunk_size = (N + NUM_CHUNKS - 1) / NUM_CHUNKS;

    double total_sum  = 0.0;
    double total_comp = 0.0;

    auto wall_start = std::chrono::steady_clock::now();

    for (int c = 0; c < NUM_CHUNKS; c++) {
        long long c_start = (long long)c * chunk_size;
        long long c_end   = std::min(c_start + chunk_size, N);
        if (c_start >= N) break;

        double chunk_sum = 0.0;

        // Set core affinity inside the first chunk's parallel region
        #pragma omp parallel reduction(+:chunk_sum)
        {
            // Pin thread to core if --cores was specified
            if (!cores.empty() && c == 0) {
                int tid = omp_get_thread_num();
                if (tid < (int)cores.size()) {
#ifdef __linux__
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(cores[tid], &cpuset);
                    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
                    SetThreadAffinityMask(GetCurrentThread(),
                                          (DWORD_PTR)1 << cores[tid]);
#endif
                }
            }

            // Each thread uses Kahan summation over its portion
            double local_sum  = 0.0;
            double local_comp = 0.0;

            #pragma omp for schedule(static) nowait
            for (long long i = c_start; i < c_end; i++) {
                double x = ((double)i + 0.5) * step;
                double f = 4.0 / (1.0 + x * x);
                double y = f - local_comp;
                double t = local_sum + y;
                local_comp = (t - local_sum) - y;
                local_sum  = t;
            }

            chunk_sum += local_sum;
        }

        // Kahan accumulation across chunks
        double y = chunk_sum - total_comp;
        double t = total_sum + y;
        total_comp = (t - total_sum) - y;
        total_sum  = t;

        // Progress bar update
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - wall_start).count();
        ui_progress_bar("Computing \xcf\x80 (CPU)", (double)(c + 1) / NUM_CHUNKS, elapsed_s);
    }

    auto wall_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    total_sum *= step;

    ui_progress_done("Computing \xcf\x80 (CPU)", elapsed_ms);

    ComputeResult res;
    res.pi_value       = total_sum;
    res.elapsed_ms     = elapsed_ms;
    res.intervals      = N;
    res.device_name    = cpu_model_name();
    res.num_threads    = num_threads;
    res.throughput_gips = (double)N / (elapsed_ms * 1e6);
    return res;

#else // !HAS_OPENMP - single-threaded fallback

    fprintf(stderr, "  %sNote:%s OpenMP not available; running single-threaded.\n",
            rgb_str(clr_rgb::warning), clr(CLR_RESET));

    double step = 1.0 / (double)N;
    double sum  = 0.0;
    double comp = 0.0;

    auto wall_start = std::chrono::steady_clock::now();

    long long report_interval = N / 100;
    if (report_interval < 1) report_interval = 1;

    for (long long i = 0; i < N; i++) {
        double x = ((double)i + 0.5) * step;
        double f = 4.0 / (1.0 + x * x);
        double y = f - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;

        if (i % report_interval == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - wall_start).count();
            ui_progress_bar("Computing \xcf\x80 (CPU, single-thread)", (double)i / N, elapsed_s);
        }
    }
    sum *= step;

    auto wall_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ui_progress_done("Computing \xcf\x80 (CPU)", elapsed_ms);

    ComputeResult res;
    res.pi_value       = sum;
    res.elapsed_ms     = elapsed_ms;
    res.intervals      = N;
    res.device_name    = cpu_model_name();
    res.num_threads    = 1;
    res.throughput_gips = (double)N / (elapsed_ms * 1e6);
    return res;
#endif
}

// ===================================================================
//  GMP-based arbitrary-precision digit computation
//
//  Chudnovsky algorithm - both iterative and binary splitting modes.
//  Uses GMP (mpz_t for integers, mpf_t for final floating-point).
//
//  1/π = 12 Σ [(-1)^k (6k)! (13591409 + 545140134k)]
//                / [(3k)! (k!)^3 640320^(3k+3/2)]
//
//  Binary splitting computes P(a,b), Q(a,b), T(a,b) recursively,
//  then π = (Q * 426880 * sqrt(10005)) / T
// ===================================================================

// ------------------------------------------------------------------
//  GMP-based helper: extract base-10^9 limbs from mpf_t
//
//  Converts an mpf_t value to the pi_digits[] array (MSB-first,
//  base-10^9 limbs, limb[0] = integer part).
// ------------------------------------------------------------------

static void extract_pi_digits(uint32_t* pi_digits, int n_limbs,
                               mpf_t pi_val, long long n_digits) {
    // Get the decimal string representation.
    // mpf_get_str returns significand digits (no decimal point) + exponent.
    // For pi ≈ 3.14159..., exponent = 1, str = "314159265..."
    mp_exp_t exponent;
    long long total_digits = (long long)n_limbs * 9 + 20;
    char* str = mpf_get_str(nullptr, &exponent, 10, total_digits, pi_val);

    const char* digits = str;
    if (*digits == '-') digits++;
    size_t slen = strlen(digits);

    // The output format is: pi_digits[0] = integer part, pi_digits[1..] = fractional
    // groups of 9 decimal digits each.  The decimal point sits after 'exponent' digits
    // in the str.  We need to pad the integer part to fill a 9-digit group
    // (left-padded with zeros), then pack the remaining digits in 9-digit groups.
    //
    // For pi: exponent=1, so integer part is "3" -> limb[0]=3,
    // fractional digits: "14159265358979..." -> limb[1]=141592653, limb[2]=589793238, ...

    // Buffer: 9 zeros for integer-part padding, then the significant digits
    int int_pad = 9 - (int)exponent;  // zeros to prepend for alignment
    if (int_pad < 0) int_pad = 0;

    long long buf_len = (long long)n_limbs * 9;
    char* buf = new char[buf_len + 1];
    memset(buf, '0', buf_len);
    buf[buf_len] = '\0';

    // Copy digits into buffer with alignment padding
    long long copy_len = std::min((long long)slen, buf_len - int_pad);
    if (copy_len > 0) {
        memcpy(buf + int_pad, digits, copy_len);
    }

    // Extract 9-digit groups as base-10^9 limbs
    for (int i = 0; i < n_limbs; i++) {
        uint32_t limb = 0;
        for (int j = 0; j < 9; j++) {
            limb = limb * 10 + (buf[i * 9 + j] - '0');
        }
        pi_digits[i] = limb;
    }

    delete[] buf;
    free(str);  // mpf_get_str allocates with malloc
}

// ------------------------------------------------------------------
//  cpu_compute_pi_digits - Chudnovsky iterative recurrence (GMP)
//
//  Uses scaled-integer arithmetic with mpz_t.  Each term is updated
//  from the previous via scalar multiply/divide (same recurrence as
//  the CUDA version), then accumulated into sum.  Final result uses
//  mpf_t for sqrt(10005) and the final division.
// ------------------------------------------------------------------

void cpu_compute_pi_digits(uint32_t* pi_digits, int n_limbs,
                            long long n_digits, const std::vector<int>& cores) {
#ifdef HAS_OPENMP
    if (!cores.empty()) {
        omp_set_num_threads((int)cores.size());
    }
#endif
    (void)cores;  // Used above for OpenMP thread count

    // Number of Chudnovsky terms needed (~14.18 digits per term)
    int n_terms = (int)(n_digits / 14.0) + 10;

    // GMP precision: enough bits for n_digits decimal digits + guard digits
    mp_bitcnt_t prec_bits = (mp_bitcnt_t)((n_digits + 100) * 3.3219281 + 256);

    auto start_time = std::chrono::steady_clock::now();

    // Scale factor: 10^(n_digits + 50) - work in scaled integers to
    // maintain precision through the iterative recurrence
    mpz_t scale;
    mpz_init(scale);
    mpz_ui_pow_ui(scale, 10, (unsigned long)(n_digits + 50));

    // term(0) = scale (i.e. 1.0 in scaled representation)
    mpz_t term, sum, contrib, temp;
    mpz_init_set(term, scale);
    mpz_init(sum);
    mpz_init(contrib);
    mpz_init(temp);

    // k=0 contribution: (+1) * 1 * 13591409 = 13591409 * scale
    mpz_mul_ui(sum, scale, 13591409UL);

    int report_interval = n_terms / 100;
    if (report_interval < 1) report_interval = 1;

    for (int k = 1; k <= n_terms; k++) {
        // Update term: multiply by (6k-5)(2k-1)(6k-1)
        unsigned long num1 = (unsigned long)(6*k - 5);
        unsigned long num2 = (unsigned long)(2*k - 1);
        unsigned long num3 = (unsigned long)(6*k - 1);

        mpz_mul_ui(term, term, num1);
        mpz_mul_ui(term, term, num2);
        mpz_mul_ui(term, term, num3);

        // Divide by k^3 * (640320^3 / 24) split into:
        // /k /k /k /26680 /640320 /640320
        unsigned long kk = (unsigned long)k;
        mpz_tdiv_q_ui(term, term, kk);
        mpz_tdiv_q_ui(term, term, kk);
        mpz_tdiv_q_ui(term, term, kk);
        mpz_tdiv_q_ui(term, term, 26680UL);
        mpz_tdiv_q_ui(term, term, 640320UL);
        mpz_tdiv_q_ui(term, term, 640320UL);

        // Compute contribution: term * (13591409 + 545140134*k)
        // contrib = term * 545140134 * k + term * 13591409
        mpz_mul_ui(contrib, term, 545140134UL);
        mpz_mul_ui(contrib, contrib, kk);
        mpz_mul_ui(temp, term, 13591409UL);
        mpz_add(contrib, contrib, temp);

        // Accumulate: even k adds, odd k subtracts
        if (k % 2 == 0) {
            mpz_add(sum, sum, contrib);
        } else {
            mpz_sub(sum, sum, contrib);
        }

        if (k % report_interval == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start_time).count();
            ui_progress_bar("Chudnovsky series (CPU)", (double)k / n_terms, elapsed_s);
        }
    }

    auto series_end = std::chrono::steady_clock::now();
    double series_ms = std::chrono::duration<double, std::milli>(series_end - start_time).count();
    ui_progress_done("Chudnovsky series (CPU)", series_ms);

    // Final: π = 426880 * sqrt(10005) * scale / sum
    fprintf(stderr, "  %sComputing final value (sqrt + division)...%s\n",
            rgb_str(clr_rgb::dim), clr(CLR_RESET));
    auto final_start = std::chrono::steady_clock::now();

    mpf_t pi_val, sqrt_10005, fsum, fscale, ftemp;
    mpf_init2(pi_val, prec_bits);
    mpf_init2(sqrt_10005, prec_bits);
    mpf_init2(fsum, prec_bits);
    mpf_init2(fscale, prec_bits);
    mpf_init2(ftemp, prec_bits);

    // sqrt(10005)
    mpf_set_ui(ftemp, 10005);
    mpf_sqrt(sqrt_10005, ftemp);

    // pi = 426880 * sqrt(10005) * scale / sum
    mpf_set_z(fsum, sum);
    mpf_set_z(fscale, scale);
    mpf_mul_ui(pi_val, sqrt_10005, 426880UL);
    mpf_mul(pi_val, pi_val, fscale);
    mpf_div(pi_val, pi_val, fsum);

    auto final_end = std::chrono::steady_clock::now();
    double final_ms = std::chrono::duration<double, std::milli>(final_end - final_start).count();
    ui_spinner_done("Final computation", final_ms);

    // Extract base-10^9 limbs
    extract_pi_digits(pi_digits, n_limbs, pi_val, n_digits);

    // Cleanup
    mpf_clear(pi_val);
    mpf_clear(sqrt_10005);
    mpf_clear(fsum);
    mpf_clear(fscale);
    mpf_clear(ftemp);
    mpz_clear(scale);
    mpz_clear(term);
    mpz_clear(sum);
    mpz_clear(contrib);
    mpz_clear(temp);
}

// ===================================================================
//  Binary Splitting Chudnovsky - CPU only (GMP)
//
//  Recursive tree structure computes P, Q, T as arbitrary-precision
//  big integers using GMP, then converts to fixed-point for the final
//  result.  OpenMP tasks parallelize left/right branches at top levels.
//
//  Formula: pi = 426880 * sqrt(10005) * Q(0,N) / T(0,N)
//
//  For range [a, b):
//    Base (b == a+1):
//      a==0: P=1, Q=1, T=A
//      a >0: P = -(6a-5)(2a-1)(6a-1)
//             Q = a^3 * (C^3/24)
//             T = P * (A + B*a)
//    Merge at mid = (a+b)/2:
//      P(a,b) = P(a,mid) * P(mid,b)
//      Q(a,b) = Q(a,mid) * Q(mid,b)
//      T(a,b) = T(a,mid) * Q(mid,b) + P(a,mid) * T(mid,b)
//
//  Where A=13591409, B=545140134, C=640320, C^3/24=10939058860032000
// ===================================================================

struct BSResult {
    mpz_t P, Q, T;
};

static void bs_init(BSResult& r) {
    mpz_init(r.P);
    mpz_init(r.Q);
    mpz_init(r.T);
}

static void bs_clear(BSResult& r) {
    mpz_clear(r.P);
    mpz_clear(r.Q);
    mpz_clear(r.T);
}

// Recursion depth above which we stop spawning OpenMP tasks.
// Each level doubles the tasks, so depth 6 = 64 tasks.
// Recursion depth threshold for OpenMP task spawning.
// Depth D creates 2^D leaf tasks.  Set high enough so that every core stays
// busy even during the heavy merge phases at the top of the tree.
// 9 -> 512 leaf tasks (~16 per core on 32-thread CPUs), which keeps all cores
// saturated through the expensive merge phases at upper tree levels.
static const int OMP_TASK_DEPTH = 9;

static const long CHUD_A  = 13591409L;
static const long CHUD_B  = 545140134L;

static void chudnovsky_bs(BSResult& r, int a, int b, int depth) {
    if (b - a == 1) {
        // Base case: single term
        if (a == 0) {
            mpz_set_ui(r.P, 1);
            mpz_set_ui(r.Q, 1);
            mpz_set_si(r.T, CHUD_A);
        } else {
            // P = -(6a-5)(2a-1)(6a-1)
            long p1 = 6L * a - 5;
            long p2 = 2L * a - 1;
            long p3 = 6L * a - 1;
            mpz_set_si(r.P, -p1);
            mpz_mul_si(r.P, r.P, p2);
            mpz_mul_si(r.P, r.P, p3);

            // Q = a^3 * C^3/24
            // C^3/24 = 10939058860032000
            mpz_set_ui(r.Q, 0);
            mpz_ui_pow_ui(r.Q, (unsigned long)a, 3);
            // Multiply by C^3/24 in two steps to stay within unsigned long:
            // 10939058860032000 = 10939058860 * 1000000 + 32000
            // Better: factor as 640320^3/24 = 26680 * 640320 * 640320
            mpz_mul_ui(r.Q, r.Q, 26680UL);
            mpz_mul_ui(r.Q, r.Q, 640320UL);
            mpz_mul_ui(r.Q, r.Q, 640320UL);

            // T = P * (A + B*a)
            long ak = CHUD_A + CHUD_B * (long)a;
            mpz_mul_si(r.T, r.P, ak);
        }
        return;
    }

    int mid = (a + b) / 2;
    BSResult left, right;
    bs_init(left);
    bs_init(right);

#ifdef HAS_OPENMP
    if (depth < OMP_TASK_DEPTH) {
        #pragma omp task shared(left) if(depth < OMP_TASK_DEPTH)
        chudnovsky_bs(left, a, mid, depth + 1);
        #pragma omp task shared(right) if(depth < OMP_TASK_DEPTH)
        chudnovsky_bs(right, mid, b, depth + 1);
        #pragma omp taskwait
    } else
#endif
    {
        chudnovsky_bs(left, a, mid, depth + 1);
        chudnovsky_bs(right, mid, b, depth + 1);
    }

    // Merge: P = Pl * Pr, Q = Ql * Qr, T = Tl * Qr + Pl * Tr
    mpz_mul(r.P, left.P, right.P);
    mpz_mul(r.Q, left.Q, right.Q);

    // T = Tl * Qr + Pl * Tr
    mpz_t tmp1, tmp2;
    mpz_init(tmp1);
    mpz_init(tmp2);
    mpz_mul(tmp1, left.T, right.Q);
    mpz_mul(tmp2, left.P, right.T);
    mpz_add(r.T, tmp1, tmp2);
    mpz_clear(tmp1);
    mpz_clear(tmp2);

    bs_clear(left);
    bs_clear(right);
}

// ------------------------------------------------------------------
//  cpu_compute_pi_digits_binsplit - Binary Splitting Chudnovsky on CPU
//
//  1. Binary splitting -> (P, Q, T) as GMP big integers
//  2. Final division + sqrt using GMP mpf_t
//  3. Extract base-10^9 limbs from result
// ------------------------------------------------------------------

void cpu_compute_pi_digits_binsplit(uint32_t* pi_digits, int n_limbs,
                                     long long n_digits,
                                     const std::vector<int>& cores) {
#ifdef HAS_OPENMP
    if (!cores.empty()) {
        omp_set_num_threads((int)cores.size());
    }
#endif
    (void)cores;

    int n_terms = (int)(n_digits / 14.0) + 10;

    // GMP precision: enough bits for n_digits decimal digits + guard digits
    mp_bitcnt_t prec_bits = (mp_bitcnt_t)((n_digits + 100) * 3.3219281 + 256);

    auto start_time = std::chrono::steady_clock::now();
    fprintf(stderr, "  %sBinary splitting: %d terms%s\n",
            rgb_str(clr_rgb::dim), n_terms, clr(CLR_RESET));

    // --- Phase 1: Binary splitting recursion ---
    BSResult bs;
    bs_init(bs);

#ifdef HAS_OPENMP
    #pragma omp parallel
    #pragma omp single
#endif
    {
        chudnovsky_bs(bs, 0, n_terms, 0);
    }

    auto bs_end = std::chrono::steady_clock::now();
    double bs_ms = std::chrono::duration<double, std::milli>(bs_end - start_time).count();
    ui_progress_done("Binary splitting (CPU)", bs_ms);

    // --- Phase 2: Compute pi = 426880 * sqrt(10005) * Q / T ---
    fprintf(stderr, "  %sComputing final value (sqrt + division)...%s\n",
            rgb_str(clr_rgb::dim), clr(CLR_RESET));
    auto final_start = std::chrono::steady_clock::now();

    mpf_t pi_val, sqrt_10005, fQ, fT, temp;
    mpf_init2(pi_val, prec_bits);
    mpf_init2(sqrt_10005, prec_bits);
    mpf_init2(fQ, prec_bits);
    mpf_init2(fT, prec_bits);
    mpf_init2(temp, prec_bits);

    // sqrt(10005)
    mpf_set_ui(temp, 10005);
    mpf_sqrt(sqrt_10005, temp);

    // Convert Q and T from mpz_t to mpf_t
    mpf_set_z(fQ, bs.Q);
    mpf_set_z(fT, bs.T);

    // pi = 426880 * sqrt(10005) * Q / T
    mpf_mul_ui(pi_val, sqrt_10005, 426880UL);
    mpf_mul(pi_val, pi_val, fQ);
    mpf_div(pi_val, pi_val, fT);

    auto final_end = std::chrono::steady_clock::now();
    double final_ms = std::chrono::duration<double, std::milli>(final_end - final_start).count();
    ui_spinner_done("Final computation", final_ms);

    // --- Phase 3: Extract base-10^9 limbs ---
    extract_pi_digits(pi_digits, n_limbs, pi_val, n_digits);

    // Cleanup
    mpf_clear(pi_val);
    mpf_clear(sqrt_10005);
    mpf_clear(fQ);
    mpf_clear(fT);
    mpf_clear(temp);
    bs_clear(bs);
}

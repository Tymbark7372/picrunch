/**
 * backends.h - PiCrunch shared types, UI utilities, and backend declarations.
 *
 * Included by pi.cu (CUDA + main), pi_opencl.cpp, and pi_cpu.cpp.
 */

#pragma once
#ifndef BACKENDS_H
#define BACKENDS_H

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX          // Prevent Windows min/max macros
#  endif
#  include <io.h>
#  include <intrin.h>
#  define isatty _isatty
#  define fileno _fileno
#else
#  include <unistd.h>
#endif

// ===================================================================
//  Cross-platform 128-bit trial-quotient helper for long division.
//  Computes:  (a * BN_BASE^2 + b * BN_BASE + c) / divisor
//  where a, b, c are < BN_BASE (uint32_t) and divisor is uint64_t.
// ===================================================================
inline uint64_t trial_quotient_128(uint32_t a, uint32_t b, uint32_t c,
                                   uint64_t divisor) {
    if (divisor == 0) return 999999999ULL;  // BN_BASE - 1
#ifdef _MSC_VER
    // Build 128-bit value (hi:lo) = a * BN_BASE^2 + b * BN_BASE + c
    // Step 1: a * BN_BASE (fits in 64-bit: < 10^18)
    uint64_t ab = (uint64_t)a * 1000000000ULL;
    // Step 2: ab * BN_BASE as 128-bit via _umul128
    uint64_t hi = 0, lo = 0;
    lo = _umul128(ab, 1000000000ULL, &hi);
    // Step 3: add b * BN_BASE + c (fits in 64-bit: < 10^18 + 10^9)
    uint64_t addend = (uint64_t)b * 1000000000ULL + c;
    lo += addend;
    if (lo < addend) hi++;
    // Step 4: 128-bit divide by 64-bit divisor
    return _udiv128(hi, lo, divisor, nullptr);
#else
    __int128 r3 = (__int128)a * 1000000000LL * 1000000000LL
                + (__int128)b * 1000000000LL + c;
    return (uint64_t)(r3 / (__int128)divisor);
#endif
}

// ===================================================================
//  Result and device types
// ===================================================================

struct ComputeResult {
    double pi_value;
    double elapsed_ms;
    long long intervals;
    std::string device_name;
    int num_threads;            // CPU mode: threads used
    double throughput_gips;     // billion intervals per second
};

struct DeviceInfo {
    int index;
    std::string name;
    std::string details;        // e.g. "16 GB, SM 8.9"
};

struct BenchmarkResult {
    std::string backend;
    std::string test_name;
    double elapsed_ms;
    double metric;
    std::string metric_unit;
    bool success;
};

// ===================================================================
//  ANSI color helpers
// ===================================================================

// Defined in pi.cu
extern bool g_use_color;

// Color codes
#define CLR_RESET   "\033[0m"
#define CLR_BOLD    "\033[1m"
#define CLR_DIM     "\033[2m"
#define CLR_RED     "\033[31m"
#define CLR_GREEN   "\033[32m"
#define CLR_YELLOW  "\033[33m"
#define CLR_BLUE    "\033[34m"
#define CLR_MAGENTA "\033[35m"
#define CLR_CYAN    "\033[36m"
#define CLR_WHITE   "\033[37m"
#define CLR_BWHITE  "\033[97m"    // bright white

// Return color code if colors enabled, else empty string
inline const char* clr(const char* code) {
    return g_use_color ? code : "";
}

// ===================================================================
//  RGB 24-bit color support for gradient effects
// ===================================================================

struct RGB { int r, g, b; };

// Emit \033[38;2;R;G;Bm foreground color (no-op when colors disabled)
inline void set_rgb_fg(FILE* f, int r, int g, int b) {
    if (g_use_color) fprintf(f, "\033[38;2;%d;%d;%dm", r, g, b);
}

// Linearly interpolate between two RGB colors (t in [0,1])
inline RGB rgb_lerp(RGB a, RGB b, double t) {
    return { a.r + (int)((b.r - a.r) * t),
             a.g + (int)((b.g - a.g) * t),
             a.b + (int)((b.b - a.b) * t) };
}

// Multi-stop gradient with evenly spaced stops (t in [0,1])
inline RGB rgb_gradient(const RGB* stops, int n, double t) {
    if (t <= 0.0 || n <= 1) return stops[0];
    if (t >= 1.0) return stops[n - 1];
    double seg = t * (n - 1);
    int i = (int)seg;
    if (i >= n - 1) i = n - 2;
    return rgb_lerp(stops[i], stops[i + 1], seg - i);
}

// ===================================================================
//  rgb_str() - return ANSI escape as string for use in printf
//  Uses rotating static buffers (8 slots) for safe multi-use in one printf.
// ===================================================================

inline const char* rgb_str(int r, int g, int b) {
    static char bufs[8][24];
    static int slot = 0;
    if (!g_use_color) return "";
    char* buf = bufs[slot++ & 7];
    snprintf(buf, 24, "\033[38;2;%d;%d;%dm", r, g, b);
    return buf;
}
inline const char* rgb_str(RGB c) { return rgb_str(c.r, c.g, c.b); }

// ===================================================================
//  Semantic RGB color palette
// ===================================================================

namespace clr_rgb {
    inline const RGB heading  = {224, 231, 255};  // bright lavender
    inline const RGB label    = {148, 163, 184};  // slate
    inline const RGB value    = {226, 232, 240};  // bright white-blue
    inline const RGB accent   = {96, 165, 250};   // blue
    inline const RGB success  = {34, 197, 94};    // green
    inline const RGB error    = {239, 68, 68};    // red
    inline const RGB warning  = {251, 191, 36};   // amber
    inline const RGB dim      = {100, 116, 139};  // muted slate
    inline const RGB pi_val   = {52, 211, 153};   // mint-green
    inline const RGB index    = {129, 140, 248};  // indigo-light
    inline const RGB dev_name = {186, 230, 253};  // sky-light
    inline const RGB mode_val = {34, 211, 238};   // cyan
}

// ===================================================================
//  Spinner and progress bar (write to stderr)
// ===================================================================

// Braille spinner frames
inline const char* spinner_frame(int tick) {
    static const char* frames[] = {
        "\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8",
        "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
        "\xe2\xa0\x87", "\xe2\xa0\x8f"
    };
    return frames[tick % 10];
}

// Show one spinner frame with color-cycling braille dot
inline void ui_spinner_tick(const char* msg, double elapsed_s, int tick) {
    const RGB sg[] = {{96,165,250}, {34,211,238}, {52,211,153}};
    double t = (sin((double)tick * 0.4) + 1.0) / 2.0;
    RGB sc = rgb_gradient(sg, 3, t);
    fprintf(stderr, "\r  ");
    set_rgb_fg(stderr, sc.r, sc.g, sc.b);
    fprintf(stderr, "%s%s %s%s%s %s[%.1fs]%s   ",
            spinner_frame(tick), clr(CLR_RESET),
            clr(CLR_WHITE), msg, clr(CLR_RESET),
            clr(CLR_DIM), elapsed_s, clr(CLR_RESET));
    fflush(stderr);
}

// Finish spinner:  ✓ msg - 12.3 ms  (bright green checkmark + timing)
inline void ui_spinner_done(const char* msg, double elapsed_ms) {
    fprintf(stderr, "\r  ");
    set_rgb_fg(stderr, 34, 197, 94);
    fprintf(stderr, "\xe2\x9c\x93%s %s%s%s - ",
            clr(CLR_RESET), clr(CLR_WHITE), msg, clr(CLR_RESET));
    set_rgb_fg(stderr, 34, 197, 94);
    fprintf(stderr, "%.3f ms%s                \n", elapsed_ms, clr(CLR_RESET));
}

// Show progress bar with smooth RGB gradient fill and sub-character resolution
inline void ui_progress_bar(const char* msg, double fraction, double elapsed_s) {
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
    const int width = 30;
    double fill_exact = fraction * width;
    int filled = (int)fill_exact;
    int partial = (int)((fill_exact - filled) * 8);

    fprintf(stderr, "\r  %s%s%s %s[", clr(CLR_WHITE), msg, clr(CLR_RESET), clr(CLR_RESET));

    // Gradient: indigo -> cyan -> green
    const RGB bg[] = {{99,102,241}, {6,182,212}, {34,197,94}};
    // Partial block chars: ▏▎▍▌▋▊▉ (1/8 through 7/8)
    static const char* part_ch[] = {
        "\xe2\x96\x8f", "\xe2\x96\x8e", "\xe2\x96\x8d",
        "\xe2\x96\x8c", "\xe2\x96\x8b", "\xe2\x96\x8a",
        "\xe2\x96\x89"
    };

    for (int i = 0; i < width; i++) {
        double t = (width > 1) ? (double)i / (width - 1) : 0.0;
        if (i < filled) {
            RGB c = rgb_gradient(bg, 3, t);
            set_rgb_fg(stderr, c.r, c.g, c.b);
            fprintf(stderr, "\xe2\x96\x88");           // █
        } else if (i == filled && partial > 0) {
            RGB c = rgb_gradient(bg, 3, t);
            set_rgb_fg(stderr, c.r, c.g, c.b);
            fprintf(stderr, "%s", part_ch[partial - 1]);
        } else {
            set_rgb_fg(stderr, 55, 55, 75);
            fprintf(stderr, "\xe2\x96\x91");           // ░
        }
    }

    fprintf(stderr, "%s] %s%3.0f%%%s %s[%.1fs]%s   ",
            clr(CLR_RESET),
            clr(CLR_BWHITE), fraction * 100.0, clr(CLR_RESET),
            clr(CLR_DIM), elapsed_s, clr(CLR_RESET));
    fflush(stderr);
}

// Finish progress bar
inline void ui_progress_done(const char* msg, double elapsed_ms) {
    fprintf(stderr, "\r  ");
    set_rgb_fg(stderr, 34, 197, 94);
    fprintf(stderr, "\xe2\x9c\x93%s %s%s%s - ",
            clr(CLR_RESET), clr(CLR_WHITE), msg, clr(CLR_RESET));
    set_rgb_fg(stderr, 34, 197, 94);
    fprintf(stderr, "%.3f ms%s                                        \n",
            elapsed_ms, clr(CLR_RESET));
}

// ===================================================================
//  Utility: format number with commas (e.g. 1000000 -> "1,000,000")
// ===================================================================

inline std::string format_number(long long n) {
    if (n < 0) return "-" + format_number(-n);
    std::string s = std::to_string(n);
    int insertPos = (int)s.length() - 3;
    while (insertPos > 0) {
        s.insert(insertPos, ",");
        insertPos -= 3;
    }
    return s;
}

// ===================================================================
//  Big-number constants
// ===================================================================

static const uint64_t BN_BASE = 1000000000ULL;  // base-10^9 limbs

// ===================================================================
//  Backend declarations
// ===================================================================

// ---- CUDA ----
int cuda_device_count();
std::vector<DeviceInfo> cuda_list_devices();
ComputeResult cuda_integrate(long long N, int device_id);
void cuda_compute_pi_digits(uint32_t* pi_digits, int n_limbs, long long n_digits);

// ---- OpenCL ----
int opencl_device_count();
std::vector<DeviceInfo> opencl_list_devices();
ComputeResult opencl_integrate(long long N, int device_id);
void opencl_compute_pi_digits(uint32_t* pi_digits, int n_limbs, long long n_digits, int device_id);

// ---- CPU ----
int cpu_core_count();
std::string cpu_model_name();
ComputeResult cpu_integrate(long long N, const std::vector<int>& cores);
void cpu_compute_pi_digits(uint32_t* pi_digits, int n_limbs, long long n_digits, const std::vector<int>& cores);
void cpu_compute_pi_digits_binsplit(uint32_t* pi_digits, int n_limbs, long long n_digits, const std::vector<int>& cores);

#endif // BACKENDS_H

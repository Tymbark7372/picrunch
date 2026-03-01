/**
 * pi.cu - PiCrunch: High-accuracy pi calculator with CUDA, OpenCL, and CPU backends.
 *
 * Three computation modes:
 *
 * 1. CUDA (default) - GPU integration with warp-shuffle reduction and
 *    Kahan compensated summation.  Full double-precision accuracy.
 *
 * 2. OpenCL - Portable GPU integration for non-CUDA GPUs.  Same midpoint
 *    rule algorithm with Kahan summation in an OpenCL kernel.
 *
 * 3. CPU - OpenMP multithreaded integration with optional core pinning.
 *
 * also supports --digits N to compute N exact decimal digits using the
 * Chudnovsky algorithm with arbitrary-precision base-10^9 arithmetic.
 *
 * Build:  cmake -B build && cmake --build build
 *         make                (Linux shortcut)
 * Run:    ./pi [N] [--mode cuda|opencl|cpu] [--device D] [--cores 0,1,2]
 *         ./pi --digits 1000000 --output pi.txt
 */

#include "backends.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>

// MSVC needs this before <cmath> for M_PI, but we define a fallback anyway
#ifdef _WIN32
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#  endif
#  include <math.h>
#  include <windows.h>
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// NVML for GPU utilization stats - optional, disabled with -DNO_NVML
#ifndef NO_NVML
#  include <nvml.h>
#  define HAS_NVML 1
#else
#  define HAS_NVML 0
#endif

// ===================================================================
//  Global: color support (referenced by backends.h inline functions)
// ===================================================================

bool g_use_color = false;

static void init_color() {
#ifdef _WIN32
    // Set console to UTF-8 so box-drawing and Unicode glyphs render correctly
    SetConsoleOutputCP(CP_UTF8);

    // Enable ANSI/VT escape sequences on Windows 10+ (stdout and stderr)
    for (DWORD handle_id : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
        HANDLE h = GetStdHandle(handle_id);
        DWORD mode = 0;
        if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode))
            SetConsoleMode(h, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */);
    }
    g_use_color = true;
#else
    // Enable color only on real terminals, respect NO_COLOR convention
    if (isatty(fileno(stdout)) && isatty(fileno(stderr))) {
        const char* nc = getenv("NO_COLOR");
        g_use_color = (nc == nullptr || nc[0] == '\0');
    }
#endif
}

// ===================================================================
//  Error-checking helper
// ===================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "%sCUDA error%s at %s:%d - %s\n",                   \
                    rgb_str(clr_rgb::error), clr(CLR_RESET),                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===================================================================
//  ASCII banner
// ===================================================================

// Print a gradient horizontal border:  ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓  (or ┗┛ variant)
static void print_hborder(const char* left, const char* right, int width,
                           const RGB* grad, int nstops) {
    printf("  ");
    for (int i = 0; i < width + 2; i++) {
        double t = (double)i / (width + 1);
        RGB c = rgb_gradient(grad, nstops, t);
        set_rgb_fg(stdout, c.r, c.g, c.b);
        if (i == 0)              printf("%s", left);
        else if (i == width + 1) printf("%s", right);
        else                     printf("\xe2\x94\x81");  // ━
    }
    printf("%s\n", clr(CLR_RESET));
}

// Print a gradient horizontal rule with optional centered title
static void print_gradient_rule(int width, const RGB* grad, int nstops,
                                 const char* title = nullptr) {
    printf("  ");
    if (title) {
        int tlen = (int)strlen(title);
        int left_n  = (width - tlen - 2) / 2;
        int right_n = width - tlen - 2 - left_n;
        for (int i = 0; i < left_n; i++) {
            RGB c = rgb_gradient(grad, nstops, (double)i / (width - 1));
            set_rgb_fg(stdout, c.r, c.g, c.b);
            printf("\xe2\x94\x81");
        }
        printf(" ");
        // Title with bright gradient
        const RGB tg[] = {{224,231,255}, {186,230,253}};
        int tidx = 0;
        for (const unsigned char* p = (const unsigned char*)title; *p; ) {
            int bytes = 1;
            if (*p >= 0xF0) bytes = 4;
            else if (*p >= 0xE0) bytes = 3;
            else if (*p >= 0xC0) bytes = 2;
            if (*p != ' ') {
                double tt = (tlen > 1) ? (double)tidx / (tlen - 1) : 0.0;
                RGB tc = rgb_gradient(tg, 2, tt);
                set_rgb_fg(stdout, tc.r, tc.g, tc.b);
            }
            for (int b = 0; b < bytes; b++) putchar(p[b]);
            p += bytes; tidx++;
        }
        printf(" ");
        for (int i = 0; i < right_n; i++) {
            RGB c = rgb_gradient(grad, nstops,
                      (double)(left_n + tlen + 2 + i) / (width - 1));
            set_rgb_fg(stdout, c.r, c.g, c.b);
            printf("\xe2\x94\x81");
        }
    } else {
        for (int i = 0; i < width; i++) {
            RGB c = rgb_gradient(grad, nstops, (double)i / (width - 1));
            set_rgb_fg(stdout, c.r, c.g, c.b);
            printf("\xe2\x94\x81");
        }
    }
    printf("%s\n", clr(CLR_RESET));
}

static void print_banner() {
    printf("\n");

    // Gradient color palettes
    const RGB gb[] = {{99,102,241}, {59,130,246}, {6,182,212}};   // border: indigo->blue->cyan
    const RGB ga[] = {{96,165,250}, {34,211,238}, {52,211,153}};  // art: blue->cyan->mint
    const RGB gs[] = {{251,191,36}, {251,146,60}};                // subtitle: amber->orange

    const int W = 50;  // inner visible width

    // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    print_hborder("\xe2\x94\x8f", "\xe2\x94\x93", W, gb, 3);

    // Figlet art lines - per-character gradient with diagonal sweep
    const char* art[] = {
        "   ____  _  ____                       _          ",
        "  |  _ \\(_)/ ___|_ __ _   _ _ __   ___| |__       ",
        "  | |_) | | |   | '__| | | | '_ \\ / __| '_ \\      ",
        "  |  __/| | |___| |  | |_| | | | | (__| | | |     ",
        "  |_|   |_|\\____|_|   \\__,_|_| |_|\\___|_| |_|     ",
    };
    for (int line = 0; line < 5; line++) {
        printf("  ");
        // Left border ┃
        set_rgb_fg(stdout, gb[0].r, gb[0].g, gb[0].b);
        printf("\xe2\x94\x83");

        // Each character with gradient + diagonal sweep
        int len = (int)strlen(art[line]);
        for (int i = 0; i < W; i++) {
            char ch = (i < len) ? art[line][i] : ' ';
            if (ch != ' ') {
                double t = (double)i / (W - 1);
                t = t * 0.85 + (double)line / 5.0 * 0.15;
                if (t > 1.0) t = 1.0;
                RGB c = rgb_gradient(ga, 3, t);
                set_rgb_fg(stdout, c.r, c.g, c.b);
            }
            putchar(ch);
        }

        // Right border ┃
        set_rgb_fg(stdout, gb[2].r, gb[2].g, gb[2].b);
        printf("\xe2\x94\x83%s\n", clr(CLR_RESET));
    }

    // Blank line
    printf("  ");
    set_rgb_fg(stdout, gb[0].r, gb[0].g, gb[0].b);
    printf("\xe2\x94\x83");
    printf("%-50s", "");
    set_rgb_fg(stdout, gb[2].r, gb[2].g, gb[2].b);
    printf("\xe2\x94\x83%s\n", clr(CLR_RESET));

    // Subtitle: "    CUDA · OpenCL · CPU  Pi Calculator  v1.0      "
    printf("  ");
    set_rgb_fg(stdout, gb[0].r, gb[0].g, gb[0].b);
    printf("\xe2\x94\x83");

    // Per-character gradient for subtitle text
    const char* sub = "      CUDA \xc2\xb7 OpenCL \xc2\xb7 CPU  PiCrunch  v1.0         ";
    int n_disp = 0;
    for (const unsigned char* p = (const unsigned char*)sub; *p; ) {
        int bytes = 1;
        if (*p >= 0xF0) bytes = 4; else if (*p >= 0xE0) bytes = 3;
        else if (*p >= 0xC0) bytes = 2;
        n_disp++; p += bytes;
    }
    int idx = 0;
    for (const unsigned char* p = (const unsigned char*)sub; *p; ) {
        int bytes = 1;
        if (*p >= 0xF0) bytes = 4; else if (*p >= 0xE0) bytes = 3;
        else if (*p >= 0xC0) bytes = 2;
        if (*p != ' ') {
            double t = (n_disp > 1) ? (double)idx / (n_disp - 1) : 0.0;
            RGB c = rgb_gradient(gs, 2, t);
            set_rgb_fg(stdout, c.r, c.g, c.b);
        }
        for (int b = 0; b < bytes; b++) putchar(p[b]);
        p += bytes; idx++;
    }

    set_rgb_fg(stdout, gb[2].r, gb[2].g, gb[2].b);
    printf("\xe2\x94\x83%s\n", clr(CLR_RESET));

    // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    print_hborder("\xe2\x94\x97", "\xe2\x94\x9b", W, gb, 3);

    printf("\n");
    fflush(stdout);
}

// ===================================================================
//  Device discovery and listing
// ===================================================================

static void print_section_header(const char* title) {
    const RGB hg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
    printf("  ");
    // Leading ━━ with gradient
    for (int i = 0; i < 2; i++) {
        RGB c = rgb_gradient(hg, 3, (double)i * 0.15);
        set_rgb_fg(stdout, c.r, c.g, c.b);
        printf("\xe2\x94\x81");
    }
    // Title
    printf(" %s%s%s ", rgb_str(clr_rgb::heading), title, clr(CLR_RESET));
    // Trailing ━━ with gradient
    for (int i = 0; i < 2; i++) {
        RGB c = rgb_gradient(hg, 3, 0.7 + (double)i * 0.15);
        set_rgb_fg(stdout, c.r, c.g, c.b);
        printf("\xe2\x94\x81");
    }
    printf("%s\n", clr(CLR_RESET));
}

static void list_all_devices() {
    {
        const RGB dg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
        print_gradient_rule(48, dg, 3, "Available Devices");
        printf("\n");
    }

    // CPU
    printf("  %sCPU%s\n", rgb_str(clr_rgb::heading), clr(CLR_RESET));
    printf("    %s\xe2\x80\xa2%s %s%s%s - %s%d cores%s\n\n",
           rgb_str(clr_rgb::accent), clr(CLR_RESET),
           rgb_str(clr_rgb::dev_name), cpu_model_name().c_str(), clr(CLR_RESET),
           rgb_str(clr_rgb::dim), cpu_core_count(), clr(CLR_RESET));

    // CUDA GPUs
    int cuda_count = cuda_device_count();
    printf("  %sCUDA GPUs%s", rgb_str(clr_rgb::heading), clr(CLR_RESET));
    if (cuda_count == 0) {
        printf("  %s(none detected)%s\n\n", rgb_str(clr_rgb::dim), clr(CLR_RESET));
    } else {
        printf("\n");
        auto devs = cuda_list_devices();
        for (auto& d : devs) {
            printf("    %s[%d]%s %s%s%s - %s%s%s\n",
                   rgb_str(clr_rgb::index), d.index, clr(CLR_RESET),
                   rgb_str(clr_rgb::dev_name), d.name.c_str(), clr(CLR_RESET),
                   rgb_str(clr_rgb::dim), d.details.c_str(), clr(CLR_RESET));
        }
        printf("\n");
    }

    // OpenCL devices
    int ocl_count = opencl_device_count();
    printf("  %sOpenCL Devices%s", rgb_str(clr_rgb::heading), clr(CLR_RESET));
    if (ocl_count == 0) {
        printf("  %s(none detected)%s\n", rgb_str(clr_rgb::dim), clr(CLR_RESET));
    } else {
        printf("\n");
        auto devs = opencl_list_devices();
        for (auto& d : devs) {
            printf("    %s[%d]%s %s%s%s - %s%s%s\n",
                   rgb_str(clr_rgb::index), d.index, clr(CLR_RESET),
                   rgb_str(clr_rgb::dev_name), d.name.c_str(), clr(CLR_RESET),
                   rgb_str(clr_rgb::dim), d.details.c_str(), clr(CLR_RESET));
        }
    }

    printf("\n");
    {
        const RGB dg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
        print_gradient_rule(48, dg, 3);
        printf("\n");
    }
}

// ===================================================================
//  Summary table  (left col = 19 chars, right col = 30 chars)
// ===================================================================

// UTF-8 box-drawing characters
#define BOX_H    "\xe2\x94\x80"  // ─  horizontal
#define BOX_V    "\xe2\x94\x82"  // │  vertical
#define BOX_TL   "\xe2\x94\x8c"  // ┌  top-left
#define BOX_TR   "\xe2\x94\x90"  // ┐  top-right
#define BOX_BL   "\xe2\x94\x94"  // └  bottom-left
#define BOX_BR   "\xe2\x94\x98"  // ┘  bottom-right
#define BOX_TM   "\xe2\x94\xac"  // ┬  top-middle
#define BOX_BM   "\xe2\x94\xb4"  // ┴  bottom-middle
#define BOX_LM   "\xe2\x94\x9c"  // ├  left-middle
#define BOX_RM   "\xe2\x94\xa4"  // ┤  right-middle
#define BOX_CROSS "\xe2\x94\xbc" // ┼  cross

static const int TBL_L = 19;   // left column visible width
static const int TBL_R = 34;   // right column visible width

// Emit a horizontal rule with RGB gradient: left + L×─ + mid + R×─ + right
static void table_hrule(const char* left, const char* mid, const char* right) {
    const RGB tg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
    const int N = TBL_L + TBL_R + 3;
    int p = 0;
    auto tc = [&]() {
        RGB c = rgb_gradient(tg, 3, (double)p++ / (N - 1));
        set_rgb_fg(stdout, c.r, c.g, c.b);
    };
    printf("  ");
    tc(); printf("%s", left);
    for (int i = 0; i < TBL_L; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s", mid);
    for (int i = 0; i < TBL_R; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s%s\n", right, clr(CLR_RESET));
}

static void table_top()       { table_hrule(BOX_TL, BOX_TM, BOX_TR); }
static void table_separator() { table_hrule(BOX_LM, BOX_CROSS, BOX_RM); }
static void table_bottom()    { table_hrule(BOX_BL, BOX_BM, BOX_BR); }

// Print a data row: │ label             │ value                          │
// Vertical borders use matched RGB gradient colors
static void table_row(const char* label, const char* value,
                      RGB val_rgb = clr_rgb::value) {
    const RGB tg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
    // label may contain multi-byte π (\xcf\x80 = 2 bytes, 1 display char).
    int label_extra = 0;
    for (const unsigned char* p = (const unsigned char*)label; *p; p++)
        if ((*p & 0xC0) == 0x80) label_extra++;
    printf("  ");
    set_rgb_fg(stdout, tg[0].r, tg[0].g, tg[0].b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(clr_rgb::label), TBL_L - 2 + label_extra, label, clr(CLR_RESET));
    RGB mc = rgb_gradient(tg, 3, 0.35);
    set_rgb_fg(stdout, mc.r, mc.g, mc.b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(val_rgb), TBL_R - 2, value, clr(CLR_RESET));
    set_rgb_fg(stdout, tg[2].r, tg[2].g, tg[2].b);
    printf(BOX_V "%s\n", clr(CLR_RESET));
}

static void print_results_table(const ComputeResult& res, const char* mode_str) {
    printf("\n");
    print_section_header("Computation Results");
    printf("\n");
    table_top();

    // Mode and device
    table_row("Mode", mode_str);
    {
        // Truncate device name to fit column with "..." if needed
        int max_len = TBL_R - 2;
        char dev_trunc[64];
        if ((int)res.device_name.length() > max_len) {
            snprintf(dev_trunc, sizeof(dev_trunc), "%.*s...",
                     max_len - 3, res.device_name.c_str());
        } else {
            snprintf(dev_trunc, sizeof(dev_trunc), "%s", res.device_name.c_str());
        }
        table_row("Device", dev_trunc);
    }
    table_row("Intervals", format_number(res.intervals).c_str());

    table_separator();

    // Pi values
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", res.pi_value);
        table_row("Computed \xcf\x80", buf, clr_rgb::pi_val);
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", M_PI);
        table_row("Reference \xcf\x80", buf, clr_rgb::dim);
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4e", fabs(res.pi_value - M_PI));
        table_row("Absolute error", buf, clr_rgb::warning);
    }

    table_separator();

    // Timing
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.3f ms", res.elapsed_ms);
        table_row("Compute time", buf);
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f Gint/s", res.throughput_gips);
        table_row("Throughput", buf);
    }

    table_bottom();
    printf("\n");
}

// ===================================================================
//  PART 1 - CUDA Kernels
// ===================================================================

// Warp-level sum reduction via shuffle intrinsics
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level sum reduction: warp shuffle + shared memory
__device__ double blockReduceSum(double val) {
    __shared__ double shared[32]; // one slot per warp (max 1024 threads)
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0;
    if (warpId == 0) val = warpReduceSum(val);
    return val;
}

// Main integration kernel - each thread evaluates a strided set of midpoints
// with Kahan summation, then participates in a block reduction.
__global__ void integrateKernel(long long N, double *d_blockSums) {
    double sum  = 0.0;
    double comp = 0.0;
    double step = 1.0 / (double)N;

    long long globalIdx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride    = (long long)blockDim.x * gridDim.x;

    for (long long i = globalIdx; i < N; i += stride) {
        double x = ((double)i + 0.5) * step;
        double f = 4.0 / (1.0 + x * x);
        // Kahan compensated addition
        double y = f - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }
    sum *= step;

    double blockTotal = blockReduceSum(sum);
    if (threadIdx.x == 0)
        d_blockSums[blockIdx.x] = blockTotal;
}

// Final reduction - single block sums the per-block partial results.
__global__ void finalReduceKernel(double *d_blockSums, int numBlocks,
                                   double *d_result) {
    double sum  = 0.0;
    double comp = 0.0;
    for (int i = threadIdx.x; i < numBlocks; i += blockDim.x) {
        double y = d_blockSums[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }
    double blockTotal = blockReduceSum(sum);
    if (threadIdx.x == 0)
        d_result[0] = blockTotal;
}

// ===================================================================
//  CUDA backend: device listing and integration
// ===================================================================

int cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

std::vector<DeviceInfo> cuda_list_devices() {
    std::vector<DeviceInfo> out;
    int count = cuda_device_count();
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        DeviceInfo info;
        info.index = i;
        info.name  = prop.name;
        char detail[128];
        snprintf(detail, sizeof(detail), "%.0f MB, SM %d.%d, %d SMs",
                 (double)prop.totalGlobalMem / 1048576.0,
                 prop.major, prop.minor, prop.multiProcessorCount);
        info.details = detail;
        out.push_back(info);
    }
    return out;
}

ComputeResult cuda_integrate(long long N, int device_id) {
    int count = cuda_device_count();
    if (count == 0) {
        fprintf(stderr, "%sError:%s No CUDA-capable GPU detected.\n",
                rgb_str(clr_rgb::error), clr(CLR_RESET));
        exit(EXIT_FAILURE);
    }
    if (device_id < 0 || device_id >= count) {
        fprintf(stderr, "%sError:%s CUDA device index %d out of range (0..%d).\n",
                rgb_str(clr_rgb::error), clr(CLR_RESET), device_id, count - 1);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    // Launch config
    int threadsPerBlock = 256;
    int numBlocks = prop.multiProcessorCount * 8;
    if (numBlocks > 65535) numBlocks = 65535;

    // Allocate device memory
    double *d_blockSums = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

    // Timing events
    cudaEvent_t tStart, tStop;
    CUDA_CHECK(cudaEventCreate(&tStart));
    CUDA_CHECK(cudaEventCreate(&tStop));

    // Launch kernels
    CUDA_CHECK(cudaEventRecord(tStart));
    integrateKernel<<<numBlocks, threadsPerBlock>>>(N, d_blockSums);
    int finalThreads = ((((numBlocks < 256) ? numBlocks : 256) + 31) / 32) * 32;
    finalReduceKernel<<<1, finalThreads>>>(d_blockSums, numBlocks, d_result);
    CUDA_CHECK(cudaEventRecord(tStop));

    // Spinner while GPU works
    fprintf(stderr, "\n");
    auto wall_start = std::chrono::steady_clock::now();
    int tick = 0;
    while (cudaEventQuery(tStop) == cudaErrorNotReady) {
        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(now - wall_start).count();
        ui_spinner_tick("Computing \xcf\x80 (CUDA)", secs, tick++);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, tStart, tStop));
    ui_spinner_done("Computing \xcf\x80 (CUDA)", elapsed_ms);

    // Copy result back
    double pi_gpu = 0.0;
    CUDA_CHECK(cudaMemcpy(&pi_gpu, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(tStart));
    CUDA_CHECK(cudaEventDestroy(tStop));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_result));

    ComputeResult res;
    res.pi_value       = pi_gpu;
    res.elapsed_ms     = (double)elapsed_ms;
    res.intervals      = N;
    res.device_name    = prop.name;
    res.num_threads    = numBlocks * threadsPerBlock;
    res.throughput_gips = (double)N / ((double)elapsed_ms * 1e6);
    return res;
}

// ===================================================================
//  NVML GPU stats
// ===================================================================

#if HAS_NVML
static void print_nvml_stats(int device_id) {
    nvmlDevice_t nvmlDev;
    if (nvmlInit() != NVML_SUCCESS) return;
    if (nvmlDeviceGetHandleByIndex(device_id, &nvmlDev) != NVML_SUCCESS) {
        nvmlShutdown();
        return;
    }

    nvmlUtilization_t util;
    nvmlMemory_t mem;
    bool have_util = (nvmlDeviceGetUtilizationRates(nvmlDev, &util) == NVML_SUCCESS);
    bool have_mem  = (nvmlDeviceGetMemoryInfo(nvmlDev, &mem) == NVML_SUCCESS);

    if (have_util || have_mem) {
        print_section_header("GPU Stats (NVML)");
        printf("\n");
        table_top();
        if (have_util) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%u %%", util.gpu);
            table_row("GPU utilization", buf, clr_rgb::success);
            snprintf(buf, sizeof(buf), "%u %%", util.memory);
            table_row("Mem utilization", buf);
        }
        if (have_mem) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.1f / %.1f MB",
                     (double)mem.used / 1048576.0, (double)mem.total / 1048576.0);
            table_row("VRAM used/total", buf);
        }
        table_bottom();
        printf("\n");
    }

    nvmlShutdown();
}
#endif

// ===================================================================
//  PART 2 - CUDA Arbitrary-Precision Digit Computation (Chudnovsky)
//
//  Uses the Chudnovsky algorithm:
//    1/π = 12 Σ [(-1)^k (6k)! (13591409 + 545140134k)]
//                / [(3k)! (k!)^3 640320^(3k+3/2)]
//
//  Iterative recurrence on device-resident big-number limb arrays.
//  ~14.18 digits per term.
//
//  Representation: uint32_t array of n_limbs elements.
//    limb[0] = integer part, limb[1..n-1] = fractional part.
//    Each fractional limb holds 9 decimal digits (0-999999999).
// ===================================================================

// ------------------------------------------------------------------
//  CUDA big-number __device__ functions (single-thread, sequential carry chain)
//  Called from inside fused kernels - no launch overhead.
// ------------------------------------------------------------------

// Add: a[] += b[]
__device__ void dev_bn_add(uint32_t* a, const uint32_t* b, int n) {
    uint64_t carry = 0;
    for (int i = n - 1; i >= 0; i--) {
        uint64_t val = (uint64_t)a[i] + b[i] + carry;
        a[i] = (uint32_t)(val % BN_BASE);
        carry = val / BN_BASE;
    }
}

// Subtract: a[] -= b[]
__device__ void dev_bn_sub(uint32_t* a, const uint32_t* b, int n) {
    int64_t borrow = 0;
    for (int i = n - 1; i >= 0; i--) {
        int64_t val = (int64_t)a[i] - (int64_t)b[i] - borrow;
        if (val < 0) {
            val += (int64_t)BN_BASE;
            borrow = 1;
        } else {
            borrow = 0;
        }
        a[i] = (uint32_t)val;
    }
}

// Multiply by small scalar: a[] *= m (sequential - carry chain)
__device__ void dev_bn_mul_small(uint32_t* a, int n, uint64_t m) {
    uint64_t carry = 0;
    for (int i = n - 1; i >= 0; i--) {
        uint64_t val = (uint64_t)a[i] * m + carry;
        a[i] = (uint32_t)(val % BN_BASE);
        carry = val / BN_BASE;
    }
}

// Divide by small scalar: a[] /= d (sequential - remainder chain)
__device__ void dev_bn_divide(uint32_t* a, int n, uint64_t d) {
    uint64_t carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t val = carry * BN_BASE + a[i];
        a[i] = (uint32_t)(val / d);
        carry = val % d;
    }
}

// Device-to-device copy (single thread)
__device__ void dev_bn_memcpy(uint32_t* dst, const uint32_t* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// Zero-fill
__device__ void dev_bn_zero(uint32_t* a, int n) {
    for (int i = 0; i < n; i++) a[i] = 0;
}

// Check if big number is zero
__device__ bool dev_bn_is_zero(const uint32_t* a, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

// Schoolbook multiply: out[] = a[] * b[], truncated to n limbs
// Internally uses 2*n limb workspace (passed as out, which must be >= 2*n limbs)
// After computation, first n limbs of out hold the result.
__device__ void dev_bn_multiply(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, int n, uint32_t* temp2n) {
    for (int i = 0; i < 2 * n; i++) temp2n[i] = 0;

    for (int i = 0; i < n; i++) {
        if (a[i] == 0) continue;
        uint64_t carry = 0;
        for (int j = n - 1; j >= 0; j--) {
            int pos = i + j;
            uint64_t prod = (uint64_t)a[i] * b[j] + (uint64_t)temp2n[pos] + carry;
            temp2n[pos] = (uint32_t)(prod % BN_BASE);
            carry = prod / BN_BASE;
        }
        if (carry > 0 && i > 0) {
            temp2n[i - 1] += (uint32_t)carry;
        }
    }

    // Final carry propagation
    uint64_t carry = 0;
    for (int i = 2 * n - 1; i >= 0; i--) {
        uint64_t val = (uint64_t)temp2n[i] + carry;
        temp2n[i] = (uint32_t)(val % BN_BASE);
        carry = val / BN_BASE;
    }

    // Copy first n limbs (integer part + most significant fraction) to output
    for (int i = 0; i < n; i++) out[i] = temp2n[i];
}

// ------------------------------------------------------------------
//  CUDA kernel: Chudnovsky series - all terms in one GPU thread.
//  Eliminates per-term kernel launches (~86k -> 1).
//
//  Outputs: denominator[] = Chudnovsky sum (sum_pos - sum_neg)
// ------------------------------------------------------------------

__global__ void chudnovsky_series_kernel(
    uint32_t* denominator,   // output: Chudnovsky sum
    uint32_t* term,          // scratch: current series term
    uint32_t* temp,          // scratch
    uint32_t* temp2,         // scratch
    uint32_t* sum_pos,       // scratch: positive sum accumulator
    uint32_t* sum_neg,       // scratch: negative sum accumulator
    int n_limbs,
    int n_terms)
{
    // ---- Initialize term(0) = 1 ----
    dev_bn_zero(term, n_limbs);
    term[0] = 1;

    // ---- Initialize sum accumulators ----
    // k=0 contribution is positive: 13591409
    dev_bn_zero(sum_pos, n_limbs);
    sum_pos[0] = 13591409u;
    dev_bn_zero(sum_neg, n_limbs);

    // ---- Chudnovsky series loop k=1..n_terms ----
    for (int k = 1; k <= n_terms; k++) {
        uint64_t kk = (uint64_t)k;

        // Numerator: multiply by (6k-5)*(2k-1) then by (6k-1)
        // (6k-5)*(2k-1) fits in uint64_t for k up to millions
        uint64_t combined_num = (uint64_t)(6*k - 5) * (uint64_t)(2*k - 1);
        dev_bn_mul_small(term, n_limbs, combined_num);
        dev_bn_mul_small(term, n_limbs, (uint64_t)(6*k - 1));

        // Denominator: divide by k^3 * 640320^3 / 24
        // The carry in dev_bn_divide must satisfy: divisor * BN_BASE < 2^64,
        // i.e. divisor < ~1.8e10.  We combine /k/k -> /(k*k) and /k/26680 -> /(k*26680)
        // when k < 135000 (safe up to ~1.9M digits).  640320^2 ≈ 4.1e11 always overflows,
        // so those two divides always stay separate.
        if (kk < 135000ULL) {
            dev_bn_divide(term, n_limbs, kk * kk);
            dev_bn_divide(term, n_limbs, kk * 26680ULL);
        } else {
            dev_bn_divide(term, n_limbs, kk);
            dev_bn_divide(term, n_limbs, kk);
            dev_bn_divide(term, n_limbs, kk);
            dev_bn_divide(term, n_limbs, 26680ULL);
        }
        dev_bn_divide(term, n_limbs, 640320ULL);
        dev_bn_divide(term, n_limbs, 640320ULL);

        // Early termination: if term is all zeros, remaining terms won't contribute
        if (dev_bn_is_zero(term, n_limbs)) break;

        // Compute contribution: term * (13591409 + 545140134*k)
        uint64_t ak = 13591409ULL + 545140134ULL * kk;

        if (ak <= 0xFFFFFFFFULL) {
            // ak fits in 32 bits - single multiply
            dev_bn_memcpy(temp, term, n_limbs);
            dev_bn_mul_small(temp, n_limbs, ak);
        } else {
            // Split: contribution = term*545140134*k + term*13591409
            dev_bn_memcpy(temp, term, n_limbs);
            dev_bn_mul_small(temp, n_limbs, 545140134ULL);
            dev_bn_mul_small(temp, n_limbs, kk);
            dev_bn_memcpy(temp2, term, n_limbs);
            dev_bn_mul_small(temp2, n_limbs, 13591409ULL);
            dev_bn_add(temp, temp2, n_limbs);
        }

        // Accumulate into positive or negative sum
        if (k % 2 == 0) {
            dev_bn_add(sum_pos, temp, n_limbs);
        } else {
            dev_bn_add(sum_neg, temp, n_limbs);
        }
    }

    // ---- denominator = sum_pos - sum_neg ----
    dev_bn_memcpy(denominator, sum_pos, n_limbs);
    dev_bn_sub(denominator, sum_neg, n_limbs);
}

// ------------------------------------------------------------------
//  CUDA kernel: reciprocal sqrt via Newton iteration with progressive precision.
//  Each Newton step doubles precision, so early iterations use fewer limbs.
//  This reduces O(n^2) multiply cost dramatically: ~4x instead of full-precision
//  at every iteration.
//
//  Computes: numerator = 426880 * sqrt(10005)
// ------------------------------------------------------------------

__global__ void chudnovsky_sqrt_kernel(
    uint32_t* numerator,     // output: 426880 * sqrt(10005)
    uint32_t* temp,          // scratch (holds y_n)
    uint32_t* temp2,         // scratch
    uint32_t* temp2n,        // scratch: 2*n_limbs for multiply
    uint32_t* scratch_a,     // scratch
    uint32_t* scratch_b,     // scratch
    int n_limbs,
    int rsqrt_iters)
{
    const int val = 10005;

    // Initial guess: y0 = 1/sqrt(10005) ≈ 0.009997501...
    // Encode double-precision initial guess into big-number format
    dev_bn_zero(temp, n_limbs);
    {
        double y0 = 1.0 / sqrt((double)val);
        temp[0] = (uint32_t)y0;  // integer part (0 for val >= 2)
        double frac = y0 - (double)temp[0];
        for (int i = 1; i < n_limbs && i < 4; i++) {
            frac *= 1e9;
            temp[i] = (uint32_t)frac;
            frac -= (double)temp[i];
        }
    }

    // Newton iterations for reciprocal sqrt with progressive precision:
    //   y_{n+1} = y_n + y_n * (1 - val * y_n^2) / 2
    //
    // Starting precision: ~53 bits = ~6 limbs (from double initial guess).
    // Each iteration doubles precision. We use min(needed, n_limbs) limbs.
    // This turns the O(iters * n^2) cost into O(n^2) total (geometric sum).
    int work_limbs = 6;  // start with ~53 bits of precision

    for (int iter = 0; iter < rsqrt_iters; iter++) {
        // Double working precision for this iteration (capped at full)
        work_limbs = work_limbs * 2 + 2;  // +2 guard limbs
        if (work_limbs > n_limbs) work_limbs = n_limbs;

        // Step 1: numerator = y_n * y_n (using work_limbs precision)
        dev_bn_multiply(numerator, temp, temp, work_limbs, temp2n);
        // Zero out remaining limbs to avoid stale data
        for (int i = work_limbs; i < n_limbs; i++) numerator[i] = 0;

        // Step 2: numerator = val * y_n^2
        dev_bn_mul_small(numerator, work_limbs, (uint64_t)val);

        // Step 3: scratch_a = 1.0 (as big number)
        dev_bn_zero(scratch_a, work_limbs);
        scratch_a[0] = 1;

        // Step 4: compute correction = y_n * |1 - val*y_n^2| / 2
        if (numerator[0] >= 1) {
            // val*y_n^2 >= 1 -> subtract correction
            dev_bn_sub(numerator, scratch_a, work_limbs);
            dev_bn_multiply(scratch_b, temp, numerator, work_limbs, temp2n);
            for (int i = work_limbs; i < n_limbs; i++) scratch_b[i] = 0;
            dev_bn_divide(scratch_b, work_limbs, 2);
            dev_bn_sub(temp, scratch_b, work_limbs);
        } else {
            // val*y_n^2 < 1 -> add correction
            dev_bn_sub(scratch_a, numerator, work_limbs);
            dev_bn_multiply(numerator, temp, scratch_a, work_limbs, temp2n);
            for (int i = work_limbs; i < n_limbs; i++) numerator[i] = 0;
            dev_bn_divide(numerator, work_limbs, 2);
            dev_bn_add(temp, numerator, work_limbs);
        }
    }

    // Final: numerator = val * y_final = sqrt(val), then * 426880
    dev_bn_memcpy(numerator, temp, n_limbs);
    dev_bn_mul_small(numerator, n_limbs, (uint64_t)val);
    dev_bn_mul_small(numerator, n_limbs, 426880ULL);
}

// ------------------------------------------------------------------
//  cuda_compute_pi_digits - Chudnovsky via fused single-kernel launch
//
//  Computes: π = 426880 * sqrt(10005) / Σ_chudnovsky
//  All big-number work runs in one GPU thread (no kernel launch overhead).
//  Only the final long division runs on the host (one-time cost).
// ------------------------------------------------------------------

void cuda_compute_pi_digits(uint32_t* pi_digits, int n_limbs, long long n_digits) {
    // Number of Chudnovsky terms needed (~14.18 digits per term)
    int n_terms = (int)(n_digits / 14.0) + 10;

    // Compute reciprocal-sqrt iteration count:
    // Newton iteration doubles precision each step, starting from ~53 bits (double).
    // We need ceil(log2(target_bits / 53)) + 2 guard iterations.
    double target_bits = (double)n_limbs * 9.0 * 3.321928;
    int rsqrt_iters = (int)ceil(log2(target_bits / 53.0)) + 2;
    if (rsqrt_iters < 5) rsqrt_iters = 5;

    auto start_time = std::chrono::steady_clock::now();

    // Allocate device arrays
    uint32_t *d_numerator, *d_denominator, *d_term, *d_temp, *d_temp2, *d_temp2n;
    uint32_t *d_sum_pos, *d_sum_neg;
    size_t limb_bytes = n_limbs * sizeof(uint32_t);
    cudaMalloc(&d_numerator, limb_bytes);
    cudaMalloc(&d_denominator, limb_bytes);
    cudaMalloc(&d_term, limb_bytes);
    cudaMalloc(&d_temp, limb_bytes);
    cudaMalloc(&d_temp2, limb_bytes);
    cudaMalloc(&d_temp2n, 2 * limb_bytes);
    cudaMalloc(&d_sum_pos, limb_bytes);
    cudaMalloc(&d_sum_neg, limb_bytes);

    fprintf(stderr, "  %sSeries: %d terms, sqrt: %d iterations (fused kernels)%s\n",
            rgb_str(clr_rgb::dim), n_terms, rsqrt_iters, clr(CLR_RESET));

    // Launch series kernel - entire Chudnovsky loop in one GPU thread
    chudnovsky_series_kernel<<<1, 1>>>(
        d_denominator,
        d_term, d_temp, d_temp2,
        d_sum_pos, d_sum_neg,
        n_limbs, n_terms);

    cudaDeviceSynchronize();
    auto series_end = std::chrono::steady_clock::now();
    double series_ms = std::chrono::duration<double, std::milli>(series_end - start_time).count();
    ui_progress_done("Chudnovsky series (CUDA)", series_ms);

    // Launch sqrt kernel - reciprocal sqrt Newton iteration in one GPU thread
    fprintf(stderr, "  %sComputing sqrt(10005) on GPU...%s\n",
            rgb_str(clr_rgb::dim), clr(CLR_RESET));
    auto sqrt_start = std::chrono::steady_clock::now();
    chudnovsky_sqrt_kernel<<<1, 1>>>(
        d_numerator,
        d_temp, d_temp2, d_temp2n,
        d_sum_pos, d_sum_neg,
        n_limbs, rsqrt_iters);

    cudaDeviceSynchronize();
    auto sqrt_end = std::chrono::steady_clock::now();
    double sqrt_ms = std::chrono::duration<double, std::milli>(sqrt_end - sqrt_start).count();
    ui_spinner_done("sqrt(10005) (CUDA)", sqrt_ms);

    // Copy results back to host for final long division
    uint32_t* h_numerator = new uint32_t[n_limbs];
    uint32_t* h_denominator = new uint32_t[n_limbs];
    cudaMemcpy(h_numerator, d_numerator, limb_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_denominator, d_denominator, limb_bytes, cudaMemcpyDeviceToHost);

    // Long division: pi = numerator / denominator (one-time, runs on host)
    memset(pi_digits, 0, n_limbs * sizeof(uint32_t));

    // Multi-precision long division: quotient = numerator / denominator
    // Uses (n_limbs+1)-limb remainder with leading zero, Knuth-style estimate.
    {
        uint32_t* rem = new uint32_t[n_limbs + 1]();
        memcpy(rem + 1, h_numerator, n_limbs * sizeof(uint32_t));

        uint64_t d2 = (uint64_t)h_denominator[0] * BN_BASE + h_denominator[1];

        for (int i = 0; i < n_limbs; i++) {
            uint64_t q = trial_quotient_128(rem[0], rem[1], rem[2], d2);
            if (q >= BN_BASE) q = BN_BASE - 1;

            int64_t borrow = 0;
            for (int j = n_limbs - 1; j >= 0; j--) {
                int64_t prod = (int64_t)(q * (uint64_t)h_denominator[j]) + borrow;
                int64_t lo = prod % (int64_t)BN_BASE;
                borrow = prod / (int64_t)BN_BASE;

                int64_t val = (int64_t)rem[j + 1] - lo;
                if (val < 0) {
                    val += BN_BASE;
                    borrow++;
                }
                rem[j + 1] = (uint32_t)val;
            }
            rem[0] -= (uint32_t)borrow;

            while ((int32_t)rem[0] < 0) {
                q--;
                uint64_t carry = 0;
                for (int j = n_limbs - 1; j >= 0; j--) {
                    uint64_t val = (uint64_t)rem[j + 1] + h_denominator[j] + carry;
                    rem[j + 1] = (uint32_t)(val % BN_BASE);
                    carry = val / BN_BASE;
                }
                rem[0] += (uint32_t)carry;
            }

            pi_digits[i] = (uint32_t)q;

            for (int j = 0; j < n_limbs; j++)
                rem[j] = rem[j + 1];
            rem[n_limbs] = 0;
        }

        delete[] rem;
    }

    delete[] h_numerator;
    delete[] h_denominator;

    // Cleanup device memory
    cudaFree(d_numerator);
    cudaFree(d_denominator);
    cudaFree(d_term);
    cudaFree(d_temp);
    cudaFree(d_temp2);
    cudaFree(d_temp2n);
    cudaFree(d_sum_pos);
    cudaFree(d_sum_neg);
}

// ===================================================================
//  Argument parsing
// ===================================================================

enum ComputeMode {
    MODE_CUDA,
    MODE_OPENCL,
    MODE_CPU
};

enum AlgoType {
    ALGO_CHUDNOVSKY,
    ALGO_BINSPLIT
};

struct Options {
    long long intervals;
    long long digits;
    char output[512];
    long long max_size_mb;
    ComputeMode mode;
    AlgoType algo;
    int device_id;
    std::vector<int> cores;
    bool show_help;
    bool print_digits;
    bool benchmark;
    bool benchmark_quick;
    bool benchmark_gpu;
    bool benchmark_cpu;
};

static std::vector<int> parse_cores(const char* str) {
    std::vector<int> result;
    char* copy = strdup(str);
    char* token = strtok(copy, ",");
    while (token) {
        result.push_back(atoi(token));
        token = strtok(nullptr, ",");
    }
    free(copy);
    return result;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "%sUsage:%s %s [N] [OPTIONS]\n\n"
        "%sPositional:%s\n"
        "  N                     Number of intervals (default: 1e9)\n\n"
        "%sMode:%s\n"
        "  --mode MODE           Compute backend: %scuda%s (default), %sopencl%s, %scpu%s\n"
        "  --device N            GPU device index for cuda/opencl (default: 0)\n"
        "  --cores LIST          CPU cores for cpu mode, e.g. 0,1,2,3\n\n"
        "%sComputation:%s\n"
        "  --digits D            Compute D exact decimal digits of pi\n"
        "  --algo ALGO           Algorithm: chudnovsky (default), binsplit (cpu only)\n"
        "  --output FILE         Output file for digits (default: pi.txt)\n"
        "  --max-size MB         Max output file size in MB (default: 10240)\n"
        "  --print               Print digits to terminal\n"
        "  --no-print            Do not print digits to terminal (default)\n\n"
        "%sBenchmark:%s\n"
        "  --benchmark           Full benchmark (GPU throughput + CPU digits)\n"
        "  --benchmark-quick     Quick benchmark (GPU + CPU 10k/100k only)\n"
        "  --benchmark-gpu       GPU integration throughput only (fast)\n"
        "  --benchmark-cpu       CPU digit computation only (10k-10M)\n\n"
        "%sGeneral:%s\n"
        "  --help                Show this help message\n",
        rgb_str(clr_rgb::heading), clr(CLR_RESET), prog,
        rgb_str(clr_rgb::heading), clr(CLR_RESET),
        rgb_str(clr_rgb::heading), clr(CLR_RESET),
        rgb_str(clr_rgb::mode_val), clr(CLR_RESET),
        rgb_str(clr_rgb::mode_val), clr(CLR_RESET),
        rgb_str(clr_rgb::mode_val), clr(CLR_RESET),
        rgb_str(clr_rgb::heading), clr(CLR_RESET),
        rgb_str(clr_rgb::heading), clr(CLR_RESET),
        rgb_str(clr_rgb::heading), clr(CLR_RESET));
}

static Options parse_args(int argc, char** argv) {
    Options opts;
    opts.intervals   = 1000000000LL;
    opts.digits      = 0;
    strncpy(opts.output, "pi.txt", sizeof(opts.output));
    opts.max_size_mb = 10240;
    opts.mode        = MODE_CUDA;
    opts.algo        = ALGO_CHUDNOVSKY;
    opts.device_id   = 0;
    opts.show_help   = false;
    opts.print_digits = false;
    opts.benchmark       = false;
    opts.benchmark_quick = false;
    opts.benchmark_gpu   = false;
    opts.benchmark_cpu   = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            opts.show_help = true;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "cuda") == 0)        opts.mode = MODE_CUDA;
            else if (strcmp(argv[i], "opencl") == 0)  opts.mode = MODE_OPENCL;
            else if (strcmp(argv[i], "cpu") == 0)     opts.mode = MODE_CPU;
            else {
                fprintf(stderr, "%sError:%s Unknown mode '%s'. Use cuda, opencl, or cpu.\n",
                        rgb_str(clr_rgb::error), clr(CLR_RESET), argv[i]);
                opts.show_help = true;
            }
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            opts.device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cores") == 0 && i + 1 < argc) {
            opts.cores = parse_cores(argv[++i]);
        } else if (strcmp(argv[i], "--digits") == 0 && i + 1 < argc) {
            opts.digits = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "chudnovsky") == 0)      opts.algo = ALGO_CHUDNOVSKY;
            else if (strcmp(argv[i], "binsplit") == 0)     opts.algo = ALGO_BINSPLIT;
            else {
                fprintf(stderr, "%sError:%s Unknown algorithm '%s'. Use chudnovsky or binsplit.\n",
                        rgb_str(clr_rgb::error), clr(CLR_RESET), argv[i]);
                opts.show_help = true;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            strncpy(opts.output, argv[++i], sizeof(opts.output) - 1);
            opts.output[sizeof(opts.output) - 1] = '\0';
        } else if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            opts.max_size_mb = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--print") == 0) {
            opts.print_digits = true;
        } else if (strcmp(argv[i], "--no-print") == 0) {
            opts.print_digits = false;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            opts.benchmark = true;
        } else if (strcmp(argv[i], "--benchmark-quick") == 0) {
            opts.benchmark_quick = true;
        } else if (strcmp(argv[i], "--benchmark-gpu") == 0) {
            opts.benchmark_gpu = true;
        } else if (strcmp(argv[i], "--benchmark-cpu") == 0) {
            opts.benchmark_cpu = true;
        } else if (argv[i][0] != '-') {
            opts.intervals = atoll(argv[i]);
        } else {
            fprintf(stderr, "%sWarning:%s Unknown option: %s\n",
                    rgb_str(clr_rgb::warning), clr(CLR_RESET), argv[i]);
            opts.show_help = true;
        }
    }
    return opts;
}

// ===================================================================
//  Benchmark suite
// ===================================================================

// Benchmark table column widths
static const int BT_C1 = 10;  // Backend
static const int BT_C2 = 22;  // Test
static const int BT_C3 = 14;  // Time
static const int BT_C4 = 20;  // Performance

// Multi-column gradient horizontal rule for benchmark table
static void bench_table_hrule(const char* c0, const char* c1, const char* c2,
                               const char* c3, const char* c4) {
    const RGB tg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
    const int N = BT_C1 + BT_C2 + BT_C3 + BT_C4 + 5;
    int p = 0;
    auto tc = [&]() {
        RGB c = rgb_gradient(tg, 3, (double)p++ / (N - 1));
        set_rgb_fg(stdout, c.r, c.g, c.b);
    };
    printf("  ");
    tc(); printf("%s", c0);
    for (int i = 0; i < BT_C1; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s", c1);
    for (int i = 0; i < BT_C2; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s", c2);
    for (int i = 0; i < BT_C3; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s", c3);
    for (int i = 0; i < BT_C4; i++) { tc(); printf(BOX_H); }
    tc(); printf("%s%s\n", c4, clr(CLR_RESET));
}

// Print a benchmark table row: │ backend  │ test               │ time         │ perf               │
static void bench_table_row(const char* backend, const char* test,
                             const char* time_str, const char* perf_str,
                             RGB perf_rgb = clr_rgb::value) {
    const RGB tg[] = {{99,102,241}, {59,130,246}, {6,182,212}};
    printf("  ");
    set_rgb_fg(stdout, tg[0].r, tg[0].g, tg[0].b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(clr_rgb::mode_val), BT_C1 - 2, backend, clr(CLR_RESET));
    RGB m1 = rgb_gradient(tg, 3, 0.2);
    set_rgb_fg(stdout, m1.r, m1.g, m1.b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(clr_rgb::label), BT_C2 - 2, test, clr(CLR_RESET));
    RGB m2 = rgb_gradient(tg, 3, 0.5);
    set_rgb_fg(stdout, m2.r, m2.g, m2.b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(clr_rgb::value), BT_C3 - 2, time_str, clr(CLR_RESET));
    RGB m3 = rgb_gradient(tg, 3, 0.75);
    set_rgb_fg(stdout, m3.r, m3.g, m3.b);
    printf(BOX_V);
    printf("%s %s%-*s%s ", clr(CLR_RESET), rgb_str(perf_rgb), BT_C4 - 2, perf_str, clr(CLR_RESET));
    set_rgb_fg(stdout, tg[2].r, tg[2].g, tg[2].b);
    printf(BOX_V "%s\n", clr(CLR_RESET));
}

// Format elapsed_ms to human-readable time string
static std::string format_time(double ms) {
    char buf[32];
    if (ms >= 1000.0)
        snprintf(buf, sizeof(buf), "%.2f s", ms / 1000.0);
    else
        snprintf(buf, sizeof(buf), "%.1f ms", ms);
    return buf;
}

// Format metric with unit (e.g., "12.3M digits/s" or "4.56 Gint/s")
static std::string format_metric(double val, const char* unit) {
    char buf[48];
    if (strcmp(unit, "Gint/s") == 0) {
        snprintf(buf, sizeof(buf), "%.2f Gint/s", val);
    } else if (val >= 1e6) {
        snprintf(buf, sizeof(buf), "%.2fM %s", val / 1e6, unit);
    } else if (val >= 1e3) {
        snprintf(buf, sizeof(buf), "%.1fK %s", val / 1e3, unit);
    } else {
        snprintf(buf, sizeof(buf), "%.1f %s", val, unit);
    }
    return buf;
}

static void print_benchmark_summary(const std::vector<BenchmarkResult>& results) {
    printf("\n");
    print_section_header("Benchmark Summary");
    printf("\n");

    // Header
    bench_table_hrule(BOX_TL, BOX_TM, BOX_TM, BOX_TM, BOX_TR);
    bench_table_row("Backend", "Test", "Time", "Performance", clr_rgb::heading);
    bench_table_hrule(BOX_LM, BOX_CROSS, BOX_CROSS, BOX_CROSS, BOX_RM);

    // Results grouped by backend
    std::string last_backend;
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        if (!last_backend.empty() && r.backend != last_backend) {
            bench_table_hrule(BOX_LM, BOX_CROSS, BOX_CROSS, BOX_CROSS, BOX_RM);
        }
        last_backend = r.backend;

        if (r.success) {
            std::string t = format_time(r.elapsed_ms);
            std::string m = format_metric(r.metric, r.metric_unit.c_str());
            bench_table_row(r.backend.c_str(), r.test_name.c_str(),
                           t.c_str(), m.c_str(), clr_rgb::pi_val);
        } else {
            bench_table_row(r.backend.c_str(), r.test_name.c_str(),
                           "FAILED", "-", clr_rgb::error);
        }
    }

    bench_table_hrule(BOX_BL, BOX_BM, BOX_BM, BOX_BM, BOX_BR);
    printf("\n");
}

// Run GPU integration throughput benchmark (CUDA, OpenCL, CPU - all fast)
static void bench_gpu(std::vector<BenchmarkResult>& results) {
    bool has_cuda = cuda_device_count() > 0;
    bool has_opencl = opencl_device_count() > 0;
    long long integ_n = 1000000000LL;

    if (has_cuda) {
        fprintf(stderr, "  %sRunning CUDA integration (%s intervals)...%s\n",
                rgb_str(clr_rgb::dim), format_number(integ_n).c_str(), clr(CLR_RESET));
        auto r = cuda_integrate(integ_n, 0);
        results.push_back({"CUDA", "Integration 1B", r.elapsed_ms,
                           r.throughput_gips, "Gint/s", true});
    }

    if (has_opencl) {
        fprintf(stderr, "  %sRunning OpenCL integration (%s intervals)...%s\n",
                rgb_str(clr_rgb::dim), format_number(integ_n).c_str(), clr(CLR_RESET));
        auto r = opencl_integrate(integ_n, 0);
        results.push_back({"OpenCL", "Integration 1B", r.elapsed_ms,
                           r.throughput_gips, "Gint/s", true});
    }

    {
        fprintf(stderr, "  %sRunning CPU integration (%s intervals)...%s\n",
                rgb_str(clr_rgb::dim), format_number(integ_n).c_str(), clr(CLR_RESET));
        std::vector<int> no_cores;
        auto r = cpu_integrate(integ_n, no_cores);
        results.push_back({"CPU", "Integration 1B", r.elapsed_ms,
                           r.throughput_gips, "Gint/s", true});
    }
}

// Run CPU digit computation benchmark (Chudnovsky iterative + binary splitting)
static void bench_cpu(std::vector<BenchmarkResult>& results,
                      const std::vector<long long>& digit_counts) {
    for (auto digits : digit_counts) {
        int n_frac  = (int)((digits + 8) / 9);
        int n_limbs = 1 + n_frac + 2;

        char chud_label[64], bs_label[64];
        if (digits >= 1000000) {
            snprintf(chud_label, sizeof(chud_label), "Chudnovsky %lldM", digits / 1000000);
            snprintf(bs_label, sizeof(bs_label), "BinSplit %lldM", digits / 1000000);
        } else {
            snprintf(chud_label, sizeof(chud_label), "Chudnovsky %lldk", digits / 1000);
            snprintf(bs_label, sizeof(bs_label), "BinSplit %lldk", digits / 1000);
        }

        // CPU Chudnovsky (iterative)
        {
            uint32_t* pi = new uint32_t[n_limbs];
            std::vector<int> no_cores;
            fprintf(stderr, "  %sRunning CPU %s (%lld digits)...%s\n",
                    rgb_str(clr_rgb::dim), chud_label, digits, clr(CLR_RESET));
            auto t0 = std::chrono::steady_clock::now();
            cpu_compute_pi_digits(pi, n_limbs, digits, no_cores);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            double dps = (double)digits / (ms / 1000.0);
            results.push_back({"CPU", chud_label, ms, dps, "digits/s", true});
            delete[] pi;
        }

        // CPU binary splitting
        {
            uint32_t* pi = new uint32_t[n_limbs];
            std::vector<int> no_cores;
            fprintf(stderr, "  %sRunning CPU %s (%lld digits)...%s\n",
                    rgb_str(clr_rgb::dim), bs_label, digits, clr(CLR_RESET));
            auto t0 = std::chrono::steady_clock::now();
            cpu_compute_pi_digits_binsplit(pi, n_limbs, digits, no_cores);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            double dps = (double)digits / (ms / 1000.0);
            results.push_back({"CPU", bs_label, ms, dps, "digits/s", true});
            delete[] pi;
        }
    }
}

// Dispatch benchmark based on flags
static void run_benchmark(bool do_gpu, bool do_cpu,
                          const std::vector<long long>& digit_counts,
                          const char* title) {
    print_banner();
    printf("\n");
    print_section_header(title);
    printf("\n");

    std::vector<BenchmarkResult> results;

    if (do_gpu) bench_gpu(results);
    if (do_cpu) bench_cpu(results, digit_counts);

    print_benchmark_summary(results);
}

// ===================================================================
//  main
// ===================================================================

int main(int argc, char **argv) {
    init_color();

    Options opts = parse_args(argc, argv);
    if (opts.show_help) {
        print_banner();
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (opts.benchmark || opts.benchmark_quick || opts.benchmark_gpu || opts.benchmark_cpu) {
        if (opts.benchmark_gpu) {
            run_benchmark(true, false, {}, "GPU Benchmark");
        } else if (opts.benchmark_cpu) {
            run_benchmark(false, true, {10000, 100000, 1000000, 10000000},
                         "CPU Benchmark");
        } else if (opts.benchmark_quick) {
            run_benchmark(true, true, {10000, 100000}, "Quick Benchmark");
        } else {
            run_benchmark(true, true, {10000, 100000, 1000000, 10000000},
                         "Full Benchmark Suite");
        }
        return EXIT_SUCCESS;
    }
    // --algo binsplit is CPU-only
    if (opts.algo == ALGO_BINSPLIT && opts.mode != MODE_CPU) {
        fprintf(stderr, "%sError:%s --algo binsplit is only supported with --mode cpu.\n",
                rgb_str(clr_rgb::error), clr(CLR_RESET));
        return EXIT_FAILURE;
    }

    if (opts.intervals <= 0) {
        fprintf(stderr, "%sError:%s Interval count must be positive.\n",
                rgb_str(clr_rgb::error), clr(CLR_RESET));
        return EXIT_FAILURE;
    }

    // ---- Banner ----
    print_banner();

    // ---- Device discovery ----
    list_all_devices();

    // ---- Determine mode / algo strings ----
    const char* mode_str;
    switch (opts.mode) {
        case MODE_CUDA:   mode_str = "CUDA";   break;
        case MODE_OPENCL: mode_str = "OpenCL"; break;
        case MODE_CPU:    mode_str = "CPU";    break;
    }
    const char* algo_str;
    switch (opts.algo) {
        case ALGO_CHUDNOVSKY: algo_str = "chudnovsky"; break;
        case ALGO_BINSPLIT:   algo_str = "binsplit";    break;
    }

    printf("  %sMode:%s %s%s%s  |  %sIntervals:%s %s%s%s",
           rgb_str(clr_rgb::label), clr(CLR_RESET),
           rgb_str(clr_rgb::mode_val), mode_str, clr(CLR_RESET),
           rgb_str(clr_rgb::label), clr(CLR_RESET),
           rgb_str(clr_rgb::value), format_number(opts.intervals).c_str(), clr(CLR_RESET));

    if (opts.mode == MODE_CPU && !opts.cores.empty()) {
        printf("  |  %sCores:%s ", rgb_str(clr_rgb::label), clr(CLR_RESET));
        for (size_t i = 0; i < opts.cores.size(); i++) {
            if (i > 0) printf(",");
            printf("%s%d%s", rgb_str(clr_rgb::value), opts.cores[i], clr(CLR_RESET));
        }
    } else if (opts.mode != MODE_CPU) {
        printf("  |  %sDevice:%s %s%d%s",
               rgb_str(clr_rgb::label), clr(CLR_RESET),
               rgb_str(clr_rgb::value), opts.device_id, clr(CLR_RESET));
    }
    printf("\n");
    fflush(stdout);

    // ---- Run integration ----
    ComputeResult result;
    switch (opts.mode) {
        case MODE_CUDA:
            result = cuda_integrate(opts.intervals, opts.device_id);
            break;
        case MODE_OPENCL:
            result = opencl_integrate(opts.intervals, opts.device_id);
            break;
        case MODE_CPU:
            result = cpu_integrate(opts.intervals, opts.cores);
            break;
    }

    // ---- Summary table ----
    print_results_table(result, mode_str);

    // ---- NVML stats (CUDA mode only) ----
#if HAS_NVML
    if (opts.mode == MODE_CUDA) {
        print_nvml_stats(opts.device_id);
    }
#endif

    // ================================================================
    //  Arbitrary-precision digit computation (if requested)
    // ================================================================
    if (opts.digits > 0) {
        long long max_digits = opts.max_size_mb * 1000000LL;
        if (opts.digits > max_digits) {
            fprintf(stderr, "  %sNote:%s Clamping digits to %lld (--max-size %lld MB)\n",
                    rgb_str(clr_rgb::warning), clr(CLR_RESET),
                    max_digits, (long long)opts.max_size_mb);
            opts.digits = max_digits;
        }

        int n_frac  = (int)((opts.digits + 8) / 9);
        int n_limbs = 1 + n_frac + 2;

        printf("\n");
        print_section_header("Arbitrary-Precision Digits");
        printf("\n");
        fprintf(stderr, "  Computing %s%lld%s decimal digits of \xcf\x80 [%s%s%s]...\n",
                rgb_str(clr_rgb::value), (long long)opts.digits, clr(CLR_RESET),
                rgb_str(clr_rgb::dim), algo_str, clr(CLR_RESET));
        fprintf(stderr, "  %sAllocating %.1f MB for big-number arrays...%s\n",
                rgb_str(clr_rgb::dim),
                (double)n_limbs * sizeof(uint32_t) * 4 / 1048576.0,
                clr(CLR_RESET));

        uint32_t* pi_digits = new uint32_t[n_limbs];
        if (opts.algo == ALGO_BINSPLIT) {
            // Binary splitting - CPU only (validated above)
            cpu_compute_pi_digits_binsplit(pi_digits, n_limbs, opts.digits, opts.cores);
        } else {
            switch (opts.mode) {
                case MODE_CUDA:
                    cuda_compute_pi_digits(pi_digits, n_limbs, opts.digits);
                    break;
                case MODE_OPENCL:
                    opencl_compute_pi_digits(pi_digits, n_limbs, opts.digits, opts.device_id);
                    break;
                case MODE_CPU:
                    cpu_compute_pi_digits(pi_digits, n_limbs, opts.digits, opts.cores);
                    break;
            }
        }

        // Print to terminal (only if --print flag is set)
        if (opts.print_digits) {
            printf("  %s%u.%s", rgb_str(clr_rgb::pi_val),
                   pi_digits[0], clr(CLR_RESET));
            long long printed = 0;
            for (int i = 1; i < n_limbs && printed < opts.digits; i++) {
                char buf[10];
                snprintf(buf, sizeof(buf), "%09u", pi_digits[i]);
                for (int j = 0; j < 9 && printed < opts.digits; j++) {
                    printf("%c", buf[j]);
                    printed++;
                    // Add space every 10 digits for readability (first 100 digits)
                    if (printed <= 100 && printed % 10 == 0 && printed < opts.digits)
                        printf(" ");
                }
            }
            printf("\n");

            if (printed > 100)
                printf("  %s(%lld digits total)%s\n",
                       rgb_str(clr_rgb::dim), printed, clr(CLR_RESET));
        }

        // Write to file
        FILE* fp = fopen(opts.output, "w");
        if (!fp) {
            fprintf(stderr, "  %sError:%s Could not open %s for writing.\n",
                    rgb_str(clr_rgb::error), clr(CLR_RESET), opts.output);
        } else {
            fprintf(fp, "%u.", pi_digits[0]);
            long long written = 0;
            for (int i = 1; i < n_limbs && written < opts.digits; i++) {
                char buf[10];
                snprintf(buf, sizeof(buf), "%09u", pi_digits[i]);
                for (int j = 0; j < 9 && written < opts.digits; j++) {
                    fputc(buf[j], fp);
                    written++;
                }
            }
            fprintf(fp, "\n");
            fclose(fp);

            fprintf(stderr, "  %s\xe2\x9c\x93%s Wrote %lld digits to %s%s%s\n",
                    rgb_str(clr_rgb::success), clr(CLR_RESET),
                    written, rgb_str(clr_rgb::value), opts.output, clr(CLR_RESET));
        }

        delete[] pi_digits;
    }

    return EXIT_SUCCESS;
}

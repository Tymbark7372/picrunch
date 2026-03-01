/**
 * pi_opencl.cpp - OpenCL backend for pi integration.
 *
 * Midpoint-rule integration of 4/(1+x²) over [0,1] with Kahan summation,
 * matching the CUDA kernel algorithm for consistency.
 */

#include "backends.h"

#ifdef HAS_OPENCL

#define CL_TARGET_OPENCL_VERSION 120   // Target OpenCL 1.2 for broad compatibility
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include <cstdlib>
#include <thread>

#ifdef __linux__
#  include <unistd.h>   // access()
#  include <dirent.h>   // opendir/readdir
#endif

// ===================================================================
//  OpenCL kernel source - embedded as a string
// ===================================================================

static const char* ocl_kernel_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Each work-item computes a strided subset of midpoints with Kahan summation,
// then writes its partial sum to the output buffer.
__kernel void integrate(__global double* partial_sums, long N) {
    int gid   = get_global_id(0);
    long total = (long)get_global_size(0);

    double step = 1.0 / (double)N;
    double sum  = 0.0;
    double comp = 0.0;

    for (long i = gid; i < N; i += total) {
        double x = ((double)i + 0.5) * step;
        double f = 4.0 / (1.0 + x * x);
        // Kahan compensated addition
        double y = f - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }

    partial_sums[gid] = sum * step;
}
)CL";

// ===================================================================
//  Helper: check OpenCL error
// ===================================================================

#define OCL_CHECK(call, msg)                                           \
    do {                                                                \
        cl_int _err = (call);                                           \
        if (_err != CL_SUCCESS) {                                       \
            fprintf(stderr, "OpenCL error at %s:%d - %s (code %d)\n",  \
                    __FILE__, __LINE__, msg, _err);                     \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ===================================================================
//  Ensure NVIDIA OpenCL ICD is discoverable
//
//  In Docker/container environments the NVIDIA driver runtime mounts
//  libnvidia-opencl.so.1 into the container, but the ICD file
//  (/etc/OpenCL/vendors/nvidia.icd) that the ocl-icd loader reads
//  is often absent.  Without it clGetPlatformIDs returns 0 platforms.
//
//  Fix: probe well-known paths for the NVIDIA OpenCL library and, if
//  found, tell the loader about it via OCL_ICD_FILENAMES before the
//  first OpenCL call.  The env var is set with overwrite=0 so an
//  explicit user setting always takes precedence.
// ===================================================================

static void ensure_opencl_icd() {
#ifdef __linux__
    // If the user already set this, trust their config.
    if (getenv("OCL_ICD_FILENAMES")) return;

    // Check specifically for an NVIDIA .icd file.  Other ICDs (mesa,
    // pocl) don't expose NVIDIA GPUs, so their presence alone is not
    // enough to skip the manual library probe below.
    bool have_nvidia_icd = false;
    if (DIR* d = opendir("/etc/OpenCL/vendors")) {
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            if (strstr(e->d_name, "nvidia") != nullptr) {
                size_t len = strlen(e->d_name);
                if (len > 4 && strcmp(e->d_name + len - 4, ".icd") == 0) {
                    have_nvidia_icd = true;
                    break;
                }
            }
        }
        closedir(d);
    }
    if (have_nvidia_icd) return;

    // No ICD files - look for the NVIDIA OpenCL library at common paths.
    static const char* candidates[] = {
        "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1",
        "/usr/lib64/libnvidia-opencl.so.1",
        "/usr/local/cuda/lib64/libnvidia-opencl.so.1",
        "/usr/lib/aarch64-linux-gnu/libnvidia-opencl.so.1",
        nullptr
    };
    for (int i = 0; candidates[i]; i++) {
        if (access(candidates[i], R_OK) == 0) {
            setenv("OCL_ICD_FILENAMES", candidates[i], 0);
            return;
        }
    }
#endif
}

// ===================================================================
//  Gather all OpenCL devices across all platforms
// ===================================================================

struct OclDeviceEntry {
    cl_platform_id platform;
    cl_device_id   device;
    std::string    name;
    std::string    vendor;
    std::string    type_str;
    cl_ulong       global_mem;
};

static std::vector<OclDeviceEntry> gather_ocl_devices() {
    std::vector<OclDeviceEntry> result;

    // Make sure the NVIDIA OpenCL ICD is registered before the first
    // OpenCL call (the loader reads OCL_ICD_FILENAMES on init).
    static bool icd_init = false;
    if (!icd_init) {
        icd_init = true;
        ensure_opencl_icd();
    }

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return result;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices) != CL_SUCCESS)
            continue;
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        for (cl_uint d = 0; d < num_devices; d++) {
            OclDeviceEntry entry;
            entry.platform = platforms[p];
            entry.device   = devices[d];

            char buf[256];
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
            entry.name = buf;

            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
            entry.vendor = buf;

            cl_device_type dtype;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            if (dtype & CL_DEVICE_TYPE_GPU)         entry.type_str = "GPU";
            else if (dtype & CL_DEVICE_TYPE_CPU)    entry.type_str = "CPU";
            else if (dtype & CL_DEVICE_TYPE_ACCELERATOR) entry.type_str = "Accelerator";
            else                                     entry.type_str = "Other";

            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(entry.global_mem),
                            &entry.global_mem, nullptr);

            result.push_back(entry);
        }
    }
    return result;
}

// ===================================================================
//  Public API
// ===================================================================

int opencl_device_count() {
    return (int)gather_ocl_devices().size();
}

std::vector<DeviceInfo> opencl_list_devices() {
    auto devs = gather_ocl_devices();
    std::vector<DeviceInfo> out;
    for (int i = 0; i < (int)devs.size(); i++) {
        DeviceInfo info;
        info.index = i;
        info.name  = devs[i].name;
        char detail[128];
        snprintf(detail, sizeof(detail), "%s, %s, %.0f MB",
                 devs[i].vendor.c_str(), devs[i].type_str.c_str(),
                 (double)devs[i].global_mem / 1048576.0);
        info.details = detail;
        out.push_back(info);
    }
    return out;
}

ComputeResult opencl_integrate(long long N, int device_id) {
    auto devs = gather_ocl_devices();
    if (devs.empty()) {
        fprintf(stderr, "%sError:%s No OpenCL devices found.\n",
                clr(CLR_RED), clr(CLR_RESET));
        exit(EXIT_FAILURE);
    }
    if (device_id < 0 || device_id >= (int)devs.size()) {
        fprintf(stderr, "%sError:%s OpenCL device index %d out of range (0..%d).\n",
                clr(CLR_RED), clr(CLR_RESET), device_id, (int)devs.size() - 1);
        exit(EXIT_FAILURE);
    }

    auto& dev = devs[device_id];
    cl_int err;

    // Create context and command queue for this single device
    cl_context ctx = clCreateContext(nullptr, 1, &dev.device, nullptr, nullptr, &err);
    OCL_CHECK(err, "clCreateContext");

    // Use clCreateCommandQueue (OpenCL 1.x compatible; 2.0+ has a different call)
    cl_command_queue queue = clCreateCommandQueue(ctx, dev.device, 0, &err);
    OCL_CHECK(err, "clCreateCommandQueue");

    // Build the kernel
    const char* src = ocl_kernel_src;
    size_t src_len = strlen(ocl_kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev.device, "-cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, dev.device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(prog, "integrate", &err);
    OCL_CHECK(err, "clCreateKernel");

    // Determine work size: use many work-items for good utilization
    // Query max work-group size
    size_t max_wg;
    clGetDeviceInfo(dev.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
    cl_uint compute_units;
    clGetDeviceInfo(dev.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units),
                    &compute_units, nullptr);

    size_t local_size  = (max_wg > 256) ? 256 : max_wg;
    size_t global_size = compute_units * 8 * local_size;
    if (global_size > 65536 * local_size) global_size = 65536 * local_size;

    int num_work_items = (int)global_size;

    // Allocate output buffer
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                num_work_items * sizeof(double), nullptr, &err);
    OCL_CHECK(err, "clCreateBuffer");

    // Set kernel arguments
    // OpenCL 'long' is always 64-bit; use cl_long to match on all platforms
    // (MSVC 'long' is 32-bit, causing CL_INVALID_ARG_SIZE if used directly)
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf), "setArg 0");
    cl_long cl_N = (cl_long)N;
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_long), &cl_N), "setArg 1");

    // Launch with timing
    fprintf(stderr, "\n");
    auto wall_start = std::chrono::steady_clock::now();

    OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                      &global_size, &local_size, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel");

    // Spinner while waiting for completion
    int tick = 0;
    clFlush(queue);
    while (true) {
        cl_int finish_err = clFinish(queue);
        // clFinish blocks, so we can't really spin. Instead just show a quick message.
        (void)finish_err;
        break;
    }
    // Note: clFinish is blocking, so we just mark the time
    auto wall_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    ui_spinner_done("Computing \xcf\x80 (OpenCL)", elapsed_ms);

    // Read back partial sums
    std::vector<double> partial_sums(num_work_items);
    OCL_CHECK(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0,
                                   num_work_items * sizeof(double),
                                   partial_sums.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer");

    // Host-side Kahan reduction
    double pi_val = 0.0, comp = 0.0;
    for (int i = 0; i < num_work_items; i++) {
        double y = partial_sums[i] - comp;
        double t = pi_val + y;
        comp = (t - pi_val) - y;
        pi_val = t;
    }

    // Cleanup
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    ComputeResult res;
    res.pi_value       = pi_val;
    res.elapsed_ms     = elapsed_ms;
    res.intervals      = N;
    res.device_name    = dev.name;
    res.num_threads    = num_work_items;
    res.throughput_gips = (double)N / (elapsed_ms * 1e6);
    return res;
}

// ===================================================================
//  OpenCL Chudnovsky - arbitrary-precision digit computation
//
//  Uses the same iterative Chudnovsky recurrence as the CUDA version,
//  with big-number operations implemented as OpenCL kernels.
// ===================================================================

// OpenCL kernel source for big-number operations
static const char* ocl_bn_kernel_src = R"CL(
// Big-number add: a[] += b[], with carry propagation
// Single work-item kernel (carry is sequential)
__kernel void bn_add(__global uint* a, __global const uint* b, int n) {
    ulong carry = 0;
    for (int i = n - 1; i >= 0; i--) {
        ulong val = (ulong)a[i] + b[i] + carry;
        a[i] = (uint)(val % 1000000000UL);
        carry = val / 1000000000UL;
    }
}

// Big-number subtract: a[] -= b[]
__kernel void bn_sub(__global uint* a, __global const uint* b, int n) {
    long borrow = 0;
    for (int i = n - 1; i >= 0; i--) {
        long val = (long)a[i] - (long)b[i] - borrow;
        if (val < 0) {
            val += 1000000000L;
            borrow = 1;
        } else {
            borrow = 0;
        }
        a[i] = (uint)val;
    }
}

// Big-number multiply by small scalar: a[] *= m
__kernel void bn_mul_small(__global uint* a, int n, ulong m) {
    ulong carry = 0;
    for (int i = n - 1; i >= 0; i--) {
        ulong val = (ulong)a[i] * m + carry;
        a[i] = (uint)(val % 1000000000UL);
        carry = val / 1000000000UL;
    }
}

// Big-number divide by small scalar: a[] /= d
__kernel void bn_divide(__global uint* a, int n, ulong d) {
    ulong carry = 0;
    for (int i = 0; i < n; i++) {
        ulong val = carry * 1000000000UL + a[i];
        a[i] = (uint)(val / d);
        carry = val % d;
    }
}

// Device-to-device copy
__kernel void bn_copy(__global uint* dst, __global const uint* src, int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        dst[gid] = src[gid];
    }
}

// Schoolbook multiply: out[] = a[] * b[] (2*n limbs output)
// Single work-item for simplicity (used only for sqrt)
__kernel void bn_multiply(__global uint* out, __global const uint* a,
                           __global const uint* b, int n) {
    for (int i = 0; i < 2 * n; i++) out[i] = 0;

    for (int i = 0; i < n; i++) {
        if (a[i] == 0) continue;
        ulong carry = 0;
        for (int j = n - 1; j >= 0; j--) {
            int pos = i + j;
            ulong prod = (ulong)a[i] * b[j] + (ulong)out[pos] + carry;
            out[pos] = (uint)(prod % 1000000000UL);
            carry = prod / 1000000000UL;
        }
        if (carry > 0 && i > 0) {
            out[i - 1] += (uint)carry;
        }
    }
    // Carry propagation
    ulong carry = 0;
    for (int i = 2 * n - 1; i >= 0; i--) {
        ulong val = (ulong)out[i] + carry;
        out[i] = (uint)(val % 1000000000UL);
        carry = val / 1000000000UL;
    }
}
)CL";

void opencl_compute_pi_digits(uint32_t* pi_digits, int n_limbs,
                               long long n_digits, int device_id) {
    auto devs = gather_ocl_devices();
    if (devs.empty()) {
        fprintf(stderr, "%sError:%s No OpenCL devices found.\n",
                clr(CLR_RED), clr(CLR_RESET));
        exit(EXIT_FAILURE);
    }
    if (device_id < 0 || device_id >= (int)devs.size()) {
        fprintf(stderr, "%sError:%s OpenCL device index %d out of range (0..%d).\n",
                clr(CLR_RED), clr(CLR_RESET), device_id, (int)devs.size() - 1);
        exit(EXIT_FAILURE);
    }

    auto& dev = devs[device_id];
    cl_int err;

    cl_context ctx = clCreateContext(nullptr, 1, &dev.device, nullptr, nullptr, &err);
    OCL_CHECK(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, dev.device, 0, &err);
    OCL_CHECK(err, "clCreateCommandQueue");

    // Build big-number kernels
    const char* src = ocl_bn_kernel_src;
    size_t src_len = strlen(ocl_bn_kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource (bn)");

    err = clBuildProgram(prog, 1, &dev.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, dev.device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    cl_kernel k_add       = clCreateKernel(prog, "bn_add", &err);       OCL_CHECK(err, "bn_add");
    cl_kernel k_sub       = clCreateKernel(prog, "bn_sub", &err);       OCL_CHECK(err, "bn_sub");
    cl_kernel k_mul_small = clCreateKernel(prog, "bn_mul_small", &err); OCL_CHECK(err, "bn_mul_small");
    cl_kernel k_divide    = clCreateKernel(prog, "bn_divide", &err);    OCL_CHECK(err, "bn_divide");
    cl_kernel k_copy      = clCreateKernel(prog, "bn_copy", &err);      OCL_CHECK(err, "bn_copy");
    cl_kernel k_multiply  = clCreateKernel(prog, "bn_multiply", &err);  OCL_CHECK(err, "bn_multiply");

    size_t limb_bytes = n_limbs * sizeof(uint32_t);

    // Allocate device buffers
    cl_mem d_term    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf term");
    cl_mem d_temp    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf temp");
    cl_mem d_temp2   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf temp2");
    cl_mem d_sum_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf sum_pos");
    cl_mem d_sum_neg = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf sum_neg");
    cl_mem d_sqrt_v  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, limb_bytes, nullptr, &err); OCL_CHECK(err, "buf sqrt");
    cl_mem d_temp2n  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2 * limb_bytes, nullptr, &err); OCL_CHECK(err, "buf temp2n");

    // Helper lambdas for kernel dispatch
    size_t one = 1;
    size_t glob_n = (size_t)n_limbs;

    auto ocl_add = [&](cl_mem a, cl_mem b) {
        clSetKernelArg(k_add, 0, sizeof(cl_mem), &a);
        clSetKernelArg(k_add, 1, sizeof(cl_mem), &b);
        clSetKernelArg(k_add, 2, sizeof(int), &n_limbs);
        clEnqueueNDRangeKernel(queue, k_add, 1, nullptr, &one, &one, 0, nullptr, nullptr);
    };

    auto ocl_sub = [&](cl_mem a, cl_mem b) {
        clSetKernelArg(k_sub, 0, sizeof(cl_mem), &a);
        clSetKernelArg(k_sub, 1, sizeof(cl_mem), &b);
        clSetKernelArg(k_sub, 2, sizeof(int), &n_limbs);
        clEnqueueNDRangeKernel(queue, k_sub, 1, nullptr, &one, &one, 0, nullptr, nullptr);
    };

    auto ocl_mul_small = [&](cl_mem a, uint64_t m) {
        cl_ulong cm = (cl_ulong)m;
        clSetKernelArg(k_mul_small, 0, sizeof(cl_mem), &a);
        clSetKernelArg(k_mul_small, 1, sizeof(int), &n_limbs);
        clSetKernelArg(k_mul_small, 2, sizeof(cl_ulong), &cm);
        clEnqueueNDRangeKernel(queue, k_mul_small, 1, nullptr, &one, &one, 0, nullptr, nullptr);
    };

    auto ocl_divide = [&](cl_mem a, uint64_t d) {
        cl_ulong cd = (cl_ulong)d;
        clSetKernelArg(k_divide, 0, sizeof(cl_mem), &a);
        clSetKernelArg(k_divide, 1, sizeof(int), &n_limbs);
        clSetKernelArg(k_divide, 2, sizeof(cl_ulong), &cd);
        clEnqueueNDRangeKernel(queue, k_divide, 1, nullptr, &one, &one, 0, nullptr, nullptr);
    };

    auto ocl_copy = [&](cl_mem dst, cl_mem src_buf) {
        clSetKernelArg(k_copy, 0, sizeof(cl_mem), &dst);
        clSetKernelArg(k_copy, 1, sizeof(cl_mem), &src_buf);
        clSetKernelArg(k_copy, 2, sizeof(int), &n_limbs);
        clEnqueueNDRangeKernel(queue, k_copy, 1, nullptr, &glob_n, nullptr, 0, nullptr, nullptr);
    };

    auto ocl_multiply = [&](cl_mem out, cl_mem a, cl_mem b) {
        clSetKernelArg(k_multiply, 0, sizeof(cl_mem), &out);
        clSetKernelArg(k_multiply, 1, sizeof(cl_mem), &a);
        clSetKernelArg(k_multiply, 2, sizeof(cl_mem), &b);
        clSetKernelArg(k_multiply, 3, sizeof(int), &n_limbs);
        clEnqueueNDRangeKernel(queue, k_multiply, 1, nullptr, &one, &one, 0, nullptr, nullptr);
    };

    // Initialize: zero all buffers
    uint32_t* zeros = new uint32_t[2 * n_limbs]();
    clEnqueueWriteBuffer(queue, d_term, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_temp, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_temp2, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_sum_pos, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_sum_neg, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_sqrt_v, CL_TRUE, 0, limb_bytes, zeros, 0, nullptr, nullptr);

    // term(0) = 1.0
    uint32_t one_val = 1;
    clEnqueueWriteBuffer(queue, d_term, CL_TRUE, 0, sizeof(uint32_t), &one_val, 0, nullptr, nullptr);

    // sum_pos = 13591409 (k=0 contribution)
    uint32_t init_a = 13591409u;
    clEnqueueWriteBuffer(queue, d_sum_pos, CL_TRUE, 0, sizeof(uint32_t), &init_a, 0, nullptr, nullptr);

    int n_terms = (int)(n_digits / 14.0) + 10;
    int report_interval = n_terms / 100;
    if (report_interval < 1) report_interval = 1;

    auto start_time = std::chrono::steady_clock::now();

    for (int k = 1; k <= n_terms; k++) {
        uint64_t num1 = (uint64_t)(6*k - 5);
        uint64_t num2 = (uint64_t)(2*k - 1);
        uint64_t num3 = (uint64_t)(6*k - 1);

        ocl_mul_small(d_term, num1);
        ocl_mul_small(d_term, num2);
        ocl_mul_small(d_term, num3);

        uint64_t kk = (uint64_t)k;
        ocl_divide(d_term, kk);
        ocl_divide(d_term, kk);
        ocl_divide(d_term, kk);
        ocl_divide(d_term, 26680ULL);
        ocl_divide(d_term, 640320ULL);
        ocl_divide(d_term, 640320ULL);

        // Compute contribution
        uint64_t ak = 13591409ULL + 545140134ULL * kk;

        if (ak <= 0xFFFFFFFFULL) {
            ocl_copy(d_temp, d_term);
            ocl_mul_small(d_temp, ak);
        } else {
            ocl_copy(d_temp, d_term);
            ocl_mul_small(d_temp, 545140134ULL);
            ocl_mul_small(d_temp, kk);
            ocl_copy(d_temp2, d_term);
            ocl_mul_small(d_temp2, 13591409ULL);
            ocl_add(d_temp, d_temp2);
        }

        if (k % 2 == 0) {
            ocl_add(d_sum_pos, d_temp);
        } else {
            ocl_add(d_sum_neg, d_temp);
        }

        if (k % report_interval == 0) {
            clFinish(queue);
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start_time).count();
            ui_progress_bar("Chudnovsky series (OpenCL)", (double)k / n_terms, elapsed_s);
        }
    }

    clFinish(queue);
    auto series_end = std::chrono::steady_clock::now();
    double series_ms = std::chrono::duration<double, std::milli>(series_end - start_time).count();
    ui_progress_done("Chudnovsky series (OpenCL)", series_ms);

    // sum = sum_pos - sum_neg
    ocl_sub(d_sum_pos, d_sum_neg);  // d_sum_pos now holds the sum

    // Compute sqrt(10005) using Newton reciprocal sqrt on device
    fprintf(stderr, "  %sComputing sqrt(10005) on OpenCL...%s\n",
            clr(CLR_DIM), clr(CLR_RESET));
    auto sqrt_start = std::chrono::steady_clock::now();

    // Initialize y = 1/sqrt(10005) from double precision
    {
        uint32_t* h_y = new uint32_t[n_limbs]();
        double y0 = 1.0 / std::sqrt(10005.0);
        h_y[0] = (uint32_t)y0;
        double frac = y0 - (double)h_y[0];
        for (int i = 1; i < n_limbs && i < 4; i++) {
            frac *= 1e9;
            h_y[i] = (uint32_t)frac;
            frac -= (double)h_y[i];
        }
        clEnqueueWriteBuffer(queue, d_sqrt_v, CL_TRUE, 0, limb_bytes, h_y, 0, nullptr, nullptr);
        delete[] h_y;
    }

    // Newton iterations: y_{n+1} = y + y*(1 - val*y^2)/2
    int rsqrt_iters = 0;
    {
        double bits_needed = (double)n_limbs * 9.0 * 3.321928;
        rsqrt_iters = (int)(std::log2(bits_needed)) + 5;
        if (rsqrt_iters < 20) rsqrt_iters = 20;
        if (rsqrt_iters > 100) rsqrt_iters = 100;
    }

    uint32_t* h_one = new uint32_t[n_limbs]();
    h_one[0] = 1;

    for (int iter = 0; iter < rsqrt_iters; iter++) {
        // temp = y * y
        ocl_multiply(d_temp2n, d_sqrt_v, d_sqrt_v);
        // Copy first n limbs to temp
        clEnqueueCopyBuffer(queue, d_temp2n, d_temp, 0, 0, limb_bytes, 0, nullptr, nullptr);

        // temp *= 10005
        ocl_mul_small(d_temp, 10005ULL);

        // Check if temp >= 1
        uint32_t h_check;
        clFinish(queue);
        clEnqueueReadBuffer(queue, d_temp, CL_TRUE, 0, sizeof(uint32_t), &h_check, 0, nullptr, nullptr);

        // temp2 = 1.0
        clEnqueueWriteBuffer(queue, d_temp2, CL_TRUE, 0, limb_bytes, h_one, 0, nullptr, nullptr);

        if (h_check >= 1) {
            // val*y^2 >= 1: subtract 1, compute correction, subtract from y
            ocl_sub(d_temp, d_temp2);
            ocl_multiply(d_temp2n, d_sqrt_v, d_temp);
            clEnqueueCopyBuffer(queue, d_temp2n, d_temp2, 0, 0, limb_bytes, 0, nullptr, nullptr);
            ocl_divide(d_temp2, 2);
            ocl_sub(d_sqrt_v, d_temp2);
        } else {
            // val*y^2 < 1: subtract from 1, compute correction, add to y
            ocl_sub(d_temp2, d_temp);
            ocl_multiply(d_temp2n, d_sqrt_v, d_temp2);
            clEnqueueCopyBuffer(queue, d_temp2n, d_temp, 0, 0, limb_bytes, 0, nullptr, nullptr);
            ocl_divide(d_temp, 2);
            ocl_add(d_sqrt_v, d_temp);
        }
    }

    // result = 10005 * y = sqrt(10005)
    ocl_mul_small(d_sqrt_v, 10005ULL);
    clFinish(queue);

    auto sqrt_end = std::chrono::steady_clock::now();
    double sqrt_ms = std::chrono::duration<double, std::milli>(sqrt_end - sqrt_start).count();
    ui_spinner_done("sqrt(10005) (OpenCL)", sqrt_ms);

    // Final: π = 426880 * sqrt(10005) / sum
    ocl_mul_small(d_sqrt_v, 426880ULL);
    clFinish(queue);

    // Read back numerator and denominator for host-side final division
    uint32_t* h_numerator = new uint32_t[n_limbs];
    uint32_t* h_denominator = new uint32_t[n_limbs];
    clEnqueueReadBuffer(queue, d_sqrt_v, CL_TRUE, 0, limb_bytes, h_numerator, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_sum_pos, CL_TRUE, 0, limb_bytes, h_denominator, 0, nullptr, nullptr);

    // Long division on host (Knuth-style estimate with n+1 limb remainder)
    memset(pi_digits, 0, n_limbs * sizeof(uint32_t));
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

            for (int j = 0; j < n_limbs; j++) rem[j] = rem[j + 1];
            rem[n_limbs] = 0;
        }
        delete[] rem;
    }

    delete[] h_numerator;
    delete[] h_denominator;
    delete[] h_one;
    delete[] zeros;

    // Release OpenCL resources
    clReleaseMemObject(d_term);
    clReleaseMemObject(d_temp);
    clReleaseMemObject(d_temp2);
    clReleaseMemObject(d_sum_pos);
    clReleaseMemObject(d_sum_neg);
    clReleaseMemObject(d_sqrt_v);
    clReleaseMemObject(d_temp2n);
    clReleaseKernel(k_add);
    clReleaseKernel(k_sub);
    clReleaseKernel(k_mul_small);
    clReleaseKernel(k_divide);
    clReleaseKernel(k_copy);
    clReleaseKernel(k_multiply);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}

#else // !HAS_OPENCL - stub implementations

int opencl_device_count() { return 0; }

std::vector<DeviceInfo> opencl_list_devices() { return {}; }

ComputeResult opencl_integrate(long long N, int device_id) {
    (void)N; (void)device_id;
    fprintf(stderr, "%sError:%s OpenCL support was not compiled in.\n"
                    "  Rebuild with -DENABLE_OPENCL=ON (requires OpenCL headers + ICD loader).\n",
            clr(CLR_RED), clr(CLR_RESET));
    exit(EXIT_FAILURE);
}

void opencl_compute_pi_digits(uint32_t* pi_digits, int n_limbs,
                               long long n_digits, int device_id) {
    (void)pi_digits; (void)n_limbs; (void)n_digits; (void)device_id;
    fprintf(stderr, "%sError:%s OpenCL support was not compiled in.\n"
                    "  Rebuild with -DENABLE_OPENCL=ON (requires OpenCL headers + ICD loader).\n",
            clr(CLR_RED), clr(CLR_RESET));
    exit(EXIT_FAILURE);
}

#endif // HAS_OPENCL

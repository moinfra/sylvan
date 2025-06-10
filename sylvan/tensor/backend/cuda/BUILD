load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "cuda_backend_impl",
    srcs = [
        "src/operators.cc",
        "src/tensor.cc",
        "src/cuda/kernels.cu.cc",
    ],
    hdrs = [
        "include/kernels.h",
        "include/macros.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//sylvan/tensor",
        "@local_cuda//:cublas",
        "@local_cuda//:cuda_runtime",
        "@local_cuda//:curand",
    ],
)

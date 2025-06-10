# sylvan/tensor/BUILD

load("@rules_cc//cc:defs.bzl", "cc_library")

# sylvan_tensor 核心库
cc_library(
    name = "tensor",
    srcs = glob(["src/**/*.cc", "src/**/*.cu.cc"]),
    hdrs = glob(["include/sylvan/tensor/**/*.h"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    # FIX: 使用 deps 而不是 linkopts 来引入 CUDA 库。
    # 这是更健壮的 "Bazel 方式"。
    deps = select({
        # 当 --config=cuda 启用时，添加对 cublas 和 cudart 库目标的依赖。
        # 这些目标由 rules_cuda 提供，并包含了所有必要的链接信息（路径和库名）。
        "@rules_cuda//cuda:is_enabled": [
            "@local_cuda//:cublas",
            "@local_cuda//:cuda_runtime",
        ],
        "//conditions:default": [],
    }),
)

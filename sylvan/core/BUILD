load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "core",
    srcs = glob(["src/**/*.cc"]),
    hdrs = glob(["include/sylvan/core/**/*.h"]),
    strip_include_prefix = "include",
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//sylvan/tensor",
        "//sylvan/tensor/backend/cuda:cuda_backend_impl",
    ],
)

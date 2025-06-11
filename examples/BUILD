load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "linear_regression",
    srcs = ["linear_regression.cc"],
    deps = [ "//sylvan/core" ],
)

cc_binary(
    name = "number_translator",
    srcs = ["number_translator.cc"],
    deps = [
        "//sylvan/core",
        "@local_cuda//:cublas",
        "@local_cuda//:cuda_runtime",
        "@local_cuda//:curand",
    ],
)

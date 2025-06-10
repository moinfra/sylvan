load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        "//sylvan/tensor",
        "@googletest//:gtest_main", # 依赖于 googletest
    ],
)

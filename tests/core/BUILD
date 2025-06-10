load("@rules_cc//cc:defs.bzl", "cc_test")

# sylvan_core 的单元测试
cc_test(
    name = "autograd_test",
    srcs = ["autograd_test.cc"],
    deps = [
        "//sylvan/core", # 依赖 sylvan_core 库
        "@googletest//:gtest_main", # Bzlmod 方式引用 gtest_main
    ],
)

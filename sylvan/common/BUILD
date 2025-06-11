load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "common",
    hdrs = glob(["include/sylvan/common/**/*.h"]),
    strip_include_prefix = "include",
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [],
)

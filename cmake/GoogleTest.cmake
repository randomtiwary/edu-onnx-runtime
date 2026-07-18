# LEARNER: FetchContent downloads GoogleTest at configure time and builds it
# as part of our project. That avoids "apt install libgtest-dev" differences
# across machines. First configure may be slower (git clone).
#
# Pin a release tag so builds are reproducible (design K8 testing stack).

include(FetchContent)

# Keep GoogleTest from installing itself when someone runs cmake --install later.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)

# For MSVC: match runtime; harmless on Linux.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# LEARNER: Do not apply our -Werror / pedantic flags to gtest itself.
# Our tests still get eduort_set_project_warnings.

add_executable(bench_rolling micro/bench_rolling.cpp)
target_compile_features(bench_rolling PRIVATE cxx_std_20)

add_executable(run_stac_a2 stac/run_stac_a2.cpp tick/TickDataReplay.cpp)
target_compile_features(run_stac_a2 PRIVATE cxx_std_20)
target_link_libraries(run_stac_a2 pthread)

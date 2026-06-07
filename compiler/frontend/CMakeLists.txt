add_library(qfxFrontend STATIC
  Token.cpp
  Lexer.cpp
  Parser.cpp
  HIR.cpp
)

target_include_directories(qfxFrontend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(qfxFrontend PUBLIC cxx_std_20)

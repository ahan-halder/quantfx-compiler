#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qfx {

struct Type {
  std::string element = "f32";
  int64_t length = 0;
};

struct NamedArg {
  std::string name;
  bool isFloat = false;
  int64_t intValue = 0;
  double floatValue = 0.0;
};

struct Expr {
  enum class Kind { Variable, Call } kind = Kind::Variable;
  std::string name;
  std::vector<std::string> positionalArgs;
  std::vector<NamedArg> namedArgs;
};

struct ImportStmt {
  std::string name;
  Type type;
};

struct LetStmt {
  std::string name;
  Expr expr;
};

struct EmitStmt {
  std::string name;
  Type type;
};

struct Module {
  std::vector<ImportStmt> imports;
  std::vector<LetStmt> lets;
  std::vector<EmitStmt> emits;
};

} // namespace qfx

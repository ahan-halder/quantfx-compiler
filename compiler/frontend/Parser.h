#pragma once

#include "AST.h"
#include "Lexer.h"

#include <memory>
#include <string>

namespace qfx {

class Parser {
public:
  explicit Parser(Lexer &lexer);

  std::unique_ptr<Module> parseModule(std::string &error);

private:
  ImportStmt parseImport();
  LetStmt parseLet();
  EmitStmt parseEmit();
  Type parseType();
  Expr parseExpr();
  Expr parseCall(const std::string &callee);

  Token expect(TokenKind kind, const char *message);
  bool check(TokenKind kind) const;

  Lexer &lexer_;
};

} // namespace qfx

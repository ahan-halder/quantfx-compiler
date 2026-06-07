#pragma once

#include "Token.h"

#include <optional>
#include <string>
#include <string_view>

namespace qfx {

class Lexer {
public:
  explicit Lexer(std::string source);

  const Token &peek() const { return current_; }
  Token consume();
  bool match(TokenKind kind);

private:
  void advance();
  void skipWhitespaceAndComments();
  Token lexIdentifierOrKeyword();
  Token lexNumber();
  Token makeToken(TokenKind kind, std::string_view text);

  std::string source_;
  std::size_t pos_ = 0;
  int line_ = 1;
  int column_ = 1;
  Token current_;
};

} // namespace qfx

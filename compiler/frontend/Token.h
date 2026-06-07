#pragma once

#include <string>
#include <string_view>

namespace qfx {

enum class TokenKind {
  Eof,
  Error,
  Newline,
  Import,
  Stream,
  Let,
  Emit,
  F32,
  Ident,
  Int,
  Float,
  Colon,
  Eq,
  Comma,
  LParen,
  RParen,
  LBracket,
  RBracket,
};

struct Token {
  TokenKind kind = TokenKind::Error;
  std::string text;
  int line = 1;
  int column = 1;
};

const char *tokenKindName(TokenKind kind);

} // namespace qfx

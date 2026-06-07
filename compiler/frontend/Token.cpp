#include "Token.h"

namespace qfx {

const char *tokenKindName(TokenKind kind) {
  switch (kind) {
  case TokenKind::Eof:
    return "EOF";
  case TokenKind::Error:
    return "error";
  case TokenKind::Newline:
    return "newline";
  case TokenKind::Import:
    return "import";
  case TokenKind::Stream:
    return "stream";
  case TokenKind::Let:
    return "let";
  case TokenKind::Emit:
    return "emit";
  case TokenKind::F32:
    return "f32";
  case TokenKind::Ident:
    return "identifier";
  case TokenKind::Int:
    return "integer";
  case TokenKind::Float:
    return "float";
  case TokenKind::Colon:
    return ":";
  case TokenKind::Eq:
    return "=";
  case TokenKind::Comma:
    return ",";
  case TokenKind::LParen:
    return "(";
  case TokenKind::RParen:
    return ")";
  case TokenKind::LBracket:
    return "[";
  case TokenKind::RBracket:
    return "]";
  }
  return "unknown";
}

} // namespace qfx

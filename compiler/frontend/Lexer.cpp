#include "Lexer.h"

#include <cctype>

namespace qfx {

Lexer::Lexer(std::string source) : source_(std::move(source)) { advance(); }

Token Lexer::consume() {
  Token token = current_;
  if (token.kind != TokenKind::Eof && token.kind != TokenKind::Error)
    advance();
  return token;
}

bool Lexer::match(TokenKind kind) {
  if (current_.kind != kind)
    return false;
  consume();
  return true;
}

void Lexer::advance() {
  skipWhitespaceAndComments();
  if (pos_ >= source_.size()) {
    current_ = makeToken(TokenKind::Eof, "");
    return;
  }

  const char c = source_[pos_];
  if (c == '\n') {
    ++pos_;
    ++line_;
    column_ = 1;
    current_ = makeToken(TokenKind::Newline, "\n");
    return;
  }

  if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
    current_ = lexIdentifierOrKeyword();
    return;
  }

  if (std::isdigit(static_cast<unsigned char>(c))) {
    current_ = lexNumber();
    return;
  }

  if (c == '.' && pos_ + 1 < source_.size() &&
      std::isdigit(static_cast<unsigned char>(source_[pos_ + 1]))) {
    current_ = lexNumber();
    return;
  }

  ++pos_;
  ++column_;
  switch (c) {
  case ':':
    current_ = makeToken(TokenKind::Colon, ":");
    return;
  case '=':
    current_ = makeToken(TokenKind::Eq, "=");
    return;
  case ',':
    current_ = makeToken(TokenKind::Comma, ",");
    return;
  case '(':
    current_ = makeToken(TokenKind::LParen, "(");
    return;
  case ')':
    current_ = makeToken(TokenKind::RParen, ")");
    return;
  case '[':
    current_ = makeToken(TokenKind::LBracket, "[");
    return;
  case ']':
    current_ = makeToken(TokenKind::RBracket, "]");
    return;
  default:
    current_ = makeToken(TokenKind::Error, std::string(1, c));
    return;
  }
}

void Lexer::skipWhitespaceAndComments() {
  while (pos_ < source_.size()) {
    const char c = source_[pos_];
    if (c == ' ' || c == '\t' || c == '\r') {
      ++pos_;
      ++column_;
      continue;
    }
    if (c == '#') {
      while (pos_ < source_.size() && source_[pos_] != '\n') {
        ++pos_;
        ++column_;
      }
      continue;
    }
    break;
  }
}

Token Lexer::lexIdentifierOrKeyword() {
  const std::size_t start = pos_;
  const int startColumn = column_;
  while (pos_ < source_.size()) {
    const char c = source_[pos_];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      break;
    ++pos_;
    ++column_;
  }

  std::string text = source_.substr(start, pos_ - start);
  TokenKind kind = TokenKind::Ident;
  if (text == "import")
    kind = TokenKind::Import;
  else if (text == "stream")
    kind = TokenKind::Stream;
  else if (text == "let")
    kind = TokenKind::Let;
  else if (text == "emit")
    kind = TokenKind::Emit;
  else if (text == "f32")
    kind = TokenKind::F32;

  Token token;
  token.kind = kind;
  token.text = std::move(text);
  token.line = line_;
  token.column = startColumn;
  return token;
}

Token Lexer::lexNumber() {
  const std::size_t start = pos_;
  const int startColumn = column_;
  bool isFloat = false;
  while (pos_ < source_.size()) {
    const char ch = source_[pos_];
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      ++pos_;
      ++column_;
      continue;
    }
    if (ch == '.' && !isFloat) {
      isFloat = true;
      ++pos_;
      ++column_;
      continue;
    }
    break;
  }

  Token token;
  token.kind = isFloat ? TokenKind::Float : TokenKind::Int;
  token.text = source_.substr(start, pos_ - start);
  token.line = line_;
  token.column = startColumn;
  return token;
}

Token Lexer::makeToken(TokenKind kind, std::string_view text) {
  Token token;
  token.kind = kind;
  token.text = std::string(text);
  token.line = line_;
  token.column = column_;
  if (!text.empty() && kind != TokenKind::Newline) {
    pos_ += text.size() - 1;
    column_ += static_cast<int>(text.size()) - 1;
  }
  return token;
}

} // namespace qfx

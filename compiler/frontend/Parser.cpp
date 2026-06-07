#include "Parser.h"

#include <sstream>
#include <stdexcept>

namespace qfx {

Parser::Parser(Lexer &lexer) : lexer_(lexer) {}

std::unique_ptr<Module> Parser::parseModule(std::string &error) {
  auto module = std::make_unique<Module>();

  while (lexer_.peek().kind != TokenKind::Eof) {
    if (lexer_.peek().kind == TokenKind::Newline) {
      lexer_.consume();
      continue;
    }

    switch (lexer_.peek().kind) {
    case TokenKind::Import:
      module->imports.push_back(parseImport());
      break;
    case TokenKind::Let:
      module->lets.push_back(parseLet());
      break;
    case TokenKind::Emit:
      module->emits.push_back(parseEmit());
      break;
    default: {
      std::ostringstream oss;
      oss << "unexpected token '" << lexer_.peek().text << "' at line "
          << lexer_.peek().line;
      error = oss.str();
      return nullptr;
    }
    }
  }

  return module;
}

ImportStmt Parser::parseImport() {
  expect(TokenKind::Import, "expected 'import'");
  expect(TokenKind::Stream, "expected 'stream'");
  Token name = expect(TokenKind::Ident, "expected stream name");
  expect(TokenKind::Colon, "expected ':'");
  Type type = parseType();
  return ImportStmt{name.text, type};
}

LetStmt Parser::parseLet() {
  expect(TokenKind::Let, "expected 'let'");
  Token name = expect(TokenKind::Ident, "expected binding name");
  expect(TokenKind::Eq, "expected '='");
  Expr expr = parseExpr();
  return LetStmt{name.text, expr};
}

EmitStmt Parser::parseEmit() {
  expect(TokenKind::Emit, "expected 'emit'");
  Token name = expect(TokenKind::Ident, "expected value name");
  expect(TokenKind::Colon, "expected ':'");
  Type type = parseType();
  return EmitStmt{name.text, type};
}

Type Parser::parseType() {
  expect(TokenKind::F32, "expected 'f32'");
  expect(TokenKind::LBracket, "expected '['");
  Token length = expect(TokenKind::Int, "expected array length");
  expect(TokenKind::RBracket, "expected ']'");
  Type type;
  type.element = "f32";
  type.length = std::stoll(length.text);
  return type;
}

Expr Parser::parseExpr() {
  Token token = expect(TokenKind::Ident, "expected identifier");
  if (lexer_.peek().kind == TokenKind::LParen)
    return parseCall(token.text);

  Expr expr;
  expr.kind = Expr::Kind::Variable;
  expr.name = token.text;
  return expr;
}

Expr Parser::parseCall(const std::string &callee) {
  Expr expr;
  expr.kind = Expr::Kind::Call;
  expr.name = callee;
  expect(TokenKind::LParen, "expected '('");

  if (lexer_.peek().kind != TokenKind::RParen) {
    while (true) {
      if (lexer_.peek().kind == TokenKind::Ident) {
        Token arg = lexer_.consume();
        if (lexer_.peek().kind == TokenKind::Eq) {
          lexer_.consume();
          Token value = lexer_.consume();
          if (value.kind != TokenKind::Int && value.kind != TokenKind::Float) {
            throw std::runtime_error("expected numeric argument");
          }
          NamedArg named;
          named.name = arg.text;
          if (value.kind == TokenKind::Float) {
            named.isFloat = true;
            named.floatValue = std::stod(value.text);
          } else {
            named.intValue = std::stoll(value.text);
          }
          expr.namedArgs.push_back(std::move(named));
        } else {
          expr.positionalArgs.push_back(arg.text);
        }
      } else {
        break;
      }

      if (lexer_.peek().kind != TokenKind::Comma)
        break;
      lexer_.consume();
    }
  }

  expect(TokenKind::RParen, "expected ')'");
  return expr;
}

Token Parser::expect(TokenKind kind, const char *message) {
  if (lexer_.peek().kind != kind) {
    std::ostringstream oss;
    oss << message << " (got " << tokenKindName(lexer_.peek().kind) << " '"
        << lexer_.peek().text << "' at line " << lexer_.peek().line << ")";
    throw std::runtime_error(oss.str());
  }
  return lexer_.consume();
}

bool Parser::check(TokenKind kind) const {
  return lexer_.peek().kind == kind;
}

} // namespace qfx

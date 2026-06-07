#ifndef QFX_DIALECT
#define QFX_DIALECT

include "mlir/IR/DialectBase.td"

def QFX_Dialect : Dialect {
  let name = "qfx";
  let summary = "QuantFX time-series analytics dialect";
  let description = [{
    The `qfx` dialect models streaming financial time-series operations
    such as rolling statistics, VWAP, and z-score signals.
  }];
  let cppNamespace = "::qfx";
}

class QFX_Op<string mnemonic, list<Trait> traits = []> :
    Op<QFX_Dialect, mnemonic, traits>;

#endif // QFX_DIALECT

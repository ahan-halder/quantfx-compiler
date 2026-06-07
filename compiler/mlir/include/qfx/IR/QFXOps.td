#ifndef QFX_OPS
#define QFX_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "qfx/IR/QFXDialect.td"

def QFX_RollingMeanOp : QFX_Op<"rolling_mean", [Pure]> {
  let summary = "Centered rolling mean (NumPy convolve same semantics)";
  let arguments = (ins AnyMemRef:$input, I64Attr:$window);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` `window` `=` $window attr-dict `:` type($input) `->` type($output)";
}

def QFX_RollingStdOp : QFX_Op<"rolling_std", [Pure]> {
  let summary = "Centered rolling standard deviation";
  let arguments = (ins AnyMemRef:$input, I64Attr:$window);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` `window` `=` $window attr-dict `:` type($input) `->` type($output)";
}

def QFX_RollingVwapOp : QFX_Op<"rolling_vwap", [Pure]> {
  let summary = "Centered rolling volume-weighted average price";
  let arguments = (ins AnyMemRef:$prices, AnyMemRef:$volumes, I64Attr:$window);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$prices `,` $volumes `,` `window` `=` $window attr-dict `:` type($prices) `,` type($volumes) `->` type($output)";
}

def QFX_RollingCovOp : QFX_Op<"rolling_cov", [Pure]> {
  let summary = "Centered rolling covariance";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, I64Attr:$window);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` `window` `=` $window attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def QFX_ZScoreOp : QFX_Op<"zscore", [Pure]> {
  let summary = "Element-wise z-score (x - mu) / sigma";
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$mu, AnyMemRef:$sigma);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` $mu `,` $sigma attr-dict `:` type($input) `,` type($mu) `,` type($sigma) `->` type($output)";
}

def QFX_EmaOp : QFX_Op<"ema", [Pure]> {
  let summary = "Exponential moving average";
  let arguments = (ins AnyMemRef:$input, F64Attr:$alpha);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` `alpha` `=` $alpha attr-dict `:` type($input) `->` type($output)";
}

def QFX_CrossoverOp : QFX_Op<"crossover", [Pure]> {
  let summary = "Signal when lhs crosses above rhs";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def QFX_EmitOp : QFX_Op<"emit", [MemoryEffects<[MemWrite]>]> {
  let summary = "Copy a computed series into an output buffer";
  let arguments = (ins AnyMemRef:$value, AnyMemRef:$output);
  let assemblyFormat = "$value `into` $output attr-dict `:` type($value) `,` type($output)";
}

#endif // QFX_OPS

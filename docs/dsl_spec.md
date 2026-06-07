# QFX DSL Specification (v0.1)

## Overview

`.qfx` is a statically-typed DSL for streaming financial time-series analytics.
Programs declare input streams, compute bindings, and emit output series.

## Syntax

```qfx
import stream <name> : f32[<length>]
let <name> = <expr>
emit <name> : f32[<length>]
```

Comments begin with `#`.

## Types

| Type | Description |
|------|-------------|
| `f32[N]` | Fixed-length float32 series |

## Operations

| Operation | Arguments | Semantics |
|-----------|-----------|-----------|
| `rolling_mean(x, window=W)` | stream, window | Centered rolling mean (NumPy `convolve` same) |
| `rolling_std(x, window=W)` | stream, window | Centered rolling standard deviation |
| `rolling_vwap(p, v, window=W)` | price, volume, window | Rolling VWAP |
| `rolling_cov(x, y, window=W)` | stream, stream, window | Rolling covariance |
| `zscore(x, mu, sigma)` | stream, stream, stream | `(x - mu) / sigma` |
| `ema(x, alpha=A)` | stream, alpha | Exponential moving average |
| `crossover(a, b)` | stream, stream | 1.0 when `a` crosses above `b` |

Named arguments use `name=value` syntax. `window` accepts integers; `alpha` accepts floats.

## Example

```qfx
import stream prices : f32[1024]
import stream volumes : f32[1024]

let vwap = rolling_vwap(prices, volumes, window=20)
let sigma = rolling_std(prices, window=20)
let signal = zscore(prices, vwap, sigma)

emit signal : f32[1024]
```

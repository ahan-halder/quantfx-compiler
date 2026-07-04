#include "qfx/Conversion/QFXToAffine.h"
#include "qfx/IR/QFXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace qfx {
namespace {

static mlir::Value load(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value memref, mlir::Value index) {
  return builder.create<mlir::memref::LoadOp>(loc, memref, mlir::ValueRange{index});
}

static void store(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
                  mlir::Value memref, mlir::Value index) {
  builder.create<mlir::memref::StoreOp>(loc, value, memref, mlir::ValueRange{index});
}

static mlir::Value allocLike(mlir::PatternRewriter &rewriter, mlir::Location loc,
                             mlir::MemRefType type) {
  return rewriter.create<mlir::memref::AllocOp>(loc, type);
}

static bool useIncremental(mlir::UnitAttr attr) { return static_cast<bool>(attr); }

static mlir::Value buildTrailingMean(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                     mlir::Value input, int64_t window) {
  auto type = llvm::cast<mlir::MemRefType>(input.getType());
  const int64_t n = type.getDimSize(0);
  auto f32 = rewriter.getF32Type();
  mlir::Value output = allocLike(rewriter, loc, type);

  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
  mlir::Value wCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, window);
  mlir::Value zeroF = rewriter.create<mlir::arith::ConstantOp>(
      loc, mlir::FloatAttr::get(f32, 0.0));

  auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one, zeroF);
  rewriter.setInsertionPointToStart(loop.getBody());
  mlir::Value k = loop.getInductionVar();
  mlir::Value runningSum = loop.getRegionIterArgs()[0];

  mlir::Value sample = load(rewriter, loc, input, k);
  mlir::Value nextSum = rewriter.create<mlir::arith::AddFOp>(loc, runningSum, sample);

  auto kGeW = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, k, wCst);
  auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, kGeW,
                                               /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::Value oldIdx = rewriter.create<mlir::arith::SubIOp>(loc, k, wCst);
  mlir::Value oldSample = load(rewriter, loc, input, oldIdx);
  mlir::Value subbed = rewriter.create<mlir::arith::SubFOp>(loc, nextSum, oldSample);
  rewriter.create<mlir::scf::YieldOp>(loc, subbed);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  rewriter.create<mlir::scf::YieldOp>(loc, nextSum);
  rewriter.setInsertionPointAfter(ifOp);

  mlir::Value countF = rewriter.create<mlir::arith::UIToFPOp>(
      loc, f32, rewriter.create<mlir::arith::AddIOp>(loc, k, one));
  auto kGeWCount = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, k,
                                                        rewriter.create<mlir::arith::SubIOp>(
                                                            loc, wCst, one));
  mlir::Value wF = rewriter.create<mlir::arith::UIToFPOp>(loc, f32, wCst);
  mlir::Value denom =
      rewriter.create<mlir::arith::SelectOp>(loc, kGeWCount, wF, countF);
  mlir::Value mean = rewriter.create<mlir::arith::DivFOp>(loc, ifOp.getResult(0), denom);
  store(rewriter, loc, mean, output, k);
  rewriter.create<mlir::scf::YieldOp>(loc, ifOp.getResult(0));
  rewriter.setInsertionPointAfter(loop);
  return output;
}

static mlir::Value buildConvolveSame(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                     mlir::Value input, int64_t window) {
  auto type = llvm::cast<mlir::MemRefType>(input.getType());
  const int64_t n = type.getDimSize(0);
  auto f32 = rewriter.getF32Type();
  mlir::Value output = allocLike(rewriter, loc, type);

  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
  mlir::Value wCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, window);
  mlir::Value invW = rewriter.create<mlir::arith::ConstantOp>(
      loc, mlir::FloatAttr::get(f32, 1.0 / static_cast<double>(window)));
  mlir::Value zeroF = rewriter.create<mlir::arith::ConstantOp>(
      loc, mlir::FloatAttr::get(f32, 0.0));

  auto kLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
  rewriter.setInsertionPointToStart(kLoop.getBody());
  mlir::Value k = kLoop.getInductionVar();
  mlir::Value sum = zeroF;

  auto iLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, wCst, one, sum);
  rewriter.setInsertionPointToStart(iLoop.getBody());
  mlir::Value i = iLoop.getInductionVar();
  mlir::Value iterSum = iLoop.getRegionIterArgs()[0];

  mlir::Value padCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, (window - 1) / 2);
  mlir::Value kPlusPad = rewriter.create<mlir::arith::AddIOp>(loc, k, padCst);
  mlir::Value j = rewriter.create<mlir::arith::SubIOp>(loc, kPlusPad, i);
  auto jGe0 = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, j,
                                                   zero);
  auto jLtN = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, j,
                                                   nCst);
  auto inBounds = rewriter.create<mlir::arith::AndIOp>(loc, jGe0, jLtN);

  auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, inBounds,
                                               /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::Value sample = load(rewriter, loc, input, j);
  mlir::Value contrib = rewriter.create<mlir::arith::MulFOp>(loc, sample, invW);
  rewriter.create<mlir::scf::YieldOp>(loc, contrib);

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  rewriter.create<mlir::scf::YieldOp>(loc, zeroF);

  rewriter.setInsertionPointAfter(ifOp);
  mlir::Value nextSum =
      rewriter.create<mlir::arith::AddFOp>(loc, iterSum, ifOp.getResult(0));
  rewriter.create<mlir::scf::YieldOp>(loc, nextSum);

  rewriter.setInsertionPointAfter(iLoop);
  store(rewriter, loc, iLoop.getResult(0), output, k);
  rewriter.setInsertionPointAfter(kLoop);

  return output;
}

static mlir::Value buildElementwiseBinary(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::Value,
                                   mlir::Value)>
        op) {
  auto type = llvm::cast<mlir::MemRefType>(lhs.getType());
  const int64_t n = type.getDimSize(0);
  mlir::Value output = allocLike(rewriter, loc, type);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);

  auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
  rewriter.setInsertionPointToStart(loop.getBody());
  mlir::Value idx = loop.getInductionVar();
  mlir::Value a = load(rewriter, loc, lhs, idx);
  mlir::Value b = load(rewriter, loc, rhs, idx);
  mlir::Value result = op(rewriter, loc, a, b);
  store(rewriter, loc, result, output, idx);
  rewriter.setInsertionPointAfter(loop);
  return output;
}

static mlir::Value buildElementwiseUnary(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value input,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::Value)>
        op) {
  return buildElementwiseBinary(
      rewriter, loc, input, input,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value a, mlir::Value) {
        return op(b, l, a);
      });
}

struct RollingMeanLowering : mlir::OpRewritePattern<RollingMeanOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingMeanOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Value result = useIncremental(op.getIncrementalAttr())
                             ? buildTrailingMean(rewriter, op.getLoc(), op.getInput(),
                                                 op.getWindow())
                             : buildConvolveSame(rewriter, op.getLoc(), op.getInput(),
                                                 op.getWindow());
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RollingStdLowering : mlir::OpRewritePattern<RollingStdOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingStdOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t window = op.getWindow();
    mlir::Value mean = buildConvolveSame(rewriter, loc, op.getInput(), window);

    auto square = buildElementwiseUnary(rewriter, loc, op.getInput(),
                                        [&](mlir::OpBuilder &b, mlir::Location l,
                                            mlir::Value v) {
                                          return b.create<mlir::arith::MulFOp>(loc, v, v);
                                        });
    mlir::Value meanSq = buildConvolveSame(rewriter, loc, square, window);

    auto f32 = rewriter.getF32Type();
    mlir::Value variance = buildElementwiseBinary(
        rewriter, loc, meanSq, mean, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value msq,
                                          mlir::Value mu) {
          mlir::Value sq = b.create<mlir::arith::MulFOp>(l, mu, mu);
          mlir::Value var = b.create<mlir::arith::SubFOp>(l, msq, sq);
          mlir::Value zero = b.create<mlir::arith::ConstantOp>(
              l, mlir::FloatAttr::get(f32, 0.0));
          auto pred = b.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, var,
                                                    zero);
          return b.create<mlir::arith::SelectOp>(l, pred, zero, var);
        });

    mlir::Value stddev = buildElementwiseUnary(
        rewriter, loc, variance, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value v) {
          return b.create<mlir::math::SqrtOp>(l, v);
        });
    rewriter.replaceOp(op, stddev);
    return mlir::success();
  }
};

struct RollingVwapLowering : mlir::OpRewritePattern<RollingVwapOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingVwapOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t window = op.getWindow();
    auto weighted = buildElementwiseBinary(
        rewriter, loc, op.getPrices(), op.getVolumes(),
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value p, mlir::Value v) {
          return b.create<mlir::arith::MulFOp>(l, p, v);
        });
    mlir::Value num = buildConvolveSame(rewriter, loc, weighted, window);
    mlir::Value den = buildConvolveSame(rewriter, loc, op.getVolumes(), window);

    auto f32 = rewriter.getF32Type();
    mlir::Value result = buildElementwiseBinary(
        rewriter, loc, num, den, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value n,
                                     mlir::Value d) {
          mlir::Value eps = b.create<mlir::arith::ConstantOp>(
              l, mlir::FloatAttr::get(f32, 1e-8));
          auto pred = b.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, d, eps);
          auto safeDen = b.create<mlir::arith::SelectOp>(l, pred, eps, d);
          return b.create<mlir::arith::DivFOp>(l, n, safeDen);
        });
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RollingCovLowering : mlir::OpRewritePattern<RollingCovOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(RollingCovOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t window = op.getWindow();
    mlir::Value meanX = buildConvolveSame(rewriter, loc, op.getLhs(), window);
    mlir::Value meanY = buildConvolveSame(rewriter, loc, op.getRhs(), window);
    auto product = buildElementwiseBinary(
        rewriter, loc, op.getLhs(), op.getRhs(),
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value x, mlir::Value y) {
          return b.create<mlir::arith::MulFOp>(l, x, y);
        });
    mlir::Value meanXY = buildConvolveSame(rewriter, loc, product, window);

    auto type = llvm::cast<mlir::MemRefType>(meanXY.getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value output = allocLike(rewriter, loc, type);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
    auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value mxy = load(rewriter, loc, meanXY, idx);
    mlir::Value mx = load(rewriter, loc, meanX, idx);
    mlir::Value my = load(rewriter, loc, meanY, idx);
    mlir::Value cov =
        rewriter.create<mlir::arith::SubFOp>(loc, mxy,
                                             rewriter.create<mlir::arith::MulFOp>(loc, mx, my));
    store(rewriter, loc, cov, output, idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct ZScoreLowering : mlir::OpRewritePattern<ZScoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ZScoreOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto f32 = rewriter.getF32Type();
    auto type = llvm::cast<mlir::MemRefType>(op.getInput().getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value output = allocLike(rewriter, loc, type);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
    mlir::Value eps = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(f32, 1e-8));

    auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value x = load(rewriter, loc, op.getInput(), idx);
    mlir::Value mu = load(rewriter, loc, op.getMu(), idx);
    mlir::Value sigma = load(rewriter, loc, op.getSigma(), idx);
    mlir::Value diff = rewriter.create<mlir::arith::SubFOp>(loc, x, mu);
    auto pred = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, sigma,
                                                     eps);
    mlir::Value safeSigma = rewriter.create<mlir::arith::SelectOp>(loc, pred, eps, sigma);
    mlir::Value z = rewriter.create<mlir::arith::DivFOp>(loc, diff, safeSigma);
    store(rewriter, loc, z, output, idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct EmaLowering : mlir::OpRewritePattern<EmaOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(EmaOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto f32 = rewriter.getF32Type();
    auto type = llvm::cast<mlir::MemRefType>(op.getInput().getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value output = allocLike(rewriter, loc, type);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
    mlir::Value alpha = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(f32, op.getAlpha()));
    mlir::Value oneMinusAlpha = rewriter.create<mlir::arith::SubFOp>(
        loc, rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(f32, 1.0)),
        alpha);

    mlir::Value x0 = load(rewriter, loc, op.getInput(), zero);
    store(rewriter, loc, x0, output, zero);

    auto loop = rewriter.create<mlir::scf::ForOp>(loc, one, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value prevIdx = rewriter.create<mlir::arith::SubIOp>(loc, idx, one);
    mlir::Value x = load(rewriter, loc, op.getInput(), idx);
    mlir::Value prev = load(rewriter, loc, output, prevIdx);
    mlir::Value termA = rewriter.create<mlir::arith::MulFOp>(loc, alpha, x);
    mlir::Value termB = rewriter.create<mlir::arith::MulFOp>(loc, oneMinusAlpha, prev);
    mlir::Value ema = rewriter.create<mlir::arith::AddFOp>(loc, termA, termB);
    store(rewriter, loc, ema, output, idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct CrossoverLowering : mlir::OpRewritePattern<CrossoverOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(CrossoverOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto f32 = rewriter.getF32Type();
    auto type = llvm::cast<mlir::MemRefType>(op.getLhs().getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value output = allocLike(rewriter, loc, type);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
    mlir::Value zeroF = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(f32, 0.0));
    mlir::Value oneF = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(f32, 1.0));

    store(rewriter, loc, zeroF, output, zero);
    auto loop = rewriter.create<mlir::scf::ForOp>(loc, one, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value prevIdx = rewriter.create<mlir::arith::SubIOp>(loc, idx, one);
    mlir::Value lhs = load(rewriter, loc, op.getLhs(), idx);
    mlir::Value rhs = load(rewriter, loc, op.getRhs(), idx);
    mlir::Value lhsPrev = load(rewriter, loc, op.getLhs(), prevIdx);
    mlir::Value rhsPrev = load(rewriter, loc, op.getRhs(), prevIdx);
    auto aboveNow =
        rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
    auto notAboveBefore = rewriter.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLE, lhsPrev, rhsPrev);
    auto crossed = rewriter.create<mlir::arith::AndIOp>(loc, aboveNow, notAboveBefore);
    mlir::Value signal = rewriter.create<mlir::arith::SelectOp>(loc, crossed, oneF, zeroF);
    store(rewriter, loc, signal, output, idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct FusedRollingStatsLowering : mlir::OpRewritePattern<FusedRollingStatsOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(FusedRollingStatsOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t window = op.getWindow();
    auto type = llvm::cast<mlir::MemRefType>(op.getInput().getType());
    const int64_t n = type.getDimSize(0);
    auto f32 = rewriter.getF32Type();
    if (useIncremental(op.getIncrementalAttr())) {
      mlir::Value mean = buildTrailingMean(rewriter, loc, op.getInput(), window);
      auto square = buildElementwiseUnary(rewriter, loc, op.getInput(),
                                          [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value v) {
                                            return b.create<mlir::arith::MulFOp>(l, v, v);
                                          });
      mlir::Value meanSq = buildTrailingMean(rewriter, loc, square, window);
      mlir::Value variance = buildElementwiseBinary(
          rewriter, loc, meanSq, mean, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value msq,
                                           mlir::Value mu) {
            mlir::Value sq = b.create<mlir::arith::MulFOp>(l, mu, mu);
            mlir::Value var = b.create<mlir::arith::SubFOp>(l, msq, sq);
            mlir::Value z = b.create<mlir::arith::ConstantOp>(l, mlir::FloatAttr::get(f32, 0.0));
            auto pred = b.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, var, z);
            return b.create<mlir::arith::SelectOp>(l, pred, z, var);
          });
      mlir::Value stddev = buildElementwiseUnary(
          rewriter, loc, variance, [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value v) {
            return b.create<mlir::math::SqrtOp>(l, v);
          });
      rewriter.replaceOp(op, mlir::ValueRange{mean, stddev});
      return mlir::success();
    }

    mlir::Value meanOut = allocLike(rewriter, loc, type);
    mlir::Value stdOut = allocLike(rewriter, loc, type);
    {
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
      mlir::Value wCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, window);
      mlir::Value invW = rewriter.create<mlir::arith::ConstantOp>(
          loc, mlir::FloatAttr::get(f32, 1.0 / static_cast<double>(window)));
      mlir::Value zeroF = rewriter.create<mlir::arith::ConstantOp>(
          loc, mlir::FloatAttr::get(f32, 0.0));

      auto kLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      mlir::Value k = kLoop.getInductionVar();
      mlir::Value sum = zeroF;
      mlir::Value sumSq = zeroF;

      auto iLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, wCst, one,
                                                     mlir::ValueRange{sum, sumSq});
      rewriter.setInsertionPointToStart(iLoop.getBody());
      mlir::Value i = iLoop.getInductionVar();
      mlir::Value iterSum = iLoop.getRegionIterArgs()[0];
      mlir::Value iterSumSq = iLoop.getRegionIterArgs()[1];
      mlir::Value padCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, (window - 1) / 2);
      mlir::Value kPlusPad = rewriter.create<mlir::arith::AddIOp>(loc, k, padCst);
      mlir::Value j = rewriter.create<mlir::arith::SubIOp>(loc, kPlusPad, i);
      auto jGe0 = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, j,
                                                       zero);
      auto jLtN = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, j,
                                                       nCst);
      auto inBounds = rewriter.create<mlir::arith::AndIOp>(loc, jGe0, jLtN);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32, f32}, inBounds,
                                                   /*withElseRegion=*/true);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      mlir::Value sample = load(rewriter, loc, op.getInput(), j);
      mlir::Value contrib = rewriter.create<mlir::arith::MulFOp>(loc, sample, invW);
      mlir::Value contribSq = rewriter.create<mlir::arith::MulFOp>(loc, contrib, sample);
      rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{contrib, contribSq});
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{zeroF, zeroF});
      rewriter.setInsertionPointAfter(ifOp);
      mlir::Value nextSum =
          rewriter.create<mlir::arith::AddFOp>(loc, iterSum, ifOp.getResult(0));
      mlir::Value nextSumSq =
          rewriter.create<mlir::arith::AddFOp>(loc, iterSumSq, ifOp.getResult(1));
      rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{nextSum, nextSumSq});
      rewriter.setInsertionPointAfter(iLoop);

      mlir::Value mean = iLoop.getResult(0);
      mlir::Value meanSq = iLoop.getResult(1);
      mlir::Value sqMean = rewriter.create<mlir::arith::MulFOp>(loc, mean, mean);
      mlir::Value var = rewriter.create<mlir::arith::SubFOp>(loc, meanSq, sqMean);
      auto pred = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, var,
                                                       zeroF);
      var = rewriter.create<mlir::arith::SelectOp>(loc, pred, zeroF, var);
      mlir::Value stddev = rewriter.create<mlir::math::SqrtOp>(loc, var);
      store(rewriter, loc, mean, meanOut, k);
      store(rewriter, loc, stddev, stdOut, k);
      rewriter.setInsertionPointAfter(kLoop);
    }

    rewriter.replaceOp(op, mlir::ValueRange{meanOut, stdOut});
    return mlir::success();
  }
};

struct FusedRollingCovLowering : mlir::OpRewritePattern<FusedRollingCovOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(FusedRollingCovOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int64_t window = op.getWindow();
    mlir::Value meanX = useIncremental(op.getIncrementalAttr())
                            ? buildTrailingMean(rewriter, loc, op.getLhs(), window)
                            : buildConvolveSame(rewriter, loc, op.getLhs(), window);
    mlir::Value meanY = useIncremental(op.getIncrementalAttr())
                            ? buildTrailingMean(rewriter, loc, op.getRhs(), window)
                            : buildConvolveSame(rewriter, loc, op.getRhs(), window);
    auto product = buildElementwiseBinary(
        rewriter, loc, op.getLhs(), op.getRhs(),
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value x, mlir::Value y) {
          return b.create<mlir::arith::MulFOp>(l, x, y);
        });
    mlir::Value meanXY = useIncremental(op.getIncrementalAttr())
                             ? buildTrailingMean(rewriter, loc, product, window)
                             : buildConvolveSame(rewriter, loc, product, window);

    auto type = llvm::cast<mlir::MemRefType>(meanXY.getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value output = allocLike(rewriter, loc, type);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);
    auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value mxy = load(rewriter, loc, meanXY, idx);
    mlir::Value mx = load(rewriter, loc, meanX, idx);
    mlir::Value my = load(rewriter, loc, meanY, idx);
    mlir::Value cov =
        rewriter.create<mlir::arith::SubFOp>(loc, mxy,
                                             rewriter.create<mlir::arith::MulFOp>(loc, mx, my));
    store(rewriter, loc, cov, output, idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct EmitLowering : mlir::OpRewritePattern<EmitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(EmitOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = llvm::cast<mlir::MemRefType>(op.getValue().getType());
    const int64_t n = type.getDimSize(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value nCst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, n);

    auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, nCst, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::Value idx = loop.getInductionVar();
    mlir::Value value = load(rewriter, loc, op.getValue(), idx);
    store(rewriter, loc, value, op.getOutput(), idx);
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct QFXToAffinePass : public mlir::PassWrapper<QFXToAffinePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QFXToAffinePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RollingMeanLowering, RollingStdLowering, RollingVwapLowering,
                 RollingCovLowering, FusedRollingStatsLowering, FusedRollingCovLowering,
                 ZScoreLowering, EmaLowering, CrossoverLowering, EmitLowering>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  llvm::StringRef getArgument() const final { return "qfx-to-affine"; }
  llvm::StringRef getDescription() const final {
    return "Lower qfx time-series ops to affine/scf loops";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createQFXToAffinePass() {
  return std::make_unique<QFXToAffinePass>();
}

} // namespace qfx

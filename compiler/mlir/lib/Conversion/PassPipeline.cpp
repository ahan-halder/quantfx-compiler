#include "qfx/Conversion/LowerToLLVM.h"
#include "qfx/Conversion/QFXToAffine.h"
#include "qfx/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace qfx {

static void addOptimizationPasses(mlir::OpPassManager &pm, OptLevel level, bool streaming) {
  if (level >= OptLevel::O1)
    pm.addPass(createWindowFusionPass());
  if (streaming)
    pm.addPass(createSlidingWindowUpdatePass());
  pm.addPass(createQFXToAffinePass());
  if (level >= OptLevel::O2) {
    pm.addPass(createLoopTilingPass(64));
    pm.addPass(createPrefetchInsertionPass(8));
  }
  if (level >= OptLevel::O3)
    pm.addPass(createVectorizeLoopsPass(16));
}

mlir::LogicalResult runCPUPipeline(mlir::ModuleOp module, mlir::MLIRContext &context,
                                   OptLevel level, bool streaming) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  mlir::PassManager pm(&context);
  addOptimizationPasses(pm, level, streaming);
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  if (level >= OptLevel::O3)
    pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createCFToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  return pm.run(module);
}

mlir::LogicalResult lowerModuleToLLVM(mlir::ModuleOp module, mlir::MLIRContext &context) {
  return runCPUPipeline(module, context, OptLevel::O2, /*streaming=*/false);
}

void registerQFXPasses() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
}

} // namespace qfx

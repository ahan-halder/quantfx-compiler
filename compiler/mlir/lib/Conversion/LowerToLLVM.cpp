#include "qfx/Conversion/LowerToLLVM.h"
#include "qfx/Conversion/QFXToAffine.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace qfx {

mlir::LogicalResult lowerModuleToLLVM(mlir::ModuleOp module,
                                      mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  mlir::PassManager pm(&context);
  pm.addPass(createQFXToAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createCFToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  return pm.run(module);
}

namespace {

mlir::LogicalResult emitLLVMIRToFile(mlir::ModuleOp module,
                                     mlir::MLIRContext &context,
                                     const std::string &path) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return mlir::failure();

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return mlir::failure();
  llvmModule->print(os, nullptr);
  return mlir::success();
}

mlir::LogicalResult emitObjectFile(mlir::ModuleOp module, mlir::MLIRContext &context,
                                   const std::string &path) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return mlir::failure();

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto triple = llvm::sys::getDefaultTargetTriple();
  llvmModule->setTargetTriple(triple);

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return mlir::failure();

  llvm::TargetMachine *machine = target->createTargetMachine(
      triple, "generic", "", llvm::TargetOptions(), std::nullopt);
  if (!machine)
    return mlir::failure();

  llvmModule->setDataLayout(machine->createDataLayout());
  std::error_code ec;
  llvm::raw_fd_ostream dest(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return mlir::failure();

  llvm::legacy::PassManager passManager;
  if (machine->addPassesToEmitFile(passManager, dest, nullptr,
                                   llvm::CGFT_ObjectFile)) {
    return mlir::failure();
  }
  passManager.run(*llvmModule);
  return mlir::success();
}

} // namespace

mlir::LogicalResult emitModuleLLVMIR(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path) {
  return emitLLVMIRToFile(module, context, path);
}

mlir::LogicalResult emitModuleObject(mlir::ModuleOp module, mlir::MLIRContext &context,
                                     const std::string &path) {
  return emitObjectFile(module, context, path);
}

} // namespace qfx

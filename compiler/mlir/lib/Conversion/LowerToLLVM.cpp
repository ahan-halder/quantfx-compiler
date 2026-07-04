#include "qfx/Conversion/LowerToLLVM.h"
#include "qfx/Transforms/Passes.h"

#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace qfx {

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
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
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
                                   llvm::CodeGenFileType::ObjectFile)) {
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

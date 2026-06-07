#include "HIR.h"
#include "Lexer.h"
#include "Parser.h"
#include "qfx/Conversion/HirToMLIR.h"
#include "GPUCodeGen.h"
#include "qfx/Conversion/LowerToLLVM.h"
#include "qfx/IR/QFXDialect.h"
#include "qfx/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace qfx;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input.qfx>"),
                                                 llvm::cl::Required);
static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output file"),
                                                 llvm::cl::value_desc("filename"));
static llvm::cl::opt<std::string> target("target", llvm::cl::desc("Compilation target"),
                                        llvm::cl::init("cpu"));
static llvm::cl::opt<bool> emitMlir("emit-mlir", llvm::cl::desc("Print MLIR and exit"));
static llvm::cl::opt<bool> emitLlvm("emit-llvm", llvm::cl::desc("Emit LLVM IR (.ll)"));
static llvm::cl::opt<bool> verifyRun("verify", llvm::cl::desc("JIT-run kernel on synthetic data"));
static llvm::cl::opt<int> optLevel("O", llvm::cl::desc("Optimization level (0-3)"), llvm::cl::init(2));
static llvm::cl::opt<bool> streamingMode("streaming",
                                          llvm::cl::desc("Trailing-window incremental semantics"));
static llvm::cl::opt<bool> emitPtx("emit-ptx", llvm::cl::desc("Emit PTX (CUDA target)"));

static std::string readFile(const std::string &path) {
  auto file = mlir::openInputFile(path);
  if (!file)
    return {};
  return (*file)->getBuffer().str();
}

static int compileQfx(const std::string &source, const std::string &outPath) {
  std::string error;
  Lexer lexer(source);
  Parser parser(lexer);
  std::unique_ptr<Module> ast;
  try {
    ast = parser.parseModule(error);
  } catch (const std::exception &ex) {
  llvm::errs() << "parse error: " << ex.what() << "\n";
    return 1;
  }
  if (!ast) {
    llvm::errs() << "parse error: " << error << "\n";
    return 1;
  }

  HirModule hir = lowerToHir(*ast, error);
  if (!error.empty()) {
    llvm::errs() << "HIR error: " << error << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect, qfx::QFXDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::ModuleOp module = lowerHirToMLIR(context, hir);
  if (!module) {
    llvm::errs() << "failed to lower HIR to MLIR\n";
    return 1;
  }

  if (emitMlir) {
    module.print(llvm::outs());
    return 0;
  }

  OptLevel level = OptLevel::O0;
  if (optLevel >= 3)
    level = OptLevel::O3;
  else if (optLevel == 2)
    level = OptLevel::O2;
  else if (optLevel == 1)
    level = OptLevel::O1;

  if (target == "cuda") {
    if (outPath.empty()) {
      llvm::errs() << "missing -o for CUDA output\n";
      return 1;
    }
    if (mlir::failed(emitCudaKernel(hir, outPath, emitPtx || outPath.ends_with(".ptx")))) {
      llvm::errs() << "CUDA codegen failed\n";
      return 1;
    }
    return 0;
  }

  if (mlir::failed(runCPUPipeline(module, context, level, streamingMode))) {
    llvm::errs() << "LLVM lowering failed\n";
    return 1;
  }

  if (emitLlvm) {
    if (outPath.empty()) {
      llvm::errs() << "missing -o for LLVM output\n";
      return 1;
    }
    if (mlir::failed(emitModuleLLVMIR(module, context, outPath))) {
      llvm::errs() << "failed to emit LLVM IR\n";
      return 1;
    }
    return 0;
  }

  if (verifyRun) {
    mlir::ExecutionEngineOptions options;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    auto engine = mlir::ExecutionEngine::create(module, options);
    if (!engine) {
      llvm::errs() << "failed to create execution engine\n";
      return 1;
    }

    const int64_t n = hir.imports.front().type.length;
    std::vector<float> prices(n), volumes(n);
    for (int64_t i = 0; i < n; ++i) {
      prices[i] = static_cast<float>(i % 97) * 0.01f - 0.5f;
      volumes[i] = 1.0f + static_cast<float>(i % 7) * 0.1f;
    }

    std::unordered_map<std::string, std::vector<float> *> streams = {
        {"prices", &prices},
        {"volumes", &volumes},
    };

    std::vector<std::vector<float>> outputs;
    outputs.reserve(hir.emits.size());
    for (size_t i = 0; i < hir.emits.size(); ++i)
      outputs.emplace_back(n, 0.0f);

    auto makeRef = [&](std::vector<float> &data) {
      StridedMemRefType<float, 1> ref;
      ref.basePtr = data.data();
      ref.data = data.data();
      ref.offset = 0;
      ref.sizes[0] = n;
      ref.strides[0] = 1;
      return ref;
    };

    std::vector<StridedMemRefType<float, 1>> importRefs;
    importRefs.reserve(hir.imports.size());
    for (auto &import : hir.imports) {
      auto it = streams.find(import.name);
      if (it == streams.end()) {
        llvm::errs() << "verify harness missing stream data for " << import.name << "\n";
        return 1;
      }
      importRefs.push_back(makeRef(*it->second));
    }

    std::vector<StridedMemRefType<float, 1>> outputRefs;
    outputRefs.reserve(outputs.size());
    for (auto &out : outputs)
      outputRefs.push_back(makeRef(out));

    std::vector<void *> args;
    for (auto &ref : importRefs)
      args.push_back(&ref);
    for (auto &ref : outputRefs)
      args.push_back(&ref);

    if (mlir::failed((*engine)->invoke("qfx_kernel", args))) {
      llvm::errs() << "kernel execution failed\n";
      return 1;
    }

    std::cout << "VERIFY " << n;
    for (float v : outputs.front())
      std::cout << ' ' << v;
    std::cout << '\n';
    return 0;
  }

  if (target != "cpu") {
    llvm::errs() << "unsupported target: " << target << " (use cpu or cuda)\n";
    return 1;
  }

  if (outPath.empty()) {
    llvm::errs() << "missing -o output path\n";
    return 1;
  }

  if (mlir::failed(emitModuleObject(module, context, outPath))) {
    llvm::errs() << "failed to emit object file\n";
    return 1;
  }

  llvm::outs() << "wrote " << outPath << "\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "quantfx compiler\n");

  std::string source = readFile(inputFilename);
  if (source.empty()) {
    llvm::errs() << "failed to read " << inputFilename << "\n";
    return 1;
  }

  return compileQfx(source, outputFilename);
}

// Coverage-only compiler settings. Instrumentation adds locals that push
// logDecision() (9 parameters) past the EVM stack limit ("Stack too deep");
// enabling the Yul optimizer for the instrumented build avoids it. The
// deployed artefact (scripts/compile.py) is not affected.
module.exports = {
  configureYulOptimizer: true,
};

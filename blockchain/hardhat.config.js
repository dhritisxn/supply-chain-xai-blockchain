// Hardhat is used only for the Solidity test suite and line coverage
// (solidity-coverage). Deployment still goes through scripts/deploy.py.
//   cd blockchain && npm ci
//   npx hardhat test        # test/*.test.js
//   npx hardhat coverage    # -> blockchain/coverage/, coverage.json
require("@nomicfoundation/hardhat-chai-matchers");
require("@nomicfoundation/hardhat-ethers");
require("solidity-coverage");

module.exports = {
  // Same settings as scripts/compile.py (and the committed ABI/bytecode).
  solidity: {
    version: "0.8.19",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./hardhat_cache",
    artifacts: "./hardhat_artifacts",
  },
};

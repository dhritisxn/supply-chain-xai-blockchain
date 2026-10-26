// Hardhat/mocha port of tests/test_contract.py, used for Solidity line
// coverage (`npx hardhat coverage` from blockchain/). The web3.py suite stays
// the primary one; this file mirrors it so the coverage figure reflects the
// same behaviours.
const { expect } = require("chai");
const { ethers } = require("hardhat");

const FP = 10n ** 18n;
const NO_ACTION = 0, AUTO = 1, HUMAN = 2;
const SHAP = "0x" + "11".repeat(32);
const fp = (x) => BigInt(Math.round(x * 1e6)) * 10n ** 12n;

async function deploy() {
  const signers = await ethers.getSigners();
  const ledger = await (await ethers.getContractFactory("ReplenishmentLedger")).deploy();
  await ledger.waitForDeployment();
  return { ledger, signers };
}

function log(ledger, signer, { status = AUTO, conf = 0.91, thr = 0.85, qty = 442.83 } = {}) {
  return ledger.connect(signer).logDecision(
    "SKU_A", "North", fp(qty), "xgboost", fp(183.23), fp(conf), fp(thr), SHAP, status);
}

async function logId(ledger, signer, opts) {
  const rcpt = await (await log(ledger, signer, opts)).wait();
  const ev = rcpt.logs.map((l) => ledger.interface.parseLog(l)).find((e) => e && e.name === "DecisionLogged");
  return { id: ev.args.decisionId, rcpt };
}

function events(ledger, rcpt, name) {
  return rcpt.logs.map((l) => ledger.interface.parseLog(l)).filter((e) => e && e.name === name);
}

describe("ReplenishmentLedger v2", function () {
  let ledger, owner, other, backend, attacker;

  beforeEach(async function () {
    const d = await deploy();
    ledger = d.ledger;
    [owner, other, backend, attacker] = d.signers;
  });

  describe("deployment", function () {
    it("makes the deployer owner and the only authorized caller", async function () {
      expect(await ledger.owner()).to.equal(owner.address);
      expect(await ledger.authorizedCallers(owner.address)).to.equal(true);
      expect(await ledger.authorizedCallers(other.address)).to.equal(false);
      expect(await ledger.minAutonomousThreshold()).to.equal(85n * 10n ** 16n);
    });
  });

  describe("logDecision", function () {
    it("stores the full record", async function () {
      const { id } = await logId(ledger, owner);
      expect(id).to.equal(1n);
      const rec = await ledger.getDecision(id);
      expect(rec.sku).to.equal("SKU_A");
      expect(rec.region).to.equal("North");
      expect(rec.orderQuantity).to.equal(fp(442.83));
      expect(rec.modelType).to.equal("xgboost");
      expect(rec.confidenceScore).to.equal(fp(0.91));
      expect(rec.shapHash).to.equal(SHAP);
      expect(rec.status).to.equal(BigInt(AUTO));
      expect(await ledger.getDecisionCount()).to.equal(1n);
      expect(await ledger.getAllDecisionIds()).to.deep.equal([1n]);
    });

    it("emits HumanApprovalRequired only for HUMAN_APPROVAL", async function () {
      const h = await logId(ledger, owner, { status: HUMAN, conf: 0.83 });
      const ev = events(ledger, h.rcpt, "HumanApprovalRequired");
      expect(ev).to.have.length(1);
      expect(ev[0].args.confidenceScore).to.equal(fp(0.83));
      for (const status of [NO_ACTION, AUTO]) {
        const r = await logId(ledger, owner, { status });
        expect(events(ledger, r.rcpt, "HumanApprovalRequired")).to.have.length(0);
      }
    });

    it("rejects an unauthorized caller", async function () {
      await expect(log(ledger, other)).to.be.revertedWith("Caller is not authorized");
      expect(await ledger.getDecisionCount()).to.equal(0n);
    });

    it("requires confidence >= threshold for AUTONOMOUS records (inclusive)", async function () {
      await expect(log(ledger, owner, { conf: 0.84, thr: 0.85 })).to.be.revertedWith("Confidence below threshold");
      await logId(ledger, owner, { conf: 0.85, thr: 0.85 });
      await logId(ledger, owner, { status: HUMAN, conf: 0.10, thr: 0.85 });
    });

    it("rejects an AUTONOMOUS threshold below the on-chain floor", async function () {
      await expect(log(ledger, owner, { conf: 0.60, thr: 0.50 })).to.be.revertedWith("Threshold below on-chain minimum");
    });

    it("rejects confidence above 1.0", async function () {
      await expect(log(ledger, owner, { status: HUMAN, conf: 1.5 })).to.be.revertedWith("Confidence above 1.0");
    });
  });

  describe("autoExecutePO", function () {
    it("creates a PO linked to its decision", async function () {
      const { id } = await logId(ledger, owner);
      const rcpt = await (await ledger.autoExecutePO(id)).wait();
      const ev = events(ledger, rcpt, "OrderExecuted")[0];
      const po = await ledger.getPO(ev.args.poId);
      expect(po.decisionId).to.equal(id);
      expect(po.sku).to.equal("SKU_A");
      expect(po.region).to.equal("North");
      expect(po.quantity).to.equal(fp(442.83));
      expect(po.status).to.equal(BigInt(AUTO));
      expect(await ledger.poForDecision(id)).to.equal(1n);
      expect(await ledger.getPOCount()).to.equal(1n);
      expect(await ledger.getAllPOIds()).to.deep.equal([1n]);
    });

    it("rejects an unauthorized caller", async function () {
      const { id } = await logId(ledger, owner);
      await expect(ledger.connect(other).autoExecutePO(id)).to.be.revertedWith("Caller is not authorized");
      expect(await ledger.getPOCount()).to.equal(0n);
    });

    it("executes at most one PO per decision", async function () {
      const { id } = await logId(ledger, owner);
      await ledger.autoExecutePO(id);
      await expect(ledger.autoExecutePO(id)).to.be.revertedWith("PO already executed for decision");
      expect(await ledger.getPOCount()).to.equal(1n);
    });

    for (const status of [NO_ACTION, HUMAN]) {
      it(`rejects a status-${status} decision`, async function () {
        const { id } = await logId(ledger, owner, { status, conf: 0.5 });
        await expect(ledger.autoExecutePO(id)).to.be.revertedWith("Decision was not marked AUTONOMOUS_EXECUTION");
      });
    }

    it("rejects a missing decision and a zero quantity", async function () {
      await expect(ledger.autoExecutePO(999)).to.be.revertedWith("Decision does not exist");
      const { id } = await logId(ledger, owner, { qty: 0 });
      await expect(ledger.autoExecutePO(id)).to.be.revertedWith("Order quantity is zero");
    });

    it("releases the reentrancy lock after each call", async function () {
      for (let i = 0; i < 3; i++) {
        const { id } = await logId(ledger, owner);
        await ledger.autoExecutePO(id);
      }
      expect(await ledger.getPOCount()).to.equal(3n);
    });
  });

  describe("administration", function () {
    it("lets the owner grant and revoke callers", async function () {
      await ledger.setAuthorizedCaller(backend.address, true);
      const { id } = await logId(ledger, backend);
      await ledger.connect(backend).autoExecutePO(id);
      await ledger.setAuthorizedCaller(backend.address, false);
      await expect(log(ledger, backend)).to.be.revertedWith("Caller is not authorized");
    });

    it("rejects administration by a non-owner", async function () {
      const a = ledger.connect(attacker);
      await expect(a.setAuthorizedCaller(attacker.address, true)).to.be.revertedWith("Caller is not the owner");
      await expect(a.setMinAutonomousThreshold(0)).to.be.revertedWith("Caller is not the owner");
      await expect(a.transferOwnership(attacker.address)).to.be.revertedWith("Caller is not the owner");
    });

    it("bounds and enforces the threshold floor", async function () {
      await expect(ledger.setMinAutonomousThreshold(FP + 1n)).to.be.revertedWith("Threshold above 1.0");
      await ledger.setMinAutonomousThreshold(90n * 10n ** 16n);
      await expect(log(ledger, owner, { conf: 0.95, thr: 0.85 })).to.be.revertedWith("Threshold below on-chain minimum");
    });

    it("transfers ownership", async function () {
      await ledger.transferOwnership(other.address);
      expect(await ledger.owner()).to.equal(other.address);
      await expect(ledger.setAuthorizedCaller(owner.address, false)).to.be.revertedWith("Caller is not the owner");
    });

    it("rejects the zero address", async function () {
      await expect(ledger.setAuthorizedCaller(ethers.ZeroAddress, true)).to.be.revertedWith("Zero address");
      await expect(ledger.transferOwnership(ethers.ZeroAddress)).to.be.revertedWith("Zero address");
    });
  });

  describe("getters", function () {
    it("revert for missing ids", async function () {
      await expect(ledger.getDecision(1)).to.be.revertedWith("Decision does not exist");
      await expect(ledger.getPO(1)).to.be.revertedWith("Purchase order does not exist");
    });

    it("expose the public mappings and arrays", async function () {
      const { id } = await logId(ledger, owner);
      await ledger.autoExecutePO(id);
      expect((await ledger.decisions(id)).id).to.equal(id);
      expect((await ledger.purchaseOrders(1)).decisionId).to.equal(id);
      expect(await ledger.decisionIds(0)).to.equal(id);
      expect(await ledger.poIds(0)).to.equal(1n);
    });
  });
});

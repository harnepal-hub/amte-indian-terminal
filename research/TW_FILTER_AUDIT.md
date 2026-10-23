# TW Filter Audit — Research Only

**Status:** Not approved for live trading. This document is a research note, not a strategy change.
**Repository:** `harnepal-hub/amte-indian-terminal`
**Baseline commit reviewed:** `4622304b05ab4cad47bc73fc5c6602a0070d9dac` (2026-10-12)

## 1. Ledger snapshot

The committed CSV ledgers currently contain:

| Strategy | Trades | Winners | Losers | Win rate | Net P&L | Gross P&L | Recorded friction | Profit factor* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMTE | 53 | 11 | 42 | 20.8% | -₹7,525.80 | -₹6,148.47 | ₹1,377.38 | 0.34 |
| TW_ORIG | 28 | 15 | 13 | 53.6% | +₹2,830.51 | +₹3,636.36 | ₹805.85 | 2.13 |
| TW_TUNED | 11 | 4 | 7 | 36.4% | -₹378.57 | -₹81.43 | ₹297.18 | 0.77 |

\*Profit factor here uses positive net trade P&L divided by the absolute sum of negative net trade P&L. Small sample size and limited trading dates make these descriptive, not robust estimates. TW_ORIG's positive ledger is not evidence of an established edge; it needs chronological out-of-sample evaluation and realistic costs.

## 2. Code findings to validate/fix before optimizing indicators

1. **No per-signal cooldown / same setup can re-enter quickly.** The engine can create a new pending trade after the prior position closes if the same candle-derived condition remains true on later polling cycles. Add a per-strategy/per-symbol signal candle key and cooldown/one-entry-per-signal guard, and test it.
2. **Risk sizing excludes transaction friction.** Quantity is calculated as `RISK_PER_TRADE_INR / abs(entry - stop)`; ledger stop losses therefore often exceed the ₹250 gross risk budget by charges and any execution gap. Size using stop loss plus estimated round-trip friction and apply a maximum notional/capital constraint.
3. **Open positions and trade state are in memory only.** The Streamlit process restart loses state; only closed trade CSVs persist. This undermines deduplication and restart safety. Persist position/signal state or conservatively require a reconciliation step after restart.
4. **Market data freshness / current candle semantics need explicit checks.** The code uses the penultimate 15m bar for the signal and the latest bar's close as `live_price`, then polls about once per minute. Add tests to show that signals are acted on once per completed candle and that no developing candle is treated as finalized.
5. **EOD handling should be verified.** Positions are squared at 15:15, but condition checks use `now_ist >= auto_square_off_time` only after market-hour gate; paper logs include timestamps after 15:15. Make the execution boundary consistent and test pending-entry cancellation at cutoff.
6. **Long-only cash-equity account constraints are not modeled.** The engine emits SHORT signals on NSE cash-equity symbols. Confirm a permitted product (e.g. intraday product) and broker/market rules before any live use; no real-order integration is in scope here.
7. **Ledger synchronization is non-transactional and has no duplicate key.** GitHub read-modify-write can race across processes/restarts and create duplicate ledger events. Add an idempotent trade identifier and handle HTTP errors/409 updates rather than assuming the write succeeded.
8. **Cost model must be checked against the user's actual broker and current schedule.** The labels say “Kotak friction,” while the implementation has zero brokerage and hard-coded taxes/charges. Do not treat it as a verified broker statement.

## 3. TW_ORIG vs TW_TUNED signal logic

Both use 15-minute candles, a 16-length EHMA/Hull-like calculation, EMA100 direction filter and ATR-based stops/targets. TW_ORIG triggers on a two-bar shifted line relationship. TW_TUNED changes that relationship to a three-bar shift and adds `ATR > ATR_50`. The current observed ledgers show fewer TW_TUNED trades and worse net results, but cannot isolate why because the samples differ and the execution is stateful. Run a paired replay where both variants use identical symbol candles, timestamps, execution assumptions, position limits and cost model.

## 4. Minimum research protocol

- Use the same time-stamped, split-adjusted 15m OHLCV source for all five symbols; record data source and retrieval timestamp.
- Build causal signals only on completed candles; enter no earlier than the next bar (or model a documented executable order rule).
- Compare TW_ORIG, current TW_TUNED, and a small predeclared candidate set. Do not tune thresholds repeatedly against the final holdout.
- Use chronological train / validation / untouched test periods, with at least 100 closed trades in aggregate before even considering a strategy candidate; report symbol-level results and sample sizes.
- Include realistic brokerage, STT, exchange charges, GST, stamp duty, SEBI charges, spread/slippage, and conservative stop gaps using the actual supported broker/product.
- Report net P&L, profit factor, expectancy/trade, drawdown, trade frequency, win rate, cost share of gross P&L, and performance by symbol and market regime.
- Compare with a no-trade baseline and a simple trend benchmark. Do not claim that a 60% win rate is sufficient without expectancy and payoff analysis.
- Acceptance gate: positive net expectancy after costs, positive and reasonably stable holdout results across more than one symbol, drawdown within explicit risk limits, and no lookahead/re-entry bugs. Continue paper testing before any live consideration.

## 5. Safety state

This research branch must not enable real-money order placement. No production parameters or running trade logic are changed by this note.

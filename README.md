# Settlement Feasibility & Fee Engine — Take-home

Welcome, and thanks for taking the time. The full problem is in
[`ASSIGNMENT.md`](./ASSIGNMENT.md). This README is just orientation.

## The task in one line

Given a client's escrow account, a settlement offer, and a creditor's rules,
decide whether the offer is affordable (and schedule it, collecting our fee as
early as allowed) or — if not — compute the minimum extra funding needed.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
hiring_takehome/
├── ASSIGNMENT.md            # full specification — read this
├── feasibility/
│   ├── models.py            # data models, JSON loaders, date/EOM helpers (provided)
│   └── engine.py            # >>> implement evaluate_offer here <<< (+ Result shape)
├── cases/                   # four example cases (client.json / offer.json / creditor_rules.json)
│   ├── case1_feasible_even
│   ├── case2_infeasible_minima
│   ├── case3_balloon
│   └── case4_tiers
├── tests/
│   ├── test_smoke.py        # scaffolding sanity tests (pass out of the box)
│   └── test_cases.py        # example expectations — make these pass, then add your own
├── run.py                   # python run.py cases/<case>
└── requirements.txt
```

## Run

```bash
# evaluate a single case (prints the Result as JSON)
python run.py cases/case1_feasible_even

# tests
pytest -q
```

Out of the box, `tests/test_smoke.py` passes and `tests/test_cases.py` fails —
the latter is your target. Go beyond those four cases with your own tests.

---

## Approach

### Architecture

All logic lives in `feasibility/engine.py`, organized into layers:

1. **Utilities** -- `round_half_up`, `compute_offer_total`, `compute_program_fee`,
   `compute_floor`, `compute_max_k`.
2. **Payment generators** -- one per shape: `generate_even_payments`,
   `generate_balloon_payments`, `generate_staircase_payments`.
3. **Fee allocation** -- `allocate_fee_frontloaded` greedily assigns program fee
   to the earliest cadence dates, with look-ahead safety to avoid causing negative
   balances on intermediate (non-cadence) dates.
4. **Schedule builder** -- `try_schedule` wires payments + fee allocation + a final
   date-by-date simulation into `ScheduleRow` objects.
5. **Feasibility search** -- `find_feasible_schedule` picks the shape from the
   creditor flags, then tries `k` from max down to 1. First feasible schedule wins.
6. **Part 2** -- `find_min_lump_sum` and `find_min_monthly_increment` each binary
   search over the amount, calling the full feasibility check as the predicate.
7. **Orchestrator** -- `evaluate_offer` ties it all together.

### Alternatives considered

- **LP / constraint solver:** Using `ortools` or `scipy.optimize.linprog` to model
  the schedule as a linear program. Would be elegant but overkill for the typical
  input sizes (k <= 12, max_segments <= 4). The combinatorial enumeration approach
  is fast enough (sub-second on all four cases) and much easier to reason about
  for correctness.
- **Top-down simulation with backtracking:** Simulate forward, choosing payments
  greedily, and backtrack on failure. Rejected because the staircase partition
  enumeration is cleaner and guarantees we find the globally best (most
  front-loaded) option, not just the first one that works.
- **Bottom-up fee allocation (fill from the end):** Allocate fee starting from
  the last cadence date to avoid cash-flow conflicts. Rejected because it
  contradicts the objective (front-load fee as early as possible).

---

## Payment shape interpretation

### Even (`even_pays = true`)

All `k` creditor payments are equal. When `offer_total` is not evenly divisible
by `k`, the base payment is `offer_total // k` and the remaining
`offer_total % k` cents are added to the **latest** payments (+1 cent each), so
the sequence stays non-decreasing. Example: 10000 / 3 = [3333, 3333, 3334].

The choice of `k` is driven by the objective: higher `k` means lower per-payment
amounts, leaving more headroom for early fee collection. We try `k` from max
down to 1 and take the first feasible result.

### Balloon (`is_ballooning_allowed = true`)

Payments 1 through k-1 are set to the **floor** at each position (the minimum
allowed by `min_payment_cents`, token-pay limits, and tier step-ups). The final
payment absorbs the entire remaining balance. This naturally keeps early payments
as low as possible, maximizing early cash available for our fee.

Token pays and tiers interact straightforwardly: each early payment is set to its
position's floor. If a tier kicks in at position `j`, payments from `j` onward
use the higher floor. The final balloon payment must still be >= the floor at its
position and >= the previous payment (non-decreasing).

### Staircase (neither even nor balloon)

Payments step up over time using at most `max_segments` distinct levels. The shape
is produced by **enumerating all valid partitions** of `k` positions into up to
`max_segments` contiguous groups, then for each partition:

1. Set each group (except the last) to its minimum feasible level -- the maximum
   floor across all positions in that group, and at least as high as the previous
   group's level (non-decreasing).
2. The last group absorbs the remainder. If the remainder doesn't divide evenly,
   the extra cents go on the latest positions within the group.
3. Validate: distinct levels <= `max_segments`, non-decreasing, all floors met,
   token-pay count within limits.

Among all valid partitions, we pick the **lexicographically smallest** payment
vector -- this is the one with the lowest early payments, which maximizes early
fee collection.

For typical inputs (k <= 12, max_segments <= 4), the number of candidate
partitions is at most C(11, 3) = 165, so enumeration is fast.

---

## Fee front-loading strategy

The program fee is allocated greedily, walking cadence dates in chronological
order. At each cadence date, after applying same-day credits and debits (creditor
payment + bank fee), we compute the maximum fee we can safely extract:

```
max_safe_fee = current_balance + min_cumulative_delta_before_next_cadence
```

where `min_cumulative_delta` accounts for any committed ledger debits (e.g.,
payments for other settled debts) that fall between the current cadence date and
the next one. This look-ahead prevents the greedy allocation from causing a
negative balance on an intermediate date.

If the fee cannot be fully collected during creditor-payment months, additional
**fee-only cadence dates** are appended (within the horizon). Fee-only dates
carry no creditor payment and no bank fee.

---

## Assumptions

1. **Rounding:** The provided `offer_total_cents()` and `program_fee_cents()`
   helpers in `models.py` use Python's `round()` (round-half-even). The
   assignment requires round-half-up. I implemented `round_half_up()` explicitly
   using `math.floor(x + 0.5)` and use it for all money computations instead of
   the provided helpers. (For the four test cases, the values happen to coincide,
   but the distinction matters for amounts ending in exactly .5.)

2. **Offer field name:** The `Offer` dataclass uses `current_balance_cents` for
   the creditor balance (the assignment text calls it `creditor_balance_cents`).
   I use the field as defined in the code.

3. **Token pay definition:** A "token pay" is a creditor payment equal to
   **exactly** `min_payment_cents` (the base minimum). If a tier step-up raises
   the effective floor above `min_payment_cents`, a payment at the tier floor is
   **not** a token pay. The token-pay rule (`max_token_pays`) only constrains
   payments sitting at the base minimum.

4. **Lump sum placement:** Since an earlier lump sum is "weakly more useful"
   (per the spec), I place it on the earliest possible date (the day after
   `as_of_date`) and binary search for the minimum amount. A single search
   suffices because earlier placement is always at least as good as later.

5. **Monthly increment scope:** The increment is added to **every** future credit
   entry in the ledger (all entries after `as_of_date` with type `credit`).
   `num_drafts` reports the total count of these entries, including drafts that
   fall after the last payment date.

6. **Fee-only dates:** When the program fee cannot be fully collected during
   creditor-payment months, I extend the cadence with fee-only dates (no creditor
   payment, no bank fee) up to the horizon. These are valid cadence dates per
   constraint 6.

7. **Shape selection is deterministic:** `even_pays` takes precedence. If not
   set, `is_ballooning_allowed` takes precedence. Otherwise, staircase. This
   follows the spec's description of the flags as mutually informative.

---

## Known edge cases and limitations

- **Staircase remainder distribution:** When the last segment's total doesn't
  divide evenly, the +1 cent payments count as a distinct level. If this pushes
  distinct levels above `max_segments`, that partition is rejected. This means
  some edge cases with tight `max_segments` and non-divisible totals may require
  a lower `k` than theoretically possible.

- **Binary search upper bound:** For Part 2, the upper bound for the binary
  search is `offer_total + total_fee + bank_fees * max_k`. This is generous and
  covers all practical cases, but for extremely large debts with tiny drafts, the
  search might take ~20 iterations (log2 of the range) per candidate.

- **Performance:** The staircase partition enumeration is O(C(k-1, S-1)) per
  value of k. For k=12 and S=4, that's 165 candidates per k, times up to 12
  values of k = ~2000 partition evaluations in the worst case. Each is O(k), so
  the total is well under a millisecond. The binary search for Part 2 adds at
  most ~20 iterations, each running the full feasibility search. All four cases
  complete in under 200ms total.

- **Leap years:** The date helpers in `models.py` handle Feb 28/29 correctly via
  `calendar.monthrange`. The EOM cadence naturally adapts (Jan 31 -> Feb 28 in
  non-leap, Feb 29 in leap).

---

## Test coverage

73 tests covering all required scenarios:

| Category | Tests |
|---|---|
| Even / staircase / balloon shapes | 20 |
| Token-pay and tier floors | 7 |
| `max_segments` cap | 2 |
| Exact-sum constraint | 2 |
| Date-by-date simulation, same-day ordering, balance = $0 | 4 |
| Horizon limit | 2 |
| Fee compliance (timing, total, bank fee rules) | 7 |
| Part 2 minima (lump sum, monthly increment, guardrails) | 7 |
| Edge cases (rounding, serialization, committed debits, single payment) | 7 |
| Round-half-up correctness | 6 |
| Provided case expectations | 4 |
| Scaffolding smoke tests | 6 |
# settlement_engine

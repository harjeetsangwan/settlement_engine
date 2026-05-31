"""Candidate implementation goes here.

Implement ``evaluate_offer`` so that it satisfies the rules in ASSIGNMENT.md and
the example expectations in tests/test_cases.py. The dataclasses below define the
required OUTPUT shape (see ASSIGNMENT.md "Output"). You may add helpers, modules,
or rewrite internals freely, but keep ``evaluate_offer``'s signature and the
serialized shape of ``Result`` (so the runner and tests work).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations

from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    default_first_payment_date,
    monthly_payment_dates,
)


# ---------------------------------------------------------------------------
# Output dataclasses (keep these as-is for runner/test compatibility)
# ---------------------------------------------------------------------------

@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    # lump-sum only:
    date: date | None = None
    # monthly-increment only:
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        out: dict = {"feasible": self.feasible, "pay_shape_used": self.pay_shape_used}
        out["schedule"] = (
            [
                {
                    "date": r.date.isoformat(),
                    "creditor_payment_cents": r.creditor_payment_cents,
                    "program_fee_cents": r.program_fee_cents,
                    "bank_fee_cents": r.bank_fee_cents,
                    "balance_cents": r.balance_cents,
                }
                for r in self.schedule
            ]
            if self.schedule is not None
            else None
        )
        if self.additional_funds is None:
            out["additional_funds"] = None
        else:
            def opt(o: FundsOption) -> dict:
                d = {
                    "amount_cents": o.amount_cents,
                    "within_guardrail": o.within_guardrail,
                    "reason": o.reason,
                }
                if o.date is not None:
                    d["date"] = o.date.isoformat()
                if o.num_drafts is not None:
                    d["num_drafts"] = o.num_drafts
                return d

            out["additional_funds"] = {
                "lump_sum": opt(self.additional_funds.lump_sum),
                "monthly_increment": opt(self.additional_funds.monthly_increment),
            }
        return out


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def round_half_up(x: float) -> int:
    """Round-half-up: 0.5 always rounds away from zero."""
    return math.floor(x + 0.5)


def compute_offer_total(offer: Offer) -> int:
    return round_half_up(offer.settlement_pct * offer.current_balance_cents)


def compute_program_fee(offer: Offer, rules: CreditorRules) -> int:
    return round_half_up(rules.program_fee_pct * offer.original_balance_cents)


def compute_floor(position: int, rules: CreditorRules, token_pays_used: int) -> int:
    """Minimum creditor payment at 1-based position.

    Floor = max of:
      - min_payment_cents (base)
      - any applicable tier floor
      - if token_pays_used >= max_token_pays, base min + 1
    """
    floor = rules.min_payment_cents
    for from_payment, min_cents in rules.min_payment_tiers:
        if position >= from_payment:
            floor = max(floor, min_cents)
    if token_pays_used >= rules.max_token_pays and floor <= rules.min_payment_cents:
        floor = rules.min_payment_cents + 1
    return floor


def _first_payment_date(client: Client, offer: Offer) -> date:
    return offer.first_payment_date or default_first_payment_date(client)


def compute_max_k(client: Client, offer: Offer, rules: CreditorRules) -> int:
    """Max number of creditor payments that fit within the horizon."""
    limit = min(rules.max_payments, rules.max_terms)
    fpd = _first_payment_date(client, offer)
    horizon = client.last_draft_date
    # Generate up to limit dates and count those within horizon
    dates = monthly_payment_dates(fpd, limit)
    return sum(1 for d in dates if d <= horizon)


# ---------------------------------------------------------------------------
# Payment shape generators
# ---------------------------------------------------------------------------

def generate_even_payments(
    offer_total: int, k: int, rules: CreditorRules
) -> list[int] | None:
    """All payments equal; remainder cents on latest payments (non-decreasing)."""
    if k <= 0 or offer_total <= 0:
        return None
    base = offer_total // k
    remainder = offer_total - base * k

    # Payments: first (k - remainder) at base, last remainder at (base + 1)
    payments = [base] * (k - remainder) + [base + 1] * remainder

    # Validate floors and token pays
    token_count = sum(1 for p in payments if p == rules.min_payment_cents)
    if token_count > rules.max_token_pays:
        return None

    for i, p in enumerate(payments):
        tokens_before = sum(1 for pp in payments[:i] if pp == rules.min_payment_cents)
        floor = compute_floor(i + 1, rules, tokens_before)
        if p < floor:
            return None

    return payments


def generate_balloon_payments(
    offer_total: int, k: int, rules: CreditorRules
) -> list[int] | None:
    """Min payments early, final payment absorbs the rest."""
    if not rules.is_ballooning_allowed or k <= 0 or offer_total <= 0:
        return None

    payments: list[int] = []
    token_count = 0
    running_sum = 0

    for i in range(k - 1):
        pos = i + 1
        floor = compute_floor(pos, rules, token_count)
        payments.append(floor)
        if floor == rules.min_payment_cents:
            token_count += 1
        running_sum += floor

    final = offer_total - running_sum
    if final <= 0:
        return None

    # Non-decreasing: final must be >= previous
    if k > 1 and final < payments[-1]:
        return None

    # Check floor for final position
    final_floor = compute_floor(k, rules, token_count)
    if final < final_floor:
        return None

    payments.append(final)
    return payments


def generate_staircase_payments(
    offer_total: int, k: int, rules: CreditorRules
) -> list[int] | None:
    """Step-up payments with at most max_segments distinct levels.

    Objective: minimize early payments (front-load fee).
    Enumerate all valid partitions, pick the lexicographically smallest.
    """
    if k <= 0 or offer_total <= 0:
        return None

    S = rules.max_segments
    best: list[int] | None = None

    for num_seg in range(1, min(S, k) + 1):
        if num_seg == 1:
            split_configs: list[tuple[int, ...]] = [()]
        else:
            split_configs = list(combinations(range(1, k), num_seg - 1))

        for splits in split_configs:
            payments = _try_staircase_partition(offer_total, k, splits, rules)
            if payments is not None:
                if best is None or payments < best:
                    best = payments

    return best


def _try_staircase_partition(
    offer_total: int,
    k: int,
    splits: tuple[int, ...],
    rules: CreditorRules,
) -> list[int] | None:
    """Try a specific partition of k positions into segments.

    splits are 0-based indices where new segments begin.
    E.g. splits=(3,) with k=6 means segment1=[0,1,2], segment2=[3,4,5].
    """
    boundaries = [0] + list(splits) + [k]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Compute minimum level for each segment
    levels: list[int] = []
    token_count = 0

    for seg_idx, (start, end) in enumerate(segments):
        seg_size = end - start

        # Floor for this segment = max floor across all positions in it
        seg_floor = 0
        temp_tokens = token_count
        for pos_idx in range(start, end):
            pos = pos_idx + 1  # 1-based
            f = compute_floor(pos, rules, temp_tokens)
            seg_floor = max(seg_floor, f)
            # Optimistically count tokens for floor calculation within segment
            if f == rules.min_payment_cents:
                temp_tokens += 1

        # Non-decreasing: level >= previous segment level
        if levels:
            seg_floor = max(seg_floor, levels[-1])

        if seg_idx < len(segments) - 1:
            # Not last segment: use the floor
            levels.append(seg_floor)
            # Count actual token pays in this segment
            if seg_floor == rules.min_payment_cents:
                token_count += seg_size
        else:
            # Last segment: absorb remainder
            spent = sum(levels[i] * (segments[i][1] - segments[i][0]) for i in range(seg_idx))
            remaining = offer_total - spent

            if remaining <= 0 or remaining < seg_floor * seg_size:
                return None

            base_level = remaining // seg_size
            rem = remaining - base_level * seg_size

            if base_level < seg_floor:
                return None

            # Build candidate levels set to check max_segments
            all_levels = set(levels)
            all_levels.add(base_level)
            if rem > 0:
                all_levels.add(base_level + 1)
            if len(all_levels) > rules.max_segments:
                return None

            levels.append(base_level)

    # Build the payment array
    payments: list[int] = []
    for seg_idx, (start, end) in enumerate(segments):
        seg_size = end - start
        if seg_idx < len(segments) - 1:
            payments.extend([levels[seg_idx]] * seg_size)
        else:
            spent = sum(payments)
            remaining = offer_total - spent
            base = remaining // seg_size
            rem = remaining - base * seg_size
            # Remainder on latest positions (non-decreasing)
            payments.extend([base] * (seg_size - rem) + [base + 1] * rem)

    # Final validations
    if sum(payments) != offer_total:
        return None
    for i in range(1, len(payments)):
        if payments[i] < payments[i - 1]:
            return None

    # Validate token pay count
    token_count = sum(1 for p in payments if p == rules.min_payment_cents)
    if token_count > rules.max_token_pays:
        return None

    # Validate all floors
    tc = 0
    for i, p in enumerate(payments):
        floor = compute_floor(i + 1, rules, tc)
        if p < floor:
            return None
        if p == rules.min_payment_cents:
            tc += 1

    if len(set(payments)) > rules.max_segments:
        return None

    return payments


# ---------------------------------------------------------------------------
# Ledger event helpers
# ---------------------------------------------------------------------------

def _build_event_map(
    client: Client,
    extra_credits: list[LedgerEntry] | None = None,
) -> dict[date, tuple[int, int]]:
    """Build a map of date -> (total_credits, total_debits) for future events.

    Only includes ledger entries after as_of_date.
    """
    events: dict[date, list[int]] = defaultdict(lambda: [0, 0])

    for entry in client.ledger:
        if entry.date <= client.as_of_date:
            continue
        if entry.type == "credit":
            events[entry.date][0] += entry.amount_cents
        else:
            events[entry.date][1] += entry.amount_cents

    if extra_credits:
        for entry in extra_credits:
            events[entry.date][0] += entry.amount_cents

    return {d: (v[0], v[1]) for d, v in events.items()}


# ---------------------------------------------------------------------------
# Fee allocation (front-loaded)
# ---------------------------------------------------------------------------

def allocate_fee_frontloaded(
    client: Client,
    offer: Offer,
    cadence_dates: list[date],
    creditor_payments: list[int],
    bank_fees: list[int],
    total_fee: int,
    extra_credits: list[LedgerEntry] | None = None,
) -> tuple[list[int], list[date]] | None:
    """Greedily allocate program fee front-loaded across cadence dates.

    May extend with fee-only dates past the last creditor payment (within horizon).

    Returns (fee_allocations, all_cadence_dates_used) or None if fee can't be
    fully collected.
    """
    if total_fee == 0:
        return [0] * len(cadence_dates), list(cadence_dates)

    horizon = client.last_draft_date
    events = _build_event_map(client, extra_credits)

    # Extend cadence with fee-only dates if needed
    fpd = _first_payment_date(client, offer)
    max_possible = 36  # generous upper bound
    all_possible_dates = monthly_payment_dates(fpd, max_possible)
    fee_only_dates = [d for d in all_possible_dates
                      if d > (cadence_dates[-1] if cadence_dates else fpd) and d <= horizon]

    # All dates we'll consider (creditor dates + possible fee-only dates)
    all_cadence = list(cadence_dates) + fee_only_dates
    all_creditor = list(creditor_payments) + [0] * len(fee_only_dates)
    all_bank = list(bank_fees) + [0] * len(fee_only_dates)

    # Gather ALL dates in timeline (ledger events + cadence dates)
    all_dates_set: set[date] = set(events.keys())
    for d in all_cadence:
        all_dates_set.add(d)
    all_dates_sorted = sorted(all_dates_set)

    cadence_index = {}
    for i, d in enumerate(all_cadence):
        cadence_index[d] = i

    # Simulate forward, greedily assigning fee
    balance = client.current_balance_cents
    fee_remaining = total_fee
    fee_alloc = [0] * len(all_cadence)

    for d in all_dates_sorted:
        # Credits first
        if d in events:
            balance += events[d][0]

        # Committed debits (non-cadence)
        if d in events:
            balance -= events[d][1]

        # Cadence debits
        if d in cadence_index:
            idx = cadence_index[d]
            balance -= all_creditor[idx]
            balance -= all_bank[idx]

            # Compute max safe fee: look ahead to all dates between now and
            # the next cadence date. Ensure balance stays >= 0 on those dates.
            next_cadence = None
            for nc in all_cadence:
                if nc > d:
                    next_cadence = nc
                    break

            # Find minimum cumulative delta between now and next cadence
            min_delta = 0
            cumulative = 0
            for future_d in all_dates_sorted:
                if future_d <= d:
                    continue
                if next_cadence is not None and future_d >= next_cadence:
                    break
                if future_d in events:
                    cumulative += events[future_d][0] - events[future_d][1]
                    min_delta = min(min_delta, cumulative)

            max_safe = balance + min_delta
            fee = min(max(0, max_safe), fee_remaining)
            fee_alloc[idx] = fee
            balance -= fee
            fee_remaining -= fee

        if balance < 0:
            return None

    if fee_remaining > 0:
        return None

    # Trim unused fee-only dates from the end
    last_used = len(cadence_dates) - 1
    for i in range(len(cadence_dates), len(all_cadence)):
        if fee_alloc[i] > 0:
            last_used = i

    used_count = last_used + 1
    return fee_alloc[:used_count], all_cadence[:used_count]


# ---------------------------------------------------------------------------
# Schedule building
# ---------------------------------------------------------------------------

def try_schedule(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    creditor_payments: list[int],
    cadence_dates: list[date],
    total_fee: int,
    extra_credits: list[LedgerEntry] | None = None,
) -> list[ScheduleRow] | None:
    """Try to build a valid schedule given creditor payments.

    Returns list of ScheduleRow if feasible, None otherwise.
    """
    k = len(creditor_payments)
    bank_fees = [rules.bank_fee_cents if creditor_payments[i] > 0 else 0
                 for i in range(k)]

    # Allocate fee
    result = allocate_fee_frontloaded(
        client, offer, cadence_dates, creditor_payments, bank_fees,
        total_fee, extra_credits,
    )
    if result is None:
        return None

    fee_alloc, all_dates_used = result

    # Extend creditor/bank arrays for fee-only dates
    all_creditor = list(creditor_payments) + [0] * (len(all_dates_used) - k)
    all_bank = list(bank_fees) + [0] * (len(all_dates_used) - k)

    # Final simulation to build ScheduleRows with correct running balances
    events = _build_event_map(client, extra_credits)

    all_dates_set: set[date] = set(events.keys())
    for d in all_dates_used:
        all_dates_set.add(d)
    all_dates_sorted = sorted(all_dates_set)

    cadence_index = {d: i for i, d in enumerate(all_dates_used)}

    balance = client.current_balance_cents
    rows: list[ScheduleRow] = []

    for d in all_dates_sorted:
        # Credits first
        if d in events:
            balance += events[d][0]
        # Committed debits
        if d in events:
            balance -= events[d][1]
        # Cadence debits
        if d in cadence_index:
            idx = cadence_index[d]
            balance -= all_creditor[idx]
            balance -= all_bank[idx]
            balance -= fee_alloc[idx]

            rows.append(ScheduleRow(
                date=d,
                creditor_payment_cents=all_creditor[idx],
                program_fee_cents=fee_alloc[idx],
                bank_fee_cents=all_bank[idx],
                balance_cents=balance,
            ))

        if balance < 0:
            return None

    return rows


# ---------------------------------------------------------------------------
# Feasibility search
# ---------------------------------------------------------------------------

def _determine_shape(rules: CreditorRules) -> str:
    if rules.even_pays:
        return "even"
    if rules.is_ballooning_allowed:
        return "balloon"
    return "staircase"


def find_feasible_schedule(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    extra_credits: list[LedgerEntry] | None = None,
) -> Result | None:
    """Try to find a feasible schedule. Returns Result or None."""
    offer_total = compute_offer_total(offer)
    total_fee = compute_program_fee(offer, rules)
    mk = compute_max_k(client, offer, rules)
    shape = _determine_shape(rules)

    if shape == "even":
        generator = generate_even_payments
    elif shape == "balloon":
        generator = generate_balloon_payments
    else:
        generator = generate_staircase_payments

    fpd = _first_payment_date(client, offer)

    for k in range(mk, 0, -1):
        cadence_dates = monthly_payment_dates(fpd, k)
        # Ensure all within horizon
        if cadence_dates[-1] > client.last_draft_date:
            continue

        payments = generator(offer_total, k, rules)
        if payments is None:
            continue

        schedule = try_schedule(
            client, offer, rules, payments, cadence_dates,
            total_fee, extra_credits,
        )
        if schedule is not None:
            return Result(
                feasible=True,
                pay_shape_used=shape,
                schedule=schedule,
            )

    return None


# ---------------------------------------------------------------------------
# Part 2: Additional funds
# ---------------------------------------------------------------------------

def _future_draft_entries(client: Client) -> list[LedgerEntry]:
    """All future credit entries (drafts dated after as_of_date)."""
    return [e for e in client.ledger
            if e.date > client.as_of_date and e.type == "credit"]


def find_min_lump_sum(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
) -> FundsOption:
    """Binary search for minimum lump sum on the earliest useful date."""
    offer_total = compute_offer_total(offer)
    total_fee = compute_program_fee(offer, rules)
    mk = compute_max_k(client, offer, rules)

    # Place lump on earliest possible date: day after as_of_date
    lump_date = client.as_of_date + timedelta(days=1)

    # Upper bound: offer_total + total_fee + bank_fees
    upper = offer_total + total_fee + rules.bank_fee_cents * mk
    lo, hi = 1, upper
    best_L: int | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        extra = [LedgerEntry(lump_date, mid, "credit")]
        result = find_feasible_schedule(client, offer, rules, extra)
        if result is not None:
            best_L = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # If no lump works at earliest date, try other dates
    if best_L is None:
        # Try each future draft date and cadence date
        candidate_dates: list[date] = sorted(set(
            [e.date for e in client.ledger if e.date > client.as_of_date]
        ))
        for try_date in candidate_dates:
            lo2, hi2 = 1, upper
            while lo2 <= hi2:
                mid2 = (lo2 + hi2) // 2
                extra2 = [LedgerEntry(try_date, mid2, "credit")]
                result2 = find_feasible_schedule(client, offer, rules, extra2)
                if result2 is not None:
                    if best_L is None or mid2 < best_L:
                        best_L = mid2
                        lump_date = try_date
                    hi2 = mid2 - 1
                else:
                    lo2 = mid2 + 1

    if best_L is None:
        best_L = upper
        lump_date = client.as_of_date + timedelta(days=1)

    guardrail_limit = round_half_up(0.65 * offer_total)
    within = best_L <= guardrail_limit
    reason = "" if within else (
        f"Lump sum {best_L} exceeds 65% of offer total ({guardrail_limit})"
    )

    return FundsOption(
        amount_cents=best_L,
        within_guardrail=within,
        reason=reason,
        date=lump_date,
    )


def find_min_monthly_increment(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
) -> FundsOption:
    """Binary search for minimum uniform monthly increment."""
    offer_total = compute_offer_total(offer)
    total_fee = compute_program_fee(offer, rules)
    mk = compute_max_k(client, offer, rules)

    future_drafts = _future_draft_entries(client)
    num_drafts = len(future_drafts)

    if num_drafts == 0:
        # No future drafts to augment
        return FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason="No future drafts to augment",
            num_drafts=0,
        )

    upper = offer_total + total_fee + rules.bank_fee_cents * mk
    lo, hi = 1, upper
    best_X: int | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        extra = [LedgerEntry(d.date, mid, "credit") for d in future_drafts]
        result = find_feasible_schedule(client, offer, rules, extra)
        if result is not None:
            best_X = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best_X is None:
        best_X = upper

    guardrail_limit = max(10000, round_half_up(0.40 * client.draft_amount_cents))
    within = best_X <= guardrail_limit
    reason = "" if within else (
        f"Monthly increment {best_X} exceeds guardrail ({guardrail_limit})"
    )

    return FundsOption(
        amount_cents=best_X,
        within_guardrail=within,
        reason=reason,
        num_drafts=num_drafts,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    """Evaluate a single offer. See ASSIGNMENT.md for the full specification."""

    result = find_feasible_schedule(client, offer, rules)
    if result is not None:
        return result

    # Infeasible — compute additional funds
    lump = find_min_lump_sum(client, offer, rules)
    monthly = find_min_monthly_increment(client, offer, rules)

    return Result(
        feasible=False,
        pay_shape_used=None,
        schedule=None,
        additional_funds=AdditionalFunds(
            lump_sum=lump,
            monthly_increment=monthly,
        ),
    )

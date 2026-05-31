"""Example expectations for the four provided cases.

These FAIL until you implement feasibility/engine.py::evaluate_offer. Treat them
as the minimum bar — your own test suite should go well beyond these. They do not
pin an exact schedule (several valid schedules may exist); they assert the
verdict, the pay shape, and the Part 2 minima.
"""

from __future__ import annotations

from datetime import date

import pytest

from feasibility.engine import (
    AdditionalFunds,
    Result,
    ScheduleRow,
    compute_floor,
    compute_offer_total,
    compute_program_fee,
    evaluate_offer,
    find_feasible_schedule,
    generate_balloon_payments,
    generate_even_payments,
    generate_staircase_payments,
    round_half_up,
)
from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    load_case,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(case: str):
    client, offer, rules = load_case(f"cases/{case}")
    return evaluate_offer(client, offer, rules)


def _make_client(
    draft_amount: int = 10000,
    draft_day: int = 1,
    first_draft: str = "2026-01-01",
    last_draft: str = "2026-06-01",
    as_of: str = "2025-12-31",
    balance: int = 0,
    ledger: list[dict] | None = None,
) -> Client:
    """Convenience factory for building test Client objects."""
    fd = date.fromisoformat(first_draft)
    ld = date.fromisoformat(last_draft)
    ao = date.fromisoformat(as_of)

    if ledger is None:
        # Auto-generate monthly credit entries
        entries: list[LedgerEntry] = []
        d = fd
        while d <= ld:
            entries.append(LedgerEntry(d, draft_amount, "credit"))
            m = d.month + 1
            y = d.year
            if m > 12:
                m = 1
                y += 1
            d = date(y, m, draft_day)
    else:
        entries = [
            LedgerEntry(date.fromisoformat(e["date"]), e["amount_cents"], e["type"])
            for e in ledger
        ]

    return Client(
        draft_amount_cents=draft_amount,
        draft_day=draft_day,
        first_draft_date=fd,
        last_draft_date=ld,
        as_of_date=ao,
        current_balance_cents=balance,
        ledger=entries,
    )


def _make_offer(
    creditor: str = "TestCo",
    creditor_balance: int = 100000,
    original_balance: int = 100000,
    settlement_pct: float = 0.5,
    first_payment_date: str | None = None,
) -> Offer:
    fpd = date.fromisoformat(first_payment_date) if first_payment_date else None
    return Offer(
        creditor=creditor,
        current_balance_cents=creditor_balance,
        original_balance_cents=original_balance,
        settlement_pct=settlement_pct,
        first_payment_date=fpd,
    )


def _make_rules(
    max_terms: int = 12,
    max_payments: int = 12,
    min_payment_cents: int = 2500,
    max_token_pays: int = 12,
    min_payment_tiers: list[tuple[int, int]] | None = None,
    even_pays: bool = False,
    is_ballooning_allowed: bool = False,
    max_segments: int = 4,
    bank_fee_cents: int = 0,
    program_fee_pct: float = 0.0,
) -> CreditorRules:
    return CreditorRules(
        max_terms=max_terms,
        max_payments=max_payments,
        min_payment_cents=min_payment_cents,
        max_token_pays=max_token_pays,
        min_payment_tiers=min_payment_tiers or [],
        even_pays=even_pays,
        is_ballooning_allowed=is_ballooning_allowed,
        max_segments=max_segments,
        bank_fee_cents=bank_fee_cents,
        program_fee_pct=program_fee_pct,
    )


# ===================================================================
# Original 4 case tests (unchanged)
# ===================================================================

def test_case1_feasible_even():
    r = _run("case1_feasible_even")
    assert r.feasible is True
    assert r.pay_shape_used == "even"
    assert r.schedule is not None
    # balance must never go negative
    assert all(row.balance_cents >= 0 for row in r.schedule)


def test_case2_infeasible_minima():
    r = _run("case2_infeasible_minima")
    assert r.feasible is False
    af = r.additional_funds
    assert af is not None
    assert af.lump_sum.amount_cents == 10000
    assert af.lump_sum.within_guardrail is True
    assert af.monthly_increment.amount_cents == 2500
    assert af.monthly_increment.num_drafts == 5
    assert af.monthly_increment.within_guardrail is True


def test_case3_requires_balloon():
    r = _run("case3_balloon")
    assert r.feasible is True
    # this creditor allows ballooning; the solver defers payment into a final balloon
    assert r.pay_shape_used == "balloon"


def test_case4_tiered_minimums():
    r = _run("case4_tiers")
    assert r.feasible is True
    assert r.pay_shape_used == "staircase"
    # payments 7+ must respect the $50 tier floor
    payments = [row.creditor_payment_cents for row in r.schedule if row.creditor_payment_cents > 0]
    assert all(p >= 5000 for p in payments[6:])


# ===================================================================
# round_half_up
# ===================================================================

class TestRoundHalfUp:
    def test_half_rounds_up(self):
        assert round_half_up(0.5) == 1

    def test_half_rounds_up_not_even(self):
        # Python's round(2.5) == 2 (banker's rounding), ours gives 3
        assert round_half_up(2.5) == 3

    def test_below_half(self):
        assert round_half_up(2.4) == 2

    def test_above_half(self):
        assert round_half_up(2.6) == 3

    def test_exact_integer(self):
        assert round_half_up(5.0) == 5

    def test_negative_half(self):
        # -0.5 + 0.5 = 0.0, floor(0) = 0 — rounds toward zero
        assert round_half_up(-0.5) == 0


# ===================================================================
# Even shape tests
# ===================================================================

class TestEvenPayments:
    def test_evenly_divisible(self):
        rules = _make_rules(min_payment_cents=100, max_token_pays=6)
        payments = generate_even_payments(6000, 3, rules)
        assert payments == [2000, 2000, 2000]
        assert sum(payments) == 6000

    def test_remainder_on_latest(self):
        rules = _make_rules(min_payment_cents=100, max_token_pays=6)
        payments = generate_even_payments(10000, 3, rules)
        # 10000 / 3 = 3333 r 1 -> [3333, 3333, 3334]
        assert payments == [3333, 3333, 3334]
        assert sum(payments) == 10000

    def test_remainder_two(self):
        rules = _make_rules(min_payment_cents=100, max_token_pays=6)
        payments = generate_even_payments(10001, 3, rules)
        # 10001 / 3 = 3333 r 2 -> [3333, 3334, 3334]
        assert payments == [3333, 3334, 3334]
        assert sum(payments) == 10001

    def test_non_decreasing(self):
        rules = _make_rules(min_payment_cents=100, max_token_pays=10)
        payments = generate_even_payments(10007, 4, rules)
        assert payments is not None
        for i in range(1, len(payments)):
            assert payments[i] >= payments[i - 1]

    def test_below_floor_returns_none(self):
        rules = _make_rules(min_payment_cents=5000, max_token_pays=6)
        # 6000 / 3 = 2000, which is below floor of 5000
        payments = generate_even_payments(6000, 3, rules)
        assert payments is None

    def test_exact_sum(self):
        rules = _make_rules(min_payment_cents=100, max_token_pays=12)
        for k in range(1, 8):
            payments = generate_even_payments(50000, k, rules)
            assert payments is not None
            assert sum(payments) == 50000

    def test_case1_even_schedule_sums_to_offer_total(self):
        r = _run("case1_feasible_even")
        payments = [row.creditor_payment_cents for row in r.schedule]
        assert sum(payments) == 50000  # 50% of 100000


# ===================================================================
# Balloon shape tests
# ===================================================================

class TestBalloonPayments:
    def test_basic_balloon(self):
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=6,
            is_ballooning_allowed=True,
        )
        payments = generate_balloon_payments(30000, 6, rules)
        assert payments is not None
        # First 5 at floor, final absorbs rest
        assert payments[:5] == [2500] * 5
        assert payments[5] == 30000 - 5 * 2500  # 17500
        assert sum(payments) == 30000

    def test_balloon_non_decreasing(self):
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=10,
            is_ballooning_allowed=True,
        )
        payments = generate_balloon_payments(50000, 6, rules)
        assert payments is not None
        for i in range(1, len(payments)):
            assert payments[i] >= payments[i - 1]

    def test_not_allowed_returns_none(self):
        rules = _make_rules(is_ballooning_allowed=False)
        payments = generate_balloon_payments(30000, 6, rules)
        assert payments is None

    def test_balloon_final_less_than_prev_returns_none(self):
        rules = _make_rules(
            min_payment_cents=10000, max_token_pays=6,
            is_ballooning_allowed=True,
        )
        # 5 payments of 10000 = 50000, final = 20000 - 50000 < 0
        payments = generate_balloon_payments(20000, 6, rules)
        assert payments is None

    def test_balloon_exact_sum(self):
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=10,
            is_ballooning_allowed=True,
        )
        payments = generate_balloon_payments(25000, 5, rules)
        assert payments is not None
        assert sum(payments) == 25000

    def test_case3_balloon_payments_sum(self):
        r = _run("case3_balloon")
        payments = [row.creditor_payment_cents for row in r.schedule]
        assert sum(payments) == 30000  # 50% of 60000


# ===================================================================
# Staircase shape tests
# ===================================================================

class TestStaircasePayments:
    def test_basic_staircase_two_segments(self):
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=6, max_segments=2,
        )
        payments = generate_staircase_payments(60000, 12, rules)
        assert payments is not None
        assert len(set(payments)) <= 2
        assert sum(payments) == 60000

    def test_non_decreasing(self):
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=4, max_segments=3,
        )
        payments = generate_staircase_payments(50000, 8, rules)
        assert payments is not None
        for i in range(1, len(payments)):
            assert payments[i] >= payments[i - 1]

    def test_max_segments_cap_respected(self):
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=12, max_segments=2,
        )
        payments = generate_staircase_payments(50000, 10, rules)
        assert payments is not None
        assert len(set(payments)) <= 2

    def test_max_segments_one_is_flat(self):
        """max_segments=1 means all payments must be the same level."""
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=12, max_segments=1,
        )
        payments = generate_staircase_payments(30000, 6, rules)
        assert payments is not None
        assert len(set(payments)) == 1
        assert sum(payments) == 30000

    def test_exact_sum(self):
        rules = _make_rules(
            min_payment_cents=500, max_token_pays=10, max_segments=3,
        )
        payments = generate_staircase_payments(77777, 7, rules)
        assert payments is not None
        assert sum(payments) == 77777

    def test_front_loaded_prefers_low_early(self):
        """Staircase should keep early payments as low as possible."""
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=6, max_segments=2,
        )
        payments = generate_staircase_payments(40000, 8, rules)
        assert payments is not None
        # First payments should be at the floor
        assert payments[0] == 1000

    def test_case4_staircase_exact_sum(self):
        r = _run("case4_tiers")
        payments = [row.creditor_payment_cents for row in r.schedule
                    if row.creditor_payment_cents > 0]
        assert sum(payments) == 60000  # 40% of 150000


# ===================================================================
# Token pay and tier floor tests
# ===================================================================

class TestTokenPaysAndTiers:
    def test_token_pay_limit_enforced(self):
        """When max_token_pays is reached, subsequent payments must exceed base min."""
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=2, max_segments=3,
        )
        # With 5 payments and only 2 allowed at 2500, payments 3+ must be > 2500
        payments = generate_staircase_payments(30000, 5, rules)
        assert payments is not None
        token_count = sum(1 for p in payments if p == 2500)
        assert token_count <= 2

    def test_compute_floor_base(self):
        rules = _make_rules(min_payment_cents=2500, max_token_pays=6)
        assert compute_floor(1, rules, 0) == 2500
        assert compute_floor(5, rules, 0) == 2500

    def test_compute_floor_token_exhausted(self):
        rules = _make_rules(min_payment_cents=2500, max_token_pays=3)
        # After 3 token pays used, floor must exceed base min
        assert compute_floor(4, rules, 3) == 2501

    def test_compute_floor_tier_override(self):
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=6,
            min_payment_tiers=[(5, 5000)],
        )
        assert compute_floor(4, rules, 0) == 2500
        assert compute_floor(5, rules, 0) == 5000
        assert compute_floor(8, rules, 0) == 5000

    def test_compute_floor_tier_and_token(self):
        """Tier floor supersedes the token-pay +1 rule when tier is higher."""
        rules = _make_rules(
            min_payment_cents=2500, max_token_pays=2,
            min_payment_tiers=[(3, 5000)],
        )
        # Position 3 with 2 tokens used: tier floor = 5000, token +1 = 2501
        # Max is 5000
        assert compute_floor(3, rules, 2) == 5000

    def test_tier_floor_respected_in_staircase(self):
        """Tier floors must be respected in generated staircase payments."""
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=6, max_segments=2,
            min_payment_tiers=[(4, 3000)],
        )
        payments = generate_staircase_payments(30000, 6, rules)
        assert payments is not None
        for i in range(3, 6):  # positions 4-6 (0-indexed 3-5)
            assert payments[i] >= 3000

    def test_tier_floor_in_even_pays(self):
        """Even pays must still satisfy tier floors; returns None if they can't."""
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=6,
            even_pays=True,
            min_payment_tiers=[(2, 50000)],  # tier forces huge minimum from pos 2
        )
        # Even pays at 5000 each, but pos 2+ needs 50000 — impossible since
        # even means all equal, and floor at pos 2 is 50000 while total/k < 50000
        payments = generate_even_payments(15000, 3, rules)
        assert payments is None


# ===================================================================
# Date-by-date simulation & same-day ordering
# ===================================================================

class TestSimulationAndOrdering:
    def test_balance_never_negative(self):
        """All 4 provided cases must produce non-negative balances."""
        for case in ["case1_feasible_even", "case3_balloon", "case4_tiers"]:
            r = _run(case)
            assert r.schedule is not None, f"Expected feasible for {case}"
            for row in r.schedule:
                assert row.balance_cents >= 0, (
                    f"Negative balance on {row.date} in {case}: {row.balance_cents}"
                )

    def test_balance_hits_exactly_zero(self):
        """Case 1: first two months should end at exactly $0 balance."""
        r = _run("case1_feasible_even")
        assert r.schedule[0].balance_cents == 0
        assert r.schedule[1].balance_cents == 0

    def test_credits_before_debits_same_day(self):
        """When a draft and payment land on the same day, credits apply first.

        Build a scenario where the draft and payment are on the same day. Without
        credits-first ordering the balance would go negative.
        """
        client = _make_client(
            draft_amount=10000, first_draft="2026-01-31", last_draft="2026-03-31",
            draft_day=31,
            ledger=[
                {"date": "2026-01-31", "amount_cents": 10000, "type": "credit"},
                {"date": "2026-02-28", "amount_cents": 10000, "type": "credit"},
                {"date": "2026-03-31", "amount_cents": 10000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=20000, original_balance=20000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        rules = _make_rules(
            max_payments=3, max_terms=3,
            min_payment_cents=1000, max_token_pays=3,
            even_pays=True, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is True
        # Without credits-before-debits, Jan 31 would start at 0 and debit would fail
        assert all(row.balance_cents >= 0 for row in r.schedule)

    def test_case3_survives_mid_month_debit(self):
        """Case 3 has a $150 committed debit on Feb 1. Balloon shape defers payments
        so the account survives that hit."""
        r = _run("case3_balloon")
        assert r.feasible is True
        assert r.schedule is not None
        # Feb 28 balance should be >= 0 even after the Feb 1 debit
        feb_row = [row for row in r.schedule if row.date.month == 2][0]
        assert feb_row.balance_cents >= 0


# ===================================================================
# Horizon limit tests
# ===================================================================

class TestHorizon:
    def test_no_payments_past_horizon(self):
        """All schedule dates must be <= last_draft_date."""
        for case in ["case1_feasible_even", "case3_balloon", "case4_tiers"]:
            client, _, _ = load_case(f"cases/{case}")
            r = _run(case)
            if r.schedule:
                for row in r.schedule:
                    assert row.date <= client.last_draft_date, (
                        f"Schedule date {row.date} exceeds horizon "
                        f"{client.last_draft_date} in {case}"
                    )

    def test_tight_horizon_limits_k(self):
        """When horizon only allows 2 cadence dates, k should be capped at 2."""
        client = _make_client(
            draft_amount=20000, first_draft="2026-01-01", last_draft="2026-02-28",
            ledger=[
                {"date": "2026-01-01", "amount_cents": 20000, "type": "credit"},
                {"date": "2026-02-01", "amount_cents": 20000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=20000, original_balance=20000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        rules = _make_rules(
            max_payments=12, max_terms=12,
            min_payment_cents=1000, max_token_pays=12,
            even_pays=True, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is True
        assert len(r.schedule) <= 2


# ===================================================================
# Fee compliance tests
# ===================================================================

class TestFeeCompliance:
    def test_no_fee_before_first_payment(self):
        """Program fee must not appear before the first creditor payment date."""
        for case in ["case1_feasible_even", "case4_tiers"]:
            r = _run(case)
            if r.schedule:
                first_creditor_date = None
                for row in r.schedule:
                    if row.creditor_payment_cents > 0:
                        first_creditor_date = row.date
                        break
                for row in r.schedule:
                    if row.date < first_creditor_date:
                        assert row.program_fee_cents == 0, (
                            f"Fee {row.program_fee_cents} collected before first "
                            f"payment date {first_creditor_date} in {case}"
                        )

    def test_total_fee_matches_formula(self):
        """Total program fee collected must equal round_half_up(pct * original_balance)."""
        client, offer, rules = load_case("cases/case1_feasible_even")
        expected_fee = round_half_up(rules.program_fee_pct * offer.original_balance_cents)
        r = evaluate_offer(client, offer, rules)
        actual_fee = sum(row.program_fee_cents for row in r.schedule)
        assert actual_fee == expected_fee

    def test_fee_fully_collected_within_horizon(self):
        """All fee must be collected on or before the horizon date."""
        for case in ["case1_feasible_even", "case4_tiers"]:
            client, offer, rules = load_case(f"cases/{case}")
            r = evaluate_offer(client, offer, rules)
            if r.schedule:
                for row in r.schedule:
                    if row.program_fee_cents > 0:
                        assert row.date <= client.last_draft_date

    def test_zero_fee_works(self):
        """Case 3 has 0% program fee — engine should handle this correctly."""
        r = _run("case3_balloon")
        assert r.schedule is not None
        total_fee = sum(row.program_fee_cents for row in r.schedule)
        assert total_fee == 0

    def test_fee_front_loaded(self):
        """Fee should be front-loaded: collected as early as possible."""
        r = _run("case1_feasible_even")
        fees = [row.program_fee_cents for row in r.schedule]
        # Fee should be concentrated in early months (non-zero early, zero later)
        assert fees[0] > 0, "Fee should be collected on the first date"
        # Later months should have zero or less fee
        assert fees[-1] <= fees[0], "Fee should not increase over time"

    def test_no_bank_fee_on_fee_only_dates(self):
        """Fee-only dates (no creditor payment) must not have a bank fee."""
        for case in ["case1_feasible_even", "case4_tiers"]:
            r = _run(case)
            if r.schedule:
                for row in r.schedule:
                    if row.creditor_payment_cents == 0:
                        assert row.bank_fee_cents == 0, (
                            f"Bank fee {row.bank_fee_cents} on fee-only date "
                            f"{row.date} in {case}"
                        )

    def test_bank_fee_on_every_creditor_payment_date(self):
        """Every date with a creditor payment must have a bank fee (if bank_fee > 0)."""
        # Case 1 has bank_fee_cents = 1000
        r = _run("case1_feasible_even")
        for row in r.schedule:
            if row.creditor_payment_cents > 0:
                assert row.bank_fee_cents == 1000


# ===================================================================
# Exact sum constraint
# ===================================================================

class TestExactSum:
    def test_creditor_payments_sum_to_offer_total_all_cases(self):
        """Creditor payments must sum exactly to offer_total in all feasible cases."""
        feasible_cases = [
            ("case1_feasible_even", 50000),   # 0.5 * 100000
            ("case3_balloon", 30000),          # 0.5 * 60000
            ("case4_tiers", 60000),            # 0.4 * 150000
        ]
        for case, expected_total in feasible_cases:
            r = _run(case)
            payments = [row.creditor_payment_cents for row in r.schedule]
            assert sum(payments) == expected_total, (
                f"Payments sum {sum(payments)} != offer total {expected_total} in {case}"
            )

    def test_exact_sum_with_odd_amounts(self):
        """Payments must sum exactly even with non-divisible amounts."""
        rules = _make_rules(min_payment_cents=100, max_token_pays=12, even_pays=True)
        # 10001 / 3 -> not evenly divisible
        payments = generate_even_payments(10001, 3, rules)
        assert payments is not None
        assert sum(payments) == 10001


# ===================================================================
# Part 2: Minimum additional funds
# ===================================================================

class TestAdditionalFunds:
    def test_lump_sum_makes_infeasible_feasible(self):
        """Adding the reported lump sum should make the case feasible."""
        client, offer, rules = load_case("cases/case2_infeasible_minima")
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is False

        lump = r.additional_funds.lump_sum
        extra = [LedgerEntry(lump.date, lump.amount_cents, "credit")]
        r2 = find_feasible_schedule(client, offer, rules, extra)
        assert r2 is not None
        assert r2.feasible is True

    def test_monthly_increment_makes_infeasible_feasible(self):
        """Adding the reported monthly increment should make the case feasible."""
        client, offer, rules = load_case("cases/case2_infeasible_minima")
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is False

        mi = r.additional_funds.monthly_increment
        future_credits = [e for e in client.ledger
                          if e.date > client.as_of_date and e.type == "credit"]
        extra = [LedgerEntry(e.date, mi.amount_cents, "credit") for e in future_credits]
        r2 = find_feasible_schedule(client, offer, rules, extra)
        assert r2 is not None
        assert r2.feasible is True

    def test_lump_minus_one_is_infeasible(self):
        """Lump sum minus 1 cent should still be infeasible (true minimum)."""
        client, offer, rules = load_case("cases/case2_infeasible_minima")
        r = evaluate_offer(client, offer, rules)
        lump = r.additional_funds.lump_sum

        extra = [LedgerEntry(lump.date, lump.amount_cents - 1, "credit")]
        r2 = find_feasible_schedule(client, offer, rules, extra)
        assert r2 is None

    def test_monthly_minus_one_is_infeasible(self):
        """Monthly increment minus 1 cent should still be infeasible."""
        client, offer, rules = load_case("cases/case2_infeasible_minima")
        r = evaluate_offer(client, offer, rules)
        mi = r.additional_funds.monthly_increment

        future_credits = [e for e in client.ledger
                          if e.date > client.as_of_date and e.type == "credit"]
        extra = [LedgerEntry(e.date, mi.amount_cents - 1, "credit") for e in future_credits]
        r2 = find_feasible_schedule(client, offer, rules, extra)
        assert r2 is None

    def test_guardrail_lump_sum(self):
        """Lump sum guardrail: reject if L > 65% of offer_total."""
        client = _make_client(
            draft_amount=1000, first_draft="2026-01-01", last_draft="2026-03-01",
            ledger=[
                {"date": "2026-01-01", "amount_cents": 1000, "type": "credit"},
                {"date": "2026-02-01", "amount_cents": 1000, "type": "credit"},
                {"date": "2026-03-01", "amount_cents": 1000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=100000, original_balance=100000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        rules = _make_rules(
            max_payments=3, max_terms=3,
            min_payment_cents=1000, max_token_pays=3,
            max_segments=3, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r = evaluate_offer(client, offer, rules)
        # offer_total = 50000, 65% = 32500
        # With only 3000 in drafts vs 50000 needed, lump must be huge
        if r.feasible is False:
            lump = r.additional_funds.lump_sum
            if lump.amount_cents > round_half_up(0.65 * 50000):
                assert lump.within_guardrail is False

    def test_guardrail_monthly_increment(self):
        """Monthly increment guardrail: reject if X > max(10000, 40% of draft)."""
        r = _run("case2_infeasible_minima")
        mi = r.additional_funds.monthly_increment
        client, _, _ = load_case("cases/case2_infeasible_minima")
        limit = max(10000, round_half_up(0.40 * client.draft_amount_cents))
        if mi.amount_cents <= limit:
            assert mi.within_guardrail is True
        else:
            assert mi.within_guardrail is False

    def test_num_drafts_counts_all_future(self):
        """num_drafts should reflect ALL future drafts, not just those before last payment."""
        r = _run("case2_infeasible_minima")
        mi = r.additional_funds.monthly_increment
        assert mi.num_drafts == 5  # all 5 drafts are after as_of_date


# ===================================================================
# Edge cases & integration
# ===================================================================

class TestEdgeCases:
    def test_single_payment(self):
        """k=1: single lump creditor payment."""
        client = _make_client(
            draft_amount=50000, first_draft="2026-01-01", last_draft="2026-02-01",
            ledger=[
                {"date": "2026-01-01", "amount_cents": 50000, "type": "credit"},
                {"date": "2026-02-01", "amount_cents": 50000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=30000, original_balance=30000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        rules = _make_rules(
            max_payments=1, max_terms=1,
            min_payment_cents=1000, max_token_pays=1,
            even_pays=True, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is True
        assert len(r.schedule) == 1
        assert r.schedule[0].creditor_payment_cents == 15000

    def test_offer_total_with_rounding(self):
        """Offer total should use round-half-up, not Python's round."""
        offer = _make_offer(creditor_balance=10001, settlement_pct=0.5)
        # 10001 * 0.5 = 5000.5, round-half-up = 5001
        assert compute_offer_total(offer) == 5001

    def test_program_fee_with_rounding(self):
        """Program fee should use round-half-up."""
        offer = _make_offer(original_balance=10001)
        rules = _make_rules(program_fee_pct=0.5)
        # 10001 * 0.5 = 5000.5, round-half-up = 5001
        assert compute_program_fee(offer, rules) == 5001

    def test_result_serialization(self):
        """Result.to_dict() should produce valid JSON-serializable output."""
        r = _run("case1_feasible_even")
        d = r.to_dict()
        assert d["feasible"] is True
        assert d["pay_shape_used"] == "even"
        assert isinstance(d["schedule"], list)
        assert d["additional_funds"] is None
        # Check a row has all expected keys
        row = d["schedule"][0]
        assert set(row.keys()) == {
            "date", "creditor_payment_cents", "program_fee_cents",
            "bank_fee_cents", "balance_cents",
        }

    def test_infeasible_result_serialization(self):
        """Infeasible result serialization has the right structure."""
        r = _run("case2_infeasible_minima")
        d = r.to_dict()
        assert d["feasible"] is False
        assert d["schedule"] is None
        af = d["additional_funds"]
        assert "lump_sum" in af
        assert "monthly_increment" in af
        assert "amount_cents" in af["lump_sum"]
        assert "date" in af["lump_sum"]
        assert "num_drafts" in af["monthly_increment"]

    def test_large_bank_fee_reduces_feasibility(self):
        """High bank fees should make previously feasible cases harder."""
        client = _make_client(
            draft_amount=5000, first_draft="2026-01-01", last_draft="2026-03-01",
            ledger=[
                {"date": "2026-01-01", "amount_cents": 5000, "type": "credit"},
                {"date": "2026-02-01", "amount_cents": 5000, "type": "credit"},
                {"date": "2026-03-01", "amount_cents": 5000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=10000, original_balance=10000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        # With low bank fee: feasible
        rules_low = _make_rules(
            max_payments=3, max_terms=3,
            min_payment_cents=1000, max_token_pays=3,
            even_pays=True, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r_low = evaluate_offer(client, offer, rules_low)
        assert r_low.feasible is True

        # With high bank fee: may become infeasible
        rules_high = _make_rules(
            max_payments=3, max_terms=3,
            min_payment_cents=1000, max_token_pays=3,
            even_pays=True, bank_fee_cents=4000, program_fee_pct=0.0,
        )
        r_high = evaluate_offer(client, offer, rules_high)
        # 5000 per month, payment ~1667 + bank_fee 4000 = 5667 > 5000 → infeasible
        assert r_high.feasible is False

    def test_committed_debit_respected(self):
        """Committed debits in ledger must be respected and never modified."""
        client = _make_client(
            draft_amount=10000, first_draft="2026-01-01", last_draft="2026-04-01",
            ledger=[
                {"date": "2026-01-01", "amount_cents": 10000, "type": "credit"},
                {"date": "2026-01-15", "amount_cents": 8000, "type": "debit"},
                {"date": "2026-02-01", "amount_cents": 10000, "type": "credit"},
                {"date": "2026-03-01", "amount_cents": 10000, "type": "credit"},
                {"date": "2026-04-01", "amount_cents": 10000, "type": "credit"},
            ],
        )
        offer = _make_offer(
            creditor_balance=10000, original_balance=10000,
            settlement_pct=0.5, first_payment_date="2026-01-31",
        )
        rules = _make_rules(
            max_payments=4, max_terms=4,
            min_payment_cents=500, max_token_pays=4,
            even_pays=True, bank_fee_cents=0, program_fee_pct=0.0,
        )
        r = evaluate_offer(client, offer, rules)
        assert r.feasible is True
        # Balance after Jan 31 should account for the 8000 debit on Jan 15
        # Jan 1: +10000 → 10000; Jan 15: -8000 → 2000; Jan 31: -1250 → 750
        assert r.schedule[0].balance_cents >= 0

    def test_multiple_tiers(self):
        """Multiple tier step-ups should all be respected."""
        rules = _make_rules(
            min_payment_cents=1000, max_token_pays=12, max_segments=3,
            min_payment_tiers=[(3, 2000), (6, 4000)],
        )
        payments = generate_staircase_payments(50000, 8, rules)
        assert payments is not None
        # Positions 1-2: >= 1000
        for p in payments[:2]:
            assert p >= 1000
        # Positions 3-5: >= 2000
        for p in payments[2:5]:
            assert p >= 2000
        # Positions 6-8: >= 4000
        for p in payments[5:]:
            assert p >= 4000

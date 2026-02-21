"""
Integration tests for gemini_handler.
These tests make real API calls to Gemini and validate the response structure and values.

Run with: pytest src/tests/test_gemini_handler.py -v
"""

import pytest
from src.llm.gemini_handler import process_message
from dotenv import load_dotenv

load_dotenv()


class TestLogExpense:
    """Tests for log_expense function detection."""

    def test_spent_money_at_place(self):
        """Most common format: 'Spent $X at Place'."""
        result = process_message("V said: Dropped $65 at Trader Joe's")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 65.0
        assert "Trader" in result["args"]["name"] or "Joe" in result["args"]["name"]

        assert result["args"]["v_paid"] == 65.0
        assert result["args"]["y_paid"] == 0.0
        assert result["args"]["share"] == 0.5

        # Labels: Just verify structure, don't assert content
        assert isinstance(result["args"].get("labels", []), list)

    def test_minimal_log_format(self):
        """Minimal format: '$X at Place'."""
        result = process_message("Y said: $30 at Starbucks")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 30.0
        assert "Starbucks" in result["args"]["name"]
        assert result["args"]["y_paid"] == 30.0
        assert result["args"]["v_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_amount_without_dollar_sign(self):
        """Handle amount without dollar sign."""
        result = process_message("V said: Spent 25 dollars at Walmart")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 25.0
        assert "Walmart" in result["args"]["name"]
        assert result["args"]["v_paid"] == 25.0
        assert result["args"]["y_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_decimal_amounts(self):
        """Precise decimal handling."""
        result = process_message("Y said: $45.99 at Target")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 45.99
        assert "Target" in result["args"]["name"]
        assert result["args"]["y_paid"] == 45.99
        assert result["args"]["v_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_split_validation(self):
        """Share must be a float between 0 and 1."""
        result = process_message("V said: Spent $75.50 at pharmacy")

        assert result["function"] == "log_expense"
        share = result["args"]["share"]

        assert isinstance(share, (int, float))
        assert 0 <= share <= 1, f"Share out of range: {share}"

    def test_large_amount_rent(self):
        """Large numbers with explicit split."""
        result = process_message("Y said: Spent $2500 at rent, split equally")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 2500.0
        assert "rent" in result["args"]["name"].lower()
        assert result["args"]["y_paid"] == 2500.0
        assert result["args"]["v_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_unequal_split_single_payer(self):
        """One person paid full amount, split unequally (e.g., 60/40)."""
        result = process_message("V said: $100 at groceries, I owe 60, Y owes 40")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 100.0
        assert "groceries" in result["args"]["name"].lower()
        # V paid everything
        assert result["args"]["v_paid"] == 100.0
        assert result["args"]["y_paid"] == 0.0
        # V's share is 60% of $100
        assert result["args"]["share"] == 0.6

    def test_both_paid_balanced(self):
        """Both paid unequal amounts, but each paid exactly what they owe (no balance)."""
        result = process_message("Y said: We went to Chipotle, V covered $35 and I got the rest which was $25")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 60.0
        assert "chipotle" in result["args"]["name"].lower()
        # Both paid their shares
        assert result["args"]["v_paid"] == 35.0
        assert result["args"]["y_paid"] == 25.0
        # Both paid separately, so share = 0 (nothing owed between them)
        assert result["args"]["share"] == 0.0

    def test_put_it_all_on_y(self):
        """'Put it all on Y' means V's share is 0 — Y bears the full cost."""
        result = process_message("V said: $60 at pharmacy, put it all on Y")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 60.0
        assert result["args"]["share"] == 0.0

    def test_expense_with_explicit_date(self):
        """Handle explicit date: 'on January 10'."""
        result = process_message("V said: Spent $20 at coffee shop on January 10")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 20.0
        # Date should be January 10 of the current year
        date_str = result["args"]["date"]
        assert "01-10" in date_str or "01/10" in date_str

    def test_expense_with_relative_date_yesterday(self):
        """Handle relative date: 'yesterday'."""
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        result = process_message("Y said: Spent $15 at lunch yesterday")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 15.0
        assert yesterday in result["args"]["date"]


class TestGetBalance:
    """Tests for get_balance function detection."""

    def test_how_much_do_i_owe(self):
        """User's exact question: 'how much do I owe?'"""
        result = process_message("Y said: how much do I owe?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}

    def test_how_much_does_v_owe(self):
        """User's exact question: 'how much does V owe?'"""
        result = process_message("V said: how much does V owe?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}

    def test_how_much_does_y_owe(self):
        """User's exact question: 'how much does y owe?'"""
        result = process_message("Y said: how much does y owe?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}

    def test_what_is_the_balance(self):
        """Informal balance check phrasing."""
        result = process_message("V said: are we square or does someone owe money?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}

    def test_who_owes_whom(self):
        """User's exact question: 'who owes whom?'"""
        result = process_message("Y said: who owes whom?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}


class TestQueryRows:
    """Tests for query_rows function detection - listing individual expenses."""

    def test_last_expense_at_merchant(self):
        """User pattern: 'what is our last expense at place x?'"""
        result = process_message("Y said: what is our last expense at Costco?")

        assert result["function"] == "query_rows"
        assert result["args"]["limit"] == 1

        conditions = result["args"]["conditions"]
        assert len(conditions) == 1
        assert conditions[0]["column"] == "Name"
        assert "Costco" in conditions[0]["value"]
        assert conditions[0]["operation"] == "=="

    def test_list_all_transactions_at_merchant(self):
        """User pattern: 'list all transactions at place z'."""
        result = process_message("V said: list all transactions at Walmart")

        assert result["function"] == "query_rows"

        conditions = result["args"]["conditions"]
        assert len(conditions) == 1
        assert conditions[0]["column"] == "Name"
        assert "Walmart" in conditions[0]["value"]

    def test_list_last_n_transactions(self):
        """User pattern: 'list last n transactions'."""
        result = process_message("Y said: list last 5 transactions")

        assert result["function"] == "query_rows"
        assert result["args"]["limit"] == 5
        # No merchant filter, so conditions should be empty
        assert result["args"]["conditions"] == []

    def test_list_last_n_at_merchant(self):
        """User pattern: 'list last n transactions at place b'."""
        result = process_message("V said: list last 3 transactions at Starbucks")

        assert result["function"] == "query_rows"
        assert result["args"]["limit"] == 3

        conditions = result["args"]["conditions"]
        assert len(conditions) == 1
        assert conditions[0]["column"] == "Name"
        assert "Starbucks" in conditions[0]["value"]

    def test_show_recent_expenses(self):
        """Generic request to show recent expenses."""
        result = process_message("Y said: show me recent expenses")

        assert result["function"] == "query_rows"
        # Should have a limit (likely 5-10)
        assert result["args"].get("limit") is not None
        assert result["args"]["limit"] > 0


class TestQueryGroupedAgg:
    """Tests for query_grouped_agg function detection - aggregations and totals."""

    def test_how_much_spent_at_merchant(self):
        """User pattern: 'how much did we spend at place y?'"""
        result = process_message("V said: how much did we spend at Target?")

        assert result["function"] == "query_grouped_agg"

        # Should have aggregation summing Amount
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) == 1
        assert aggregations[0]["column"] == "Amount"
        assert aggregations[0]["function"] == "sum"

        # Should filter by merchant name
        conditions = result["args"]["conditions"]
        assert len(conditions) >= 1
        # Find the Name condition
        name_condition = next((c for c in conditions if c["column"] == "Name"), None)
        assert name_condition is not None
        assert "Target" in name_condition["value"]

    def test_total_spending(self):
        """Query for total spending across all expenses."""
        result = process_message("Y said: what's our total expenditure so far?")

        assert result["function"] == "query_grouped_agg"

        # Should have aggregation summing Amount
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) == 1
        assert aggregations[0]["column"] == "Amount"
        assert aggregations[0]["function"] == "sum"

        # No filters, so conditions should be empty
        assert result["args"]["conditions"] == []

    def test_how_much_paid_this_month(self):
        """User pattern: 'how much did we pay this month?'"""
        result = process_message("V said: how much did we pay this month?")

        assert result["function"] == "query_grouped_agg"

        # Should aggregate Amount
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) >= 1
        assert any(agg["column"] == "Amount" and agg["function"] == "sum" for agg in aggregations)

        # Should filter by current month
        conditions = result["args"]["conditions"]
        assert len(conditions) >= 1
        # Should have a Date condition with month transform
        date_condition = next((c for c in conditions if c["column"] == "Date"), None)
        assert date_condition is not None
        # Could use transform or direct value - just verify Date is involved

    def test_who_owed_what_in_march(self):
        """User pattern: 'break down what each person owed in March'."""
        result = process_message("Y said: break down what each person owed in March")

        assert result["function"] == "query_grouped_agg"

        # Should filter by March
        conditions = result["args"]["conditions"]
        assert len(conditions) >= 1
        # Should have condition for March
        march_condition = next(
            (c for c in conditions if isinstance(c.get("value"), str) and "march" in c["value"].lower()),
            None
        )
        assert march_condition is not None

        # Should aggregate owes columns or have grouping
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) >= 1

    def test_how_much_v_paid(self):
        """Query for how much V paid (distinct from 'how much does V owe?')."""
        result = process_message("V said: how much did V pay?")

        assert result["function"] == "query_grouped_agg"

        # Should aggregate v_paid column
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) >= 1
        assert any(agg["column"] == "v_paid" and agg["function"] == "sum" for agg in aggregations)

    def test_category_spending(self):
        """Query for spending in a category."""
        result = process_message("Y said: what have we been dropping on dining out?")

        assert result["function"] == "query_grouped_agg"

        # Should aggregate Amount
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) >= 1
        assert any(agg["column"] == "Amount" for agg in aggregations)

        # Should filter by dining label
        conditions = result["args"]["conditions"]
        assert len(conditions) >= 1
        # Should have Labels condition
        labels_condition = next((c for c in conditions if c["column"] == "Labels"), None)
        assert labels_condition is not None
        assert "dining" in str(labels_condition["value"]).lower()

    def test_monthly_spending_breakdown(self):
        """Query with group_by for monthly breakdown."""
        result = process_message("V said: give me a month by month summary of what we spent")

        assert result["function"] == "query_grouped_agg"

        # Should group by month
        group_by = result["args"]["group_by"]
        assert len(group_by) >= 1
        assert any("month" in g.lower() for g in group_by)

        # Should aggregate Amount
        aggregations = result["args"]["aggregations"]
        assert len(aggregations) >= 1
        assert any(agg["column"] == "Amount" and agg["function"] == "sum" for agg in aggregations)

    def test_top_n_categories(self):
        """Top N categories should use order_by_agg_index and limit."""
        result = process_message("Y said: where does most of our money go? show me the top 5")

        assert result["function"] == "query_grouped_agg"

        # Should group by Labels
        group_by = result["args"]["group_by"]
        assert "Labels" in group_by

        # Should aggregate Amount with sum
        aggregations = result["args"]["aggregations"]
        assert any(agg["column"] == "Amount" and agg["function"] == "sum" for agg in aggregations)

        # Should sort descending and limit to 5
        assert result["args"]["order_by_agg_index"] is not None
        assert result["args"]["order_desc"] is True
        assert result["args"]["limit"] == 5

    def test_group_by_date_month_value(self):
        """Grouping by month should use 'Date.month' in group_by."""
        result = process_message("V said: how much did we spend each month this year?")

        assert result["function"] == "query_grouped_agg"

        group_by = result["args"]["group_by"]
        assert "Date.month" in group_by


class TestSettleBalance:
    """Tests for settle_balance function detection."""

    def test_we_settled_up(self):
        """Settle up with novel phrasing."""
        result = process_message("V said: we're even now, mark it as settled")

        assert result["function"] == "settle_balance"
        assert result["args"] == {}


class TestDeleteExpense:
    """Tests for delete_expense function detection."""

    def test_delete_last_expense(self):
        """Delete most recent expense."""
        result = process_message("Y said: delete my last expense")

        assert result["function"] == "delete_expense"
        assert result["args"]["delete_mode"] == "last"

    def test_delete_by_merchant(self):
        """Delete expense at specific merchant."""
        result = process_message("V said: remove that Walmart charge")

        assert result["function"] == "delete_expense"
        assert result["args"]["delete_mode"] == "by_merchant"
        assert "Walmart" in result["args"]["merchant"]

    def test_delete_by_date(self):
        """Delete expense from a specific date."""
        result = process_message("Y said: remove the expense from yesterday")

        assert result["function"] == "delete_expense"
        assert result["args"]["delete_mode"] == "by_date"
        assert "date" in result["args"]


class TestEditExpense:
    """Tests for edit_expense function detection."""

    def test_edit_amount(self):
        """Edit expense amount."""
        result = process_message("Y said: actually that was $92, fix it")

        assert result["function"] == "edit_expense"
        assert result["args"]["new_amount"] == 92.0

    def test_edit_split(self):
        """Edit expense split (only owes fields change)."""
        result = process_message("V said: make it 70/30 instead")

        assert result["function"] == "edit_expense"
        assert "new_v_paid" in result["args"]
        assert "new_y_paid" in result["args"]

    def test_edit_merchant(self):
        """Edit merchant name only."""
        result = process_message("Y said: change the merchant to Trader Joe's")

        assert result["function"] == "edit_expense"
        assert "new_merchant" in result["args"]
        assert "trader" in result["args"]["new_merchant"].lower() or "joe" in result["args"]["new_merchant"].lower()


class TestClarify:
    """Tests for clarify function - missing or ambiguous info."""

    def test_missing_amount(self):
        """Should ask for amount when not provided."""
        result = process_message("V said: picked up stuff from the store")

        assert result["function"] == "clarify"
        assert "amount" in result["args"]["missing_fields"]

    def test_greeting(self):
        """Should clarify intent for greetings."""
        result = process_message("Y said: hello")

        assert result["function"] == "clarify"
        assert "message" in result["args"]

    def test_ambiguous(self):
        """Should clarify when intent is unclear."""
        result = process_message("V said: groceries")

        assert result["function"] == "clarify"


class TestResponseStructure:
    """Tests for correct response structure."""

    def test_has_required_fields(self):
        """All responses should have reasoning, function, and args."""
        result = process_message("Y said: Spent $10 at cafe")

        assert "reasoning" in result
        assert "function" in result
        assert "args" in result

class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_decimal_amount(self):
        """Handle decimal amounts correctly."""
        result = process_message("Y said: $45.99 at CVS")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 45.99
        assert "CVS" in result["args"]["name"]

        assert result["args"]["y_paid"] == 45.99
        assert result["args"]["v_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_large_amount(self):
        """Handle large amounts."""
        result = process_message("V said: $2500 at rent")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 2500.0
        assert result["args"]["v_paid"] == 2500.0
        assert result["args"]["y_paid"] == 0.0
        assert result["args"]["share"] == 0.5

    def test_share_is_valid(self):
        """Share must be a float between 0 and 1 for all log_expense results."""
        test_cases = [
            "V said: Spent $50 at store",
            "Y said: $75.50 at pharmacy",
            "V said: $2500 at rent",
            "Y said: $45.99 at Target"
        ]

        for msg in test_cases:
            result = process_message(msg)
            if result["function"] == "log_expense":
                share = result["args"]["share"]

                assert isinstance(share, (int, float)), \
                    f"Share is not numeric for '{msg}': {share}"
                assert 0 <= share <= 1, \
                    f"Share out of range for '{msg}': {share}"

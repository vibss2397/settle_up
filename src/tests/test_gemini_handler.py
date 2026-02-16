"""
Integration tests for gemini_handler.
These tests make real API calls to Gemini and validate the response structure and values.

Run with: pytest src/tests/test_gemini_handler.py -v
"""

import pytest
from src.llm.gemini_handler import process_message


class TestLogExpense:
    """Tests for log_expense function detection."""

    def test_simple_expense_v_default(self):
        """Default behavior is 50:50 split."""
        result = process_message("Spent $50 at Costco")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 50.0
        assert result["args"]["v_paid"] == 25
        assert result["args"]["y_paid"] == 25
        assert "Costco" in result["args"]["name"]

    def test_expense_with_label(self):
        """Expense with explicit category."""
        result = process_message("$25 on groceries at Walmart")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 25.0
        assert "groceries" in [l.lower() for l in result["args"].get("labels", [])]

    def test_y_paid_full(self):
        """y pays 100% when explicitly stated."""
        result = process_message("y paid $40 for gas, all on y")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 40.0
        assert result["args"]["v_paid"] == 0.0
        assert result["args"]["y_paid"] == 40.0

    def test_split_fifty_fifty(self):
        """50/50 split when 'split' mentioned."""
        result = process_message("$100 dinner, split it")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 100.0
        assert result["args"]["v_paid"] == 50.0
        assert result["args"]["y_paid"] == 50.0

    def test_explicit_split_amounts(self):
        """Honor explicit split amounts."""
        result = process_message("Dinner was $50, v paid $30 and y paid $20")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 50.0
        assert result["args"]["v_paid"] == 30.0
        assert result["args"]["y_paid"] == 20.0

    def test_share_sum_equals_amount(self):
        """Validate v_paid + y_paid == amount."""
        result = process_message("Spent $75.50 at Target, split equally")

        assert result["function"] == "log_expense"
        amount = result["args"]["amount"]
        v = result["args"]["v_paid"]
        y = result["args"]["y_paid"]
        assert abs((v + y) - amount) < 0.01, f"Shares don't sum to amount: {v} + {y} != {amount}"


class TestClarify:
    """Tests for clarify function - missing or ambiguous info."""

    def test_missing_amount(self):
        """Should ask for amount when not provided."""
        result = process_message("Bought groceries at Trader Joe's")

        assert result["function"] == "clarify"
        assert "amount" in result["args"]["missing_fields"]

    def test_missing_all_details(self):
        """Should ask for details when nothing provided."""
        result = process_message("Add an expense")

        assert result["function"] == "clarify"
        assert len(result["args"]["missing_fields"]) > 0

    def test_greeting_not_expense(self):
        """Should clarify intent for greetings."""
        result = process_message("hello")

        assert result["function"] == "clarify"
        assert "message" in result["args"]

    def test_ambiguous_intent(self):
        """Should clarify when intent is unclear."""
        result = process_message("groceries")

        assert result["function"] == "clarify"


class TestQueryGroupedAgg:
    """Tests for query_grouped_agg function detection."""

    def test_total_spending_query(self):
        """Query for total spending."""
        result = process_message("How much did we spend in January?")
        print(result)

        assert result["function"] == "query_grouped_agg"
        # Should have aggregations with sum
        assert len(result["args"]["aggregations"]) > 0
        assert result["args"]["aggregations"][0]["function"] == "sum"
        # Should have a condition for January
        conditions = result["args"]["conditions"]
        assert any(c["value"].lower() == "january" for c in conditions if isinstance(c.get("value"), str))

    def test_category_spending_query(self):
        """Query for spending in a category."""
        result = process_message("How much on groceries?")

        assert result["function"] == "query_grouped_agg"
        assert len(result["args"]["aggregations"]) > 0
        conditions = result["args"]["conditions"]
        assert any("groceries" in str(c.get("value", "")).lower() for c in conditions)

    def test_combined_filters_query(self):
        """Query with multiple filters."""
        result = process_message("How much did we spend on dining in February?")

        assert result["function"] == "query_grouped_agg"
        conditions = result["args"]["conditions"]
        assert len(conditions) >= 2

    def test_v_spending_query(self):
        """Query for v's spending specifically."""
        result = process_message("How much did v spend?")

        assert result["function"] == "query_grouped_agg"
        # Should aggregate v_paid column
        assert any(agg["column"] == "v_paid" for agg in result["args"]["aggregations"])


class TestQueryRows:
    """Tests for query_rows function detection."""

    def test_show_recent_expenses(self):
        """Query to show recent expenses."""
        result = process_message("Show me the last 5 expenses")

        assert result["function"] == "query_rows"
        assert result["args"]["limit"] == 5

    def test_show_expenses_with_filter(self):
        """Query to show expenses with a filter."""
        result = process_message("Show me all grocery expenses")

        assert result["function"] == "query_rows"
        conditions = result["args"]["conditions"]
        assert any("groceries" in str(c.get("value", "")).lower() for c in conditions)

    def test_list_all_expenses(self):
        """Query to list all expenses."""
        result = process_message("List all expenses")

        assert result["function"] == "query_rows"


class TestGetBalance:
    """Tests for get_balance function detection."""

    def test_whats_the_balance(self):
        """Direct balance question."""
        result = process_message("What's the balance?")

        assert result["function"] == "get_balance"
        assert result["args"] == {}

    def test_who_owes_whom(self):
        """Who owes whom question."""
        result = process_message("Who owes whom?")

        assert result["function"] == "get_balance"

    def test_how_much_do_i_owe(self):
        """How much do I owe question."""
        result = process_message("How much do I owe?")

        assert result["function"] == "get_balance"

    def test_are_we_even(self):
        """Are we even question."""
        result = process_message("Are we even?")

        assert result["function"] == "get_balance"


class TestSettleBalance:
    """Tests for settle_balance function detection."""

    def test_we_settled_up(self):
        """Direct settle up statement."""
        result = process_message("We settled up")

        assert result["function"] == "settle_balance"
        assert result["args"] == {}

    def test_paid_back(self):
        """Paid back statement."""
        result = process_message("I paid back y")

        assert result["function"] == "settle_balance"

    def test_cleared_balance(self):
        """Cleared balance statement."""
        result = process_message("Cleared the balance")

        assert result["function"] == "settle_balance"


class TestResponseStructure:
    """Tests for correct response structure."""

    def test_has_required_fields(self):
        """Response should have reasoning, function, args."""
        result = process_message("Spent $10 at cafe")

        assert "reasoning" in result
        assert "function" in result
        assert "args" in result

    def test_function_is_valid(self):
        """Function should be one of the allowed values."""
        valid_functions = {"log_expense", "query_grouped_agg", "query_rows", "get_balance", "settle_balance", "clarify"}

        test_messages = [
            "Spent $50 at store",
            "How much in January?",
            "Show last 3 expenses",
            "What's the balance?",
            "We settled up",
            "hello",
        ]

        for msg in test_messages:
            result = process_message(msg)
            assert result["function"] in valid_functions, f"Invalid function '{result['function']}' for message: {msg}"

    def test_condition_has_all_fields(self):
        """Conditions should have all required fields."""
        result = process_message("How much did we spend on groceries?")

        if result["function"] == "query_grouped_agg" and result["args"].get("conditions"):
            for condition in result["args"]["conditions"]:
                assert "column" in condition
                assert "value" in condition
                assert "operation" in condition
                assert "is_inverse" in condition
                assert "since_last_settle_up" in condition


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_decimal_amount(self):
        """Handle decimal amounts correctly."""
        result = process_message("Spent $45.99 at pharmacy")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 45.99

    def test_no_dollar_sign(self):
        """Handle amount without dollar sign."""
        result = process_message("Spent 30 dollars on lunch")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 30.0

    def test_large_amount(self):
        """Handle large amounts."""
        result = process_message("Rent was $2500, split equally")

        assert result["function"] == "log_expense"
        assert result["args"]["amount"] == 2500.0
        assert result["args"]["v_paid"] == 1250.0
        assert result["args"]["y_paid"] == 1250.0

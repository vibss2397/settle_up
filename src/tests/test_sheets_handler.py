"""
Integration tests for sheets_handler.
These tests make real API calls to Google Sheets.

Prerequisites:
- Set SHEET_NAME env var to a test sheet
- Ensure service_account.json is present

Run with: pytest src/tests/test_sheets_handler.py -v
"""

import pytest
from datetime import datetime
from pathlib import Path
from src.llm.sheets_handler import SheetsHandler, Condition

# Build absolute path: tests/ -> src/ -> creds/service_account.json
CREDS_PATH = Path(__file__).parent.parent / "creds" / "service_account.json"
SHEET_NAME="Test Expenses"
SPREADSHEET_NAME="Settle_up"

@pytest.fixture(scope="module")
def handler():
    """Create a SheetsHandler instance for all tests in this module."""
    return SheetsHandler(spreadsheet_name=SPREADSHEET_NAME, sheet_name=SHEET_NAME, credentials_path=str(CREDS_PATH))


class TestLogExpense:
    """Tests for log_expense method."""

    def test_log_basic_expense(self, handler):
        """Log expense with required fields, verify return structure."""
        result = handler.log_expense(
            date=datetime.now(),
            name="Test Merchant",
            amount=50.0,
            v_paid=25.0,
            y_paid=25.0,
            split=0.5,
            notes=None,
            labels=[],
        )

        assert result["merchant"] == "Test Merchant"
        assert result["amount"] == 50.0
        assert result["v_paid"] == 25.0
        assert result["y_paid"] == 25.0
        assert result["v_owes"] == 25.0
        assert result["y_owes"] == 25.0

    def test_log_expense_v_pays_all_split_equal(self, handler):
        """V pays full amount, split 50/50."""
        result = handler.log_expense(
            date=datetime.now(),
            name="Grocery Store",
            amount=75.0,
            v_paid=75.0,
            y_paid=0,
            split=0.5,
            notes=None,
            labels=["groceries", "food"],
        )

        assert result["labels"] == ["groceries", "food"]
        assert result["v_paid"] == 75.0
        assert result["y_paid"] == 0
        assert result["v_owes"] == 37.5
        assert result["y_owes"] == 37.5

    def test_log_expense_y_pays_all_unequal_split(self, handler):
        """Y pays full amount, unequal split (V=30%, Y=70%)."""
        result = handler.log_expense(
            date=datetime.now(),
            name="Restaurant",
            amount=60.0,
            v_paid=0,
            y_paid=60.0,
            split=0.3,
            notes="Birthday dinner",
            labels=["dining"],
        )

        assert result["notes"] == "Birthday dinner"
        assert result["v_paid"] == 0
        assert result["y_paid"] == 60.0
        assert result["v_owes"] == 18.0
        assert result["y_owes"] == 42.0

    def test_log_expense_both_pay_no_one_owes(self, handler):
        """Both pay exactly what they owe, no balance created."""
        result = handler.log_expense(
            date=datetime.now(),
            name="Shared Cab",
            amount=40.0,
            v_paid=15.0,
            y_paid=25.0,
            split=0.375,
            notes="Airport ride",
            labels=["transport"],
        )

        assert result["v_paid"] == result["v_owes"]
        assert result["y_paid"] == result["y_owes"]

    def test_expense_appears_in_query(self, handler):
        """Log expense and verify it appears in query results."""
        unique_name = f"TestExpense_{datetime.now().timestamp()}"
        handler.log_expense(
            date=datetime.now(),
            name=unique_name,
            amount=99.99,
            v_paid=99.99,
            y_paid=0,
            split=0.3,
            notes=None,
            labels=["test"],
        )

        result = handler.query_rows(
            conditions=[Condition(
                column="Name",
                value=unique_name,
                operation="==",
                is_inverse=False,
                since_last_settle_up=False,
            )],
            limit=1,
        )

        assert result["record_count"] >= 1
        assert any(r.get("Name") == unique_name for r in result["rows"])


class TestQueryGroupedAgg:
    """Tests for query_grouped_agg method."""

    def test_query_total_all(self, handler):
        """Query with no conditions returns valid structure."""
        result = handler.query_grouped_agg(
            conditions=[],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "sum"}],
        )

        assert "results" in result
        assert "filters_applied" in result
        assert "record_count" in result
        assert isinstance(result["record_count"], int)
        # Results should have sum_Amount
        if result["results"]:
            assert "sum_Amount" in result["results"][0]

    def test_query_with_label_filter(self, handler):
        """Filter by Labels contains."""
        result = handler.query_grouped_agg(
            conditions=[Condition(
                column="Labels",
                value="test",
                operation="contains",
                is_inverse=False,
                since_last_settle_up=False,
            )],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "sum"}],
        )

        assert "results" in result
        assert len(result["filters_applied"]) == 1

    def test_query_v_paid_column(self, handler):
        """Query aggregating v_paid."""
        result = handler.query_grouped_agg(
            conditions=[],
            group_by=[],
            aggregations=[{"column": "v_paid", "function": "sum"}],
        )

        assert "results" in result
        if result["results"]:
            assert "sum_v_paid" in result["results"][0]
            assert isinstance(result["results"][0]["sum_v_paid"], (int, float))

    def test_query_y_paid_column(self, handler):
        """Query aggregating y_paid."""
        result = handler.query_grouped_agg(
            conditions=[],
            group_by=[],
            aggregations=[{"column": "y_paid", "function": "sum"}],
        )

        assert "results" in result
        if result["results"]:
            assert "sum_y_paid" in result["results"][0]
            assert isinstance(result["results"][0]["sum_y_paid"], (int, float))

    def test_query_combined_conditions(self, handler):
        """Multiple conditions applied with AND logic."""
        result = handler.query_grouped_agg(
            conditions=[
                Condition(
                    column="Labels",
                    value="test",
                    operation="contains",
                    is_inverse=False,
                    since_last_settle_up=False,
                ),
                Condition(
                    column="Amount",
                    value=0,
                    operation=">",
                    is_inverse=False,
                    since_last_settle_up=False,
                ),
            ],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "sum"}],
        )

        assert len(result["filters_applied"]) == 2

    def test_query_with_group_by(self, handler):
        """Test grouping by Labels."""
        result = handler.query_grouped_agg(
            conditions=[],
            group_by=["Labels"],
            aggregations=[{"column": "Amount", "function": "sum"}],
        )

        assert "results" in result
        assert "group_by" in result
        assert result["group_by"] == ["Labels"]


class TestQueryRows:
    """Tests for query_rows method."""

    def test_query_rows_no_filter(self, handler):
        """Get all rows without filters."""
        result = handler.query_rows(conditions=[])

        assert "rows" in result
        assert "filters_applied" in result
        assert "record_count" in result
        assert isinstance(result["rows"], list)

    def test_query_rows_with_limit(self, handler):
        """Verify limit parameter works."""
        result = handler.query_rows(conditions=[], limit=3)

        assert len(result["rows"]) <= 3

    def test_query_rows_with_condition(self, handler):
        """Filter rows by condition."""
        result = handler.query_rows(
            conditions=[Condition(
                column="Labels",
                value="test",
                operation="contains",
                is_inverse=False,
                since_last_settle_up=False,
            )],
        )

        for row in result["rows"]:
            assert "test" in str(row.get("Labels", "")).lower()

    def test_query_rows_structure(self, handler):
        """Verify row structure has expected columns."""
        result = handler.query_rows(conditions=[], limit=1)

        if result["rows"]:
            row = result["rows"][0]
            # These should be the column headers in the sheet
            assert "Name" in row or len(row) > 0


class TestGetBalance:
    """Tests for get_balance method."""

    def test_get_balance_structure(self, handler):
        """Verify return dict has all expected keys."""
        result = handler.get_balance()

        assert "v_paid_total" in result
        assert "y_paid_total" in result
        assert "v_owes_total" in result
        assert "y_owes_total" in result
        assert "total" in result
        assert "amount_owed" in result
        assert "who_owes" in result

    def test_balance_values_are_numeric(self, handler):
        """Verify balance values are numbers."""
        result = handler.get_balance()

        assert isinstance(result["v_paid_total"], (int, float))
        assert isinstance(result["y_paid_total"], (int, float))
        assert isinstance(result["total"], (int, float))
        assert isinstance(result["amount_owed"], (int, float))

    def test_balance_math_is_correct(self, handler):
        """Verify total equals v + y and amount_owed is the net diff."""
        result = handler.get_balance()

        assert abs(result["total"] - (result["v_paid_total"] + result["y_paid_total"])) < 0.01
        v_net = result["v_paid_total"] - result["v_owes_total"]
        assert abs(result["amount_owed"] - abs(v_net)) < 0.01

    def test_who_owes_logic(self, handler):
        """Verify who_owes is correct based on net (paid - owes)."""
        result = handler.get_balance()

        v_net = result["v_paid_total"] - result["v_owes_total"]
        if v_net == 0:
            assert result["who_owes"] is None
        elif v_net > 0:
            assert result["who_owes"] == "y"
        else:
            assert result["who_owes"] == "v"


class TestSettleBalance:
    """Tests for settle_balance method."""

    def test_settle_returns_valid_structure(self, handler):
        """Verify return dict has expected keys."""
        result = handler.settle_balance()

        # Either settled successfully or no balance to settle
        if result.get("settled"):
            assert "payer" in result
            assert "payee" in result
            assert "amount" in result
        else:
            assert "message" in result


class TestConditionFiltering:
    """Tests for various condition operations."""

    def test_equals_operation(self, handler):
        """Test == operator."""
        # V pays all, unequal split (60/40)
        handler.log_expense(
            date=datetime.now(),
            name="EqualsTest",
            amount=100.0,
            v_paid=100.0,
            y_paid=0,
            split=0.6,
            notes=None,
            labels=[],
        )

        result = handler.query_rows(
            conditions=[Condition(
                column="Name",
                value="EqualsTest",
                operation="==",
                is_inverse=False,
                since_last_settle_up=False,
            )],
        )

        assert result["record_count"] >= 1

    def test_contains_operation_case_insensitive(self, handler):
        """Test contains operator is case-insensitive."""
        # Y pays all, split 50/50
        handler.log_expense(
            date=datetime.now(),
            name="ContainsTest",
            amount=25.0,
            v_paid=0,
            y_paid=25.0,
            split=0.5,
            notes=None,
            labels=["UPPERCASE"],
        )

        result = handler.query_rows(
            conditions=[Condition(
                column="Labels",
                value="uppercase",  # lowercase query
                operation="contains",
                is_inverse=False,
                since_last_settle_up=False,
            )],
        )

        assert result["record_count"] >= 1

    def test_greater_than_operation(self, handler):
        """Test > operator."""
        result = handler.query_rows(
            conditions=[Condition(
                column="Amount",
                value=50,
                operation=">",
                is_inverse=False,
                since_last_settle_up=False,
            )],
        )

        for row in result["rows"]:
            assert float(row.get("Amount", 0)) > 50

    def test_less_than_operation(self, handler):
        """Test < operator."""
        result = handler.query_rows(
            conditions=[Condition(
                column="Amount",
                value=1000,
                operation="<",
                is_inverse=False,
                since_last_settle_up=False,
            )],
        )

        for row in result["rows"]:
            assert float(row.get("Amount", 0)) < 1000

    def test_is_inverse_negates_condition(self, handler):
        """Test is_inverse=True negates the condition."""
        # Get count without inverse
        normal = handler.query_grouped_agg(
            conditions=[Condition(
                column="Labels",
                value="test",
                operation="contains",
                is_inverse=False,
                since_last_settle_up=False,
            )],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "count"}],
        )

        # Get count with inverse
        inverted = handler.query_grouped_agg(
            conditions=[Condition(
                column="Labels",
                value="test",
                operation="contains",
                is_inverse=True,
                since_last_settle_up=False,
            )],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "count"}],
        )

        # Total of both should equal all records
        all_records = handler.query_grouped_agg(
            conditions=[],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "count"}],
        )

        # The counts should be complementary (allowing for floating point)
        assert normal["record_count"] + inverted["record_count"] == all_records["record_count"]


class TestResponseStructure:
    """Tests for correct response structures across all methods."""

    def test_log_expense_returns_dict(self, handler):
        """log_expense should return a dict."""
        # Both pay unequally, each pays exactly their share
        result = handler.log_expense(
            date=datetime.now(),
            name="StructureTest",
            amount=10.0,
            v_paid=7.0,
            y_paid=3.0,
            split=0.7,
            notes=None,
            labels=[],
        )
        assert isinstance(result, dict)

    def test_query_grouped_agg_returns_dict(self, handler):
        """query_grouped_agg should return a dict."""
        result = handler.query_grouped_agg(
            conditions=[],
            group_by=[],
            aggregations=[{"column": "Amount", "function": "sum"}],
        )
        assert isinstance(result, dict)

    def test_query_rows_returns_dict(self, handler):
        """query_rows should return a dict."""
        result = handler.query_rows(conditions=[])
        assert isinstance(result, dict)

    def test_get_balance_returns_dict(self, handler):
        """get_balance should return a dict."""
        result = handler.get_balance()
        assert isinstance(result, dict)

    def test_settle_balance_returns_dict(self, handler):
        """settle_balance should return a dict."""
        result = handler.settle_balance()
        assert isinstance(result, dict)

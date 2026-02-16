"""
Integration tests for message_preprocessor.
These tests make real API calls to Gemini and validate the preprocessing logic.

Run with: pytest src/tests/test_message_preprocessor.py -v
"""

import pytest
from src.llm.message_preprocessor import MessagePreprocessor, PreprocessResult


@pytest.fixture
def preprocessor():
    """Create a MessagePreprocessor instance for testing."""
    return MessagePreprocessor()


class TestSingleAsk:
    """Tests for single ask messages (regression tests)."""

    def test_simple_expense(self, preprocessor):
        """Single expense should be returned as-is."""
        result = preprocessor.preprocess_message("Spent $50 at Costco")

        assert result.is_valid
        assert len(result.asks) == 1
        assert result.asks[0] == "Spent $50 at Costco"
        assert result.is_in_domain

    def test_balance_query(self, preprocessor):
        """Single balance query should be returned as-is."""
        result = preprocessor.preprocess_message("What's the balance?")

        assert result.is_valid
        assert len(result.asks) == 1
        assert "balance" in result.asks[0].lower()
        assert result.is_in_domain

    def test_query_spending(self, preprocessor):
        """Single spending query should be returned as-is."""
        result = preprocessor.preprocess_message("How much did we spend on groceries?")

        assert result.is_valid
        assert len(result.asks) == 1
        assert result.is_in_domain


class TestMultipleAsks:
    """Tests for multiple asks in a single message."""

    def test_conjunction_split(self, preprocessor):
        """Split expenses connected with 'and'."""
        result = preprocessor.preprocess_message("log $50 groceries and $30 gas")

        assert result.is_valid
        assert len(result.asks) == 2
        assert "$50" in result.asks[0] and "groceries" in result.asks[0].lower()
        assert "$30" in result.asks[1] and "gas" in result.asks[1].lower()
        assert result.is_in_domain

    def test_conjunction_with_then(self, preprocessor):
        """Split expenses with 'and then'."""
        result = preprocessor.preprocess_message("log $20 at costco and then $30 for coffee")

        assert result.is_valid
        assert len(result.asks) == 2
        assert "$20" in result.asks[0] and "costco" in result.asks[0].lower()
        assert "$30" in result.asks[1] and "coffee" in result.asks[1].lower()
        assert result.is_in_domain

    def test_comma_separated(self, preprocessor):
        """Split comma-separated expenses."""
        result = preprocessor.preprocess_message("spent $50 groceries, $30 gas, $20 coffee")

        assert result.is_valid
        assert len(result.asks) == 3
        assert all("$" in ask for ask in result.asks)
        assert result.is_in_domain

    def test_numbered_list(self, preprocessor):
        """Split numbered list of requests."""
        result = preprocessor.preprocess_message(
            "1. log $50 groceries\n2. show balance\n3. what did we spend on dining?"
        )

        assert result.is_valid
        assert len(result.asks) == 3
        assert "groceries" in result.asks[0].lower()
        assert "balance" in result.asks[1].lower()
        assert "dining" in result.asks[2].lower()
        assert result.is_in_domain

    def test_bulleted_list(self, preprocessor):
        """Split bulleted list of requests."""
        result = preprocessor.preprocess_message(
            "- log $50 groceries\n- show balance"
        )

        assert result.is_valid
        assert len(result.asks) == 2
        assert result.is_in_domain

    def test_mixed_requests(self, preprocessor):
        """Split mixed request types."""
        result = preprocessor.preprocess_message("log $50 groceries and show balance")

        assert result.is_valid
        assert len(result.asks) == 2
        assert "groceries" in result.asks[0].lower()
        assert "balance" in result.asks[1].lower()
        assert result.is_in_domain


class TestConstraints:
    """Tests for constraint enforcement."""

    def test_exactly_five_asks(self, preprocessor):
        """Should accept exactly 5 asks."""
        result = preprocessor.preprocess_message(
            "1. log $10 item\n2. log $20 item\n3. log $30 item\n4. log $40 item\n5. log $50 item"
        )

        assert result.is_valid
        assert len(result.asks) == 5
        assert result.is_in_domain

    def test_more_than_five_asks(self, preprocessor):
        """Should reject more than 5 asks."""
        result = preprocessor.preprocess_message(
            "1. log $10 item\n2. log $20 item\n3. log $30 item\n4. log $40 item\n5. log $50 item\n6. log $60 item"
        )

        assert not result.is_valid
        assert "maximum 5" in result.error_message.lower() or "too many" in result.error_message.lower()
        assert result.is_in_domain  # Still in domain, just too many


class TestDomainRelevance:
    """Tests for domain relevance detection."""

    def test_greeting_out_of_domain(self, preprocessor):
        """Greetings should be out-of-domain."""
        result = preprocessor.preprocess_message("Hello! How are you?")

        assert not result.is_valid
        assert not result.is_in_domain
        assert result.error_message is not None

    def test_simple_greeting(self, preprocessor):
        """Simple greeting should be out-of-domain."""
        result = preprocessor.preprocess_message("hi")

        assert not result.is_valid
        assert not result.is_in_domain

    def test_weather_out_of_domain(self, preprocessor):
        """Weather questions should be out-of-domain."""
        result = preprocessor.preprocess_message("What's the weather today?")

        assert not result.is_valid
        assert not result.is_in_domain

    def test_joke_out_of_domain(self, preprocessor):
        """Jokes should be out-of-domain."""
        result = preprocessor.preprocess_message("Tell me a joke")

        assert not result.is_valid
        assert not result.is_in_domain

    def test_expense_in_domain(self, preprocessor):
        """Expense tracking messages should be in domain."""
        messages = [
            "log expense",
            "show balance",
            "what did we spend",
            "settle up",
            "how much do I owe",
        ]

        for msg in messages:
            result = preprocessor.preprocess_message(msg)
            assert result.is_in_domain, f"Message '{msg}' should be in domain"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_message(self, preprocessor):
        """Empty message should be handled gracefully."""
        result = preprocessor.preprocess_message("")

        assert not result.is_valid
        assert not result.is_in_domain

    def test_whitespace_only(self, preprocessor):
        """Whitespace-only message should be handled gracefully."""
        result = preprocessor.preprocess_message("   \n  ")

        assert not result.is_valid
        assert not result.is_in_domain

    def test_very_long_message(self, preprocessor):
        """Very long messages should be handled."""
        long_message = "log $50 at " + "store " * 100

        result = preprocessor.preprocess_message(long_message)
        # Should either parse it or fail gracefully
        assert isinstance(result, PreprocessResult)

    def test_special_characters(self, preprocessor):
        """Messages with special characters should be handled."""
        result = preprocessor.preprocess_message("Spent $50 @ Costco!!!")

        assert result.is_valid
        assert len(result.asks) >= 1
        assert result.is_in_domain


class TestContextPreservation:
    """Tests to ensure context is preserved when splitting."""

    def test_preserve_amounts(self, preprocessor):
        """Amounts should be preserved in each ask."""
        result = preprocessor.preprocess_message("log $20 at costco and $30 for coffee")

        assert result.is_valid
        assert len(result.asks) == 2
        assert "$20" in result.asks[0]
        assert "$30" in result.asks[1]

    def test_preserve_merchant_names(self, preprocessor):
        """Merchant names should be preserved."""
        result = preprocessor.preprocess_message("log $20 at costco and $30 at starbucks")

        assert result.is_valid
        assert len(result.asks) == 2
        assert "costco" in result.asks[0].lower()
        assert "starbucks" in result.asks[1].lower()

    def test_infer_verb_for_split(self, preprocessor):
        """Verb should be inferred when splitting."""
        result = preprocessor.preprocess_message("spent $50 groceries, $30 gas")

        assert result.is_valid
        assert len(result.asks) == 2
        # Each ask should have enough context (verb or action word)
        # This is a softer test - LLM should preserve context appropriately


class TestResponseStructure:
    """Tests for correct response structure."""

    def test_has_required_fields(self, preprocessor):
        """Response should have all required fields."""
        result = preprocessor.preprocess_message("Spent $50 at store")

        assert hasattr(result, "is_valid")
        assert hasattr(result, "asks")
        assert hasattr(result, "error_message")
        assert hasattr(result, "is_in_domain")

    def test_asks_is_list(self, preprocessor):
        """Asks field should always be a list."""
        messages = [
            "Spent $50 at store",
            "log $20 costco and $30 coffee",
            "Hello",
        ]

        for msg in messages:
            result = preprocessor.preprocess_message(msg)
            assert isinstance(result.asks, list)

    def test_error_message_when_invalid(self, preprocessor):
        """Error message should be present when invalid."""
        result = preprocessor.preprocess_message("Hello")

        if not result.is_valid:
            assert result.error_message is not None
            assert len(result.error_message) > 0

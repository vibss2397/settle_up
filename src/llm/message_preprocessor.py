import os
from typing import Optional

import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# --- Data Models ---

class PreprocessResult(BaseModel):
    """Result of preprocessing a user message."""
    is_valid: bool
    asks: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    is_in_domain: bool = True


# --- Preprocessing Prompt ---

PREPROCESSING_PROMPT = """
You are a message preprocessor for an expense tracking bot.

Your job:
1. Check if the message is related to expense tracking (logging expenses, checking balance, querying spending, settling up)
2. Split messages containing multiple distinct requests into individual asks
3. Preserve all context needed for each ask

Domain: Expense tracking (log expense, check balance, query spending, settle up, delete expense)
Out-of-domain: Greetings, weather, jokes, general conversation

Rules:
- If message contains 1 request: return it as-is in the asks list
- If message contains multiple requests (2-5): split them into separate asks
- If message contains >5 requests: return error with is_valid=false
- If out-of-domain: return error with is_valid=false and is_in_domain=false
- Preserve amounts, names, and context when splitting
- When splitting expenses with "and" or commas, infer the verb (log/spent/bought/paid) for each
- IMPORTANT: If a date/time is mentioned ANYWHERE in the message (beginning, middle, or end), it applies to ALL expenses in that message. Append the date to EACH split ask.

Examples:
Input: "log $20 at costco and then $30 for coffee"
Output: {"is_valid": true, "asks": ["log $20 at costco", "log $30 for coffee"], "is_in_domain": true}

Input: "brunch for $30 and coffee for $10. On January 10"
Output: {"is_valid": true, "asks": ["brunch for $30 on January 10", "coffee for $10 on January 10"], "is_in_domain": true}

Input: "On January 10 we had brunch for $30 and coffee for $10"
Output: {"is_valid": true, "asks": ["brunch for $30 on January 10", "coffee for $10 on January 10"], "is_in_domain": true}

Input: "Yesterday spent $50 groceries and $30 gas"
Output: {"is_valid": true, "asks": ["spent $50 groceries yesterday", "spent $30 gas yesterday"], "is_in_domain": true}

Input: "spent $50 groceries, $30 gas, $20 coffee"
Output: {"is_valid": true, "asks": ["spent $50 groceries", "spent $30 gas", "spent $20 coffee"], "is_in_domain": true}

Input: "1. log $50 groceries\\n2. show balance\\n3. what did we spend on dining?"
Output: {"is_valid": true, "asks": ["log $50 groceries", "show balance", "what did we spend on dining?"], "is_in_domain": true}

Input: "log $50 groceries and show balance"
Output: {"is_valid": true, "asks": ["log $50 groceries", "show balance"], "is_in_domain": true}

Input: "Spent $50 at Costco"
Output: {"is_valid": true, "asks": ["Spent $50 at Costco"], "is_in_domain": true}

Input: "Hello! How are you?"
Output: {"is_valid": false, "is_in_domain": false, "error_message": "This is not related to expense tracking"}

Input: "What's the weather today?"
Output: {"is_valid": false, "is_in_domain": false, "error_message": "This is not related to expense tracking"}

Input: "1. log $10 item\\n2. log $20 item\\n3. log $30 item\\n4. log $40 item\\n5. log $50 item\\n6. log $60 item"
Output: {"is_valid": false, "error_message": "Too many requests. Maximum 5 allowed.", "is_in_domain": true}

Input: "delete my last expense"
Output: {"is_valid": true, "asks": ["delete my last expense"], "is_in_domain": true}

Input: "remove the costco expense"
Output: {"is_valid": true, "asks": ["remove the costco expense"], "is_in_domain": true}

Return JSON matching PreprocessResult schema exactly.
"""


# --- Preprocessor Class ---

class MessagePreprocessor:
    """Splits user messages into individual asks using LLM."""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def preprocess_message(self, message: str) -> PreprocessResult:
        """
        Split message into individual asks and validate.

        Uses Gemini LLM to:
        1. Detect if message is domain-relevant (expense tracking)
        2. Split into individual asks (max 5)
        3. Preserve context for each ask

        Args:
            message: Raw user message

        Returns:
            PreprocessResult with validation status and split asks

        Example:
            Input: "log $20 at costco and then $30 for coffee"
            Output: PreprocessResult(
                is_valid=True,
                asks=["log $20 at costco", "log $30 for coffee"],
                is_in_domain=True
            )
        """
        # Handle empty messages
        if not message or not message.strip():
            return PreprocessResult(
                is_valid=False,
                error_message="Empty message received",
                is_in_domain=False
            )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=PREPROCESSING_PROMPT,
                    response_mime_type="application/json",
                    response_schema=PreprocessResult,
                ),
                contents=message
            )

            parsed: PreprocessResult = response.parsed
            return parsed

        except Exception as e:
            print(f"Preprocessing Error: {e}")
            # Fallback: treat as single ask if preprocessing fails
            return PreprocessResult(
                is_valid=True,
                asks=[message],
                is_in_domain=True
            )


if __name__ == "__main__":
    # Quick test
    preprocessor = MessagePreprocessor()

    test_messages = [
        "log $20 at costco and then $30 for coffee",
        "Spent $50 at Costco",
        "Hello! How are you?",
        "1. log $50 groceries\n2. show balance",
    ]

    for msg in test_messages:
        print(f"\nInput: {msg}")
        result = preprocessor.preprocess_message(msg)
        print(f"Output: {result.model_dump()}")

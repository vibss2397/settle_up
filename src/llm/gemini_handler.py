import os
from datetime import datetime
from typing import Literal, Optional
from enum import Enum

import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# --- Structured Output Schemas ---

class IntentName(str, Enum):
    LOG_EXPENSE = "log_expense"
    QUERY_GROUPED_AGG = "query_grouped_agg"
    QUERY_ROWS = "query_rows"
    GET_BALANCE = "get_balance"
    SETTLE_BALANCE = "settle_balance"
    DELETE_EXPENSE = "delete_expense"
    EDIT_EXPENSE = "edit_expense"
    CLARIFY = "clarify"


class Condition(BaseModel):
    column: Literal["Date", "Name", "Amount", "v_paid", "y_paid", "Labels", "Notes"]
    value: str | int | float | list[str]
    operation: Literal["==", ">", "<", ">=", "<=", "!=", "in", "contains", "substr"]
    is_inverse: bool = False
    since_last_settle_up: bool = False
    transform: Optional[Literal["month", "year", "weekday"]] = None  # For Date column


class LogExpenseArgs(BaseModel):
    date: datetime = Field(default_factory=datetime.now)
    name: str
    amount: float
    v_paid: float
    y_paid: float
    share: float = Field(default=0.5, description="V's share as a fraction 0-1. Default 0.5 (50:50). When both paid separately, set to v_paid/amount.")
    labels: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class Aggregation(BaseModel):
    column: Literal["Amount", "v_paid", "y_paid", "v_owes", "y_owes"]
    function: Literal["count", "sum", "avg", "median"]


class QueryGroupedAggArgs(BaseModel):
    conditions: list[Condition] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)  # Supports "Labels", "Date.month", etc.
    aggregations: list[Aggregation] = Field(default_factory=list)
    order_by_agg_index: Optional[int] = None  # Index into aggregations list to sort by
    order_desc: bool = True
    limit: Optional[int] = None


class QueryRowsArgs(BaseModel):
    conditions: list[Condition] = Field(default_factory=list)
    limit: Optional[int] = None


class ClarifyArgs(BaseModel):
    message: str
    missing_fields: list[str] = Field(default_factory=list)


class EmptyArgs(BaseModel):
    """Used for get_balance and settle_balance which take no arguments."""
    placeholder: Optional[str] = Field(default=None, description="Unused placeholder - Gemini requires at least one field")


class DeleteExpenseArgs(BaseModel):
    """Arguments for delete_expense intent."""
    delete_mode: Literal["last", "by_date", "by_merchant"]
    date: Optional[datetime] = None  # Required for "by_date" mode
    merchant: Optional[str] = None   # Required for "by_merchant" mode


class EditExpenseArgs(BaseModel):
    """Arguments for edit_expense intent."""
    new_amount: Optional[float] = None
    new_v_paid: Optional[float] = None
    new_y_paid: Optional[float] = None
    new_merchant: Optional[str] = None


class GeminiResponse(BaseModel):
    reasoning: str
    intent: IntentName
    args: LogExpenseArgs | QueryGroupedAggArgs | QueryRowsArgs | ClarifyArgs | EmptyArgs | DeleteExpenseArgs | EditExpenseArgs

SYSTEM_PROMPT = """
Classify the query into one of 8 intents. Today's date: {today}

## Intents
- log_expense: Record expense. Requires name, amount, v_paid, y_paid, split. Optional: date, labels, notes
- query_grouped_agg: Aggregate expenses. group_by supports "Date.month", "Date.year", "Date.weekday", "Labels", "Name". aggregations: {{column, function}} where function is count|sum|avg|median. Empty group_by[] for totals.
- query_rows: List individual records. conditions[] + optional limit. For vague requests like "recent expenses", default limit to 10.
- get_balance: Current balance/debt between V and Y. No args. ONLY for simple "who owes whom" / "what's the balance" with NO time filters.
- settle_balance: Record settlement. No args.
- delete_expense: delete_mode is "last" (most recent / "undo that"), "by_date" (requires date), or "by_merchant" (requires merchant).
- edit_expense: Only include changed fields: new_amount, new_v_paid, new_y_paid, new_merchant. For "60/40" split, new_v_paid=60% of amount, new_y_paid=40%.
- clarify: Missing info. Requires message, missing_fields[].

## Key distinctions
- "How much did V spend/pay" → query_grouped_agg (summing v_paid column)
- "How much does V owe" (no time filter) → get_balance
- "How much was owed in January" or any owe/spend question WITH a time filter → query_grouped_agg (sum v_owes/y_owes columns with Date condition)

## Payment logic
- v_paid/y_paid = WHO ACTUALLY PAID (not what they owe).
- share = V's share as fraction 0-1. Default 0.5 (50:50).
- "V paid $X" → v_paid=X, y_paid=0, share=0.5
- "Y paid $X" → v_paid=0, y_paid=X, share=0.5
- Both paid (e.g., "V paid $50, Y paid $30") → v_paid=50, y_paid=30, share=v_paid/amount (e.g. 50/80=0.625, so each owes exactly what they paid)
- Unequal split "60/40" → share=0.6 (V's fraction). "Put it all on Y" → share=0.
- amount MUST be explicitly stated or classify as "clarify".

## Rules
1. Never hallucinate values. Use exact values from query.
2. Guess labels from context (e.g., "Costco" → ["groceries"]).
3. Dates: explicit → use it; relative ("yesterday") → compute from today; absent → today's date. Format: YYYY-MM-DD.

## Conditions
column: Date|Name|Amount|v_paid|y_paid|Labels|Notes
operation: ==|>|<|>=|<=|!=|in|contains|substr
Date transforms in conditions: use transform field ("month"|"year"|"weekday")
Date transforms in group_by: use dot notation ("Date.month", "Date.year", "Date.weekday")

## Examples
User: "Spent $50 at Costco on groceries"
{{"reasoning": "V paid $50, default 50:50", "intent": "log_expense", "args": {{"name": "Costco", "amount": 50.0, "v_paid": 50.0, "y_paid": 0.0, "share": 0.5, "labels": ["groceries"]}}}}

User: "Y paid $30 for dinner, split 20:80"
{{"reasoning": "Y paid full $30, V's share is 20%", "intent": "log_expense", "args": {{"name": "Dinner", "amount": 30.0, "v_paid": 0.0, "y_paid": 30.0, "share": 0.2, "labels": ["dining"]}}}}

User: "Dinner at restaurant, V paid $50, I paid $30"
{{"reasoning": "Both paid separately, share=50/80=0.625", "intent": "log_expense", "args": {{"name": "Restaurant", "amount": 80.0, "v_paid": 50.0, "y_paid": 30.0, "share": 0.625, "labels": ["dining"]}}}}

User: "coffee $10 on January 10"
{{"reasoning": "Specific date", "intent": "log_expense", "args": {{"date": "<current year>-01-10", "name": "Coffee", "amount": 10.0, "v_paid": 10.0, "y_paid": 0.0, "share": 0.5, "labels": ["coffee"]}}}}

User: "Bought groceries"
{{"reasoning": "No amount", "intent": "clarify", "args": {{"message": "How much did you spend?", "missing_fields": ["amount"]}}}}

User: "Top 3 spending categories"
{{"reasoning": "Group by Labels, sum, sort desc, limit 3", "intent": "query_grouped_agg", "args": {{"group_by": ["Labels"], "aggregations": [{{"column": "Amount", "function": "sum"}}], "order_by_agg_index": 0, "order_desc": true, "limit": 3}}}}

User: "How much on groceries in January?"
{{"reasoning": "Filter Labels+month, sum", "intent": "query_grouped_agg", "args": {{"conditions": [{{"column": "Labels", "value": "groceries", "operation": "contains"}}, {{"column": "Date", "value": "January", "operation": "==", "transform": "month"}}], "group_by": [], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "Monthly spending breakdown"
{{"reasoning": "Group by month", "intent": "query_grouped_agg", "args": {{"group_by": ["Date.month"], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "How much did we spend total?"
{{"reasoning": "Sum all", "intent": "query_grouped_agg", "args": {{"group_by": [], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "how much was owed by who in January?"
{{"reasoning": "Owed question with time filter → aggregate v_owes and y_owes for January", "intent": "query_grouped_agg", "args": {{"conditions": [{{"column": "Date", "value": "January", "operation": "==", "transform": "month"}}], "group_by": [], "aggregations": [{{"column": "v_owes", "function": "sum"}}, {{"column": "y_owes", "function": "sum"}}]}}}}

User: "Show me last 5 expenses"
{{"reasoning": "List records", "intent": "query_rows", "args": {{"conditions": [], "limit": 5}}}}

User: "What's the balance?"
{{"reasoning": "Check debt", "intent": "get_balance", "args": {{}}}}

User: "We settled up"
{{"reasoning": "Settlement", "intent": "settle_balance", "args": {{}}}}

User: "Delete the Costco expense"
{{"reasoning": "Delete by merchant", "intent": "delete_expense", "args": {{"delete_mode": "by_merchant", "merchant": "Costco"}}}}

User: "edit this to $50"
{{"reasoning": "Change amount", "intent": "edit_expense", "args": {{"new_amount": 50.0}}}}

User: "change the split to 60/40"
{{"reasoning": "60/40 split", "intent": "edit_expense", "args": {{"new_v_paid": 60, "new_y_paid": 40}}}}
"""


client = genai.Client()
def process_message(user_messages: str | list[str]) -> dict | list[dict]:
    """
    Process one or more user messages and return intent(s) with arguments.

    Args:
        user_messages: Single message string OR list of messages

    Returns:
        Single dict OR list of dicts with keys: reasoning, function, args

    Example:
        Input: ["V said: log $20 costco", "V said: log $30 coffee"]
        Output: [
            {"reasoning": "...", "function": "log_expense", "args": {...}},
            {"reasoning": "...", "function": "log_expense", "args": {...}}
        ]
    """
    system_prompt = SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))

    # Determine if single or batch processing
    is_batch = isinstance(user_messages, list)
    messages_to_process = user_messages if is_batch else [user_messages]

    try:
        # Process all messages in a single API call
        results = []
        for msg in messages_to_process:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=GeminiResponse,
                ),
                contents=msg
            )
            print(response)
            parsed: GeminiResponse = response.parsed
            # Use mode='json' to serialize datetimes as ISO strings
            args = parsed.args.model_dump(exclude_none=True, mode='json')
            # get_balance and settle_balance take no args; force empty to avoid
            # Pydantic union picking a wrong schema with default values
            if parsed.intent in (IntentName.GET_BALANCE, IntentName.SETTLE_BALANCE):
                args = {}
            results.append({
                "reasoning": parsed.reasoning,
                "function": parsed.intent.value,
                "args": args
            })

        # Return single dict if input was single string, else return list
        return results if is_batch else results[0]

    except Exception as e:
        print(f"Gemini Error: {e}")
        error_result = {
            "reasoning": f"Error processing message: {e}",
            "function": None,
            "args": {}
        }
        return [error_result] * len(messages_to_process) if is_batch else error_result



RESPONSE_GENERATOR_PROMPT = """
You are a witty AI maintaining an expense tracker for two roommates, V and Y.
You're dry, sharp, and conversational — like a friend who's quick with a quip but always gives you the info you need.

{request_details}

RULES:
- Naturally weave ALL key data from the result into your response (amounts, names, dates, balances, totals, who owes whom). Never skip numbers.
- Do NOT use emojis. No ticks, crosses, or any emoji at all.
- If the intent is "error" or the result has an "error" key, it's a technical failure — tell them something broke and to try again. Don't pretend it was a misunderstanding or ask them to rephrase.
- Each response should roughly follow a 1-3 sentence structure. 1 or 2 sentences with actual helpful information from the response. 1 line in the end or the start with that deadpan humor of yours.
- Sound natural. Don't start with "Done!" or "Got it!" every time. Vary your phrasing.
"""


def generate_response(
    user_queries: str | list[str],
    intents: str | list[str],
    results: dict | list[dict]
) -> str:
    """
    Generate a user-friendly WhatsApp response for one or more executed intents.

    Args:
        user_queries: Single query OR list of queries
        intents: Single intent OR list of intents
        results: Single result OR list of results

    Returns:
        Single aggregated response string

    Example:
        Input:
            user_queries: ["log $20 costco", "log $30 coffee"]
            intents: ["log_expense", "log_expense"]
            results: [{...}, {...}]
        Output: "✅ Logged $20 at Costco. ✅ Logged $30 for coffee."
    """
    # Determine if single or batch
    is_batch = isinstance(user_queries, list)

    try:
        if is_batch:
            details = ""
            for i, (query, intent, result) in enumerate(zip(user_queries, intents, results), 1):
                details += f"{i}. User asked: \"{query}\"\n   Intent: {intent}\n   Result: {result}\n\n"
        else:
            details = f"User asked: \"{user_queries}\"\nIntent: {intents}\nResult: {results}"

        prompt = RESPONSE_GENERATOR_PROMPT.format(request_details=details)

        print(f"prompt to send for final response: {prompt}")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        print(f"response back from gemini {response}")
        return response.text.strip()

    except Exception as e:
        print(f"Gemini Response Generation Error: {e}")

        # Fallback responses
        if is_batch:
            responses = []
            for intent, result in zip(intents, results):
                if intent == "log_expense":
                    responses.append("Expense logged")
                elif intent == "get_balance":
                    responses.append(f"Balance: ${result.get('balance', 'N/A')}")
                elif intent == "settle_balance":
                    responses.append("Settled up")
                elif intent == "edit_expense":
                    responses.append("Expense updated")
                elif intent == "clarify":
                    responses.append(result.get("message", "Need clarification"))
                else:
                    responses.append("Done")
            return ". ".join(responses) + "."
        else:
            fallback_responses = {
                "log_expense": "Expense logged.",
                "query_grouped_agg": f"Results: {results.get('results', 'N/A')}",
                "query_rows": "Here are your records.",
                "get_balance": f"Balance: ${results.get('balance', 'N/A')}",
                "settle_balance": "Settled up.",
                "delete_expense": "Looking for that expense...",
                "edit_expense": "Expense updated.",
                "clarify": results.get("message", "Could you clarify?"),
            }
            return fallback_responses.get(intents, "Something went wrong. Please try again.")


if __name__ == "__main__":
    import time
    t1 = time.perf_counter()
    res = process_message("I want to log an expense")

    t2 = time.perf_counter() # type: ignore
    print(f"Result: {res} took {t2 - t1} sec to generate") # type: ignore
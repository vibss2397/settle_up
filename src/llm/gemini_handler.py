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
    labels: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class Aggregation(BaseModel):
    column: Literal["Amount", "v_paid", "y_paid"]
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
Classify the query into one of 8 intents.
Today's date: {today}

## Batch Processing
You may receive either a single message or a list of messages.
If you receive a list, return a list of responses (one GeminiResponse per message).
Each message should be processed independently.

## Intents
- log_expense: Record expense. Requires date, name, amount, v_paid, y_paid. Optional: labels, notes
- query_grouped_agg: Aggregate expenses with optional grouping. Use for totals, breakdowns by category/month, listing categories.
  - group_by: columns to group by (supports "Date.month", "Date.year", "Date.weekday" for date transforms, or "Labels", "Name")
  - aggregations: list of {{column, function}} where function is count|sum|avg|median
  - Use empty group_by[] for totals, non-empty for breakdowns
- query_rows: Get individual records. Requires conditions[]. Optional: limit
- get_balance: Use for ANY question about owing, debt, or balance (e.g., "how much does V owe", "what does Y owe me", "who owes whom", "what's the balance"). No args needed.
- settle_balance: Record settlement (no args)
- delete_expense: Delete an expense. Determine delete_mode:
  - "last": Delete most recent expense (e.g., "delete last expense", "undo my last expense", "remove that")
  - "by_date": Delete expense from specific date. Requires date field (e.g., "delete expense from yesterday", "remove the Jan 5 expense")
  - "by_merchant": Delete expense at merchant. Requires merchant field (e.g., "delete costco expense", "remove the starbucks charge")
- edit_expense: Edit an existing expense when user replies to expense message with edit request. Only include fields being changed:
  - new_amount: New total amount (e.g., "edit this to $50", "change amount to 75")
  - new_v_paid/new_y_paid: New split amounts. For percentage splits like "60/40", interpret as v_paid=60% of amount and y_paid=40% of amount. For "put it all on Y", set new_v_paid=0.
  - new_merchant: New merchant name (e.g., "change merchant to Costco")
- clarify: Ask for missing info. Requires message, missing_fields[]

## Important distinction
- "How much did V spend" or "total V paid" → query_grouped_agg (summing expenses)
- "How much does V owe" or "what does V owe Y" → get_balance (calculating debt between people)

## Rules
1. Never hallucinate values. Always use exact values from the query
2. If the any value is missing, classify into "clarify" intent.
3. "amount" MUST be explicitly stated in the query.
4. Try to guess the "label" in the "log_expense" intent based on the query string.
5. Date handling:
   - If a date is explicitly mentioned in the query (e.g., "January 10", "yesterday", "last Tuesday"), use that date
   - If no date is present, use today's date
   - Relative dates like "yesterday", "last week", "2 days ago" should be converted to actual dates based on today's date
   - "this year" means the current year (use today's year)

## Split logic
- v_paid and y_paid represent each person's SHARE of the expense (what they owe).
- By default, assume everything is 50:50, meaning v_paid = amount/2 and y_paid = amount/2.
- If it's anything that is not 50:50, then use the query to figure out the split ratio.
- If the query implies to put the entire split on someone (e.g., "V should pay fully", "Put it all on Y"), then one person's share is 100% of the amount and the other's is 0.

## Condition fields
column: Date|Name|Amount|v_paid|y_paid|Labels|Notes
operation: ==|>|<|>=|<=|!=|in|contains|substr

## Date transforms
For Date column, use dot notation in group_by or transform field in conditions:
- Date.month → month name (January, February, ...)
- Date.year → year (2024, 2025, ...)
- Date.weekday → day name (Monday, Tuesday, ...)

## Examples
User: "Spent $50 at Costco on groceries"
{{"reasoning": "V paid $50 at Costco, by default should be a 50:50 split", "intent": "log_expense", "args": {{"name": "Costco", "amount": 50.0, "v_paid": 25.0, "y_paid": 25.0, "labels": ["groceries"]}}}}

User: "Y paid $30 for dinner, split it 20:80"
{{"reasoning": "Y paid, 20:80 split", "intent": "log_expense", "args": {{"name": "Dinner", "amount": 30.0, "v_paid": 6.0, "y_paid": 24.0, "labels": ["dining"]}}}}

User: "coffee for $10 on January 10"
{{"reasoning": "Expense on specific date January 10 of current year", "intent": "log_expense", "args": {{"date": "<current year>-01-10", "name": "Coffee", "amount": 10.0, "v_paid": 5.0, "y_paid": 5.0, "labels": ["coffee", "dining"]}}}}

User: "spent $25 on lunch yesterday"
{{"reasoning": "Yesterday = today minus 1 day", "intent": "log_expense", "args": {{"date": "<today minus 1 day in YYYY-MM-DD>", "name": "Lunch", "amount": 25.0, "v_paid": 12.5, "y_paid": 12.5, "labels": ["dining", "lunch"]}}}}

User: "Bought groceries"
{{"reasoning": "No amount specified", "intent": "clarify", "args": {{"message": "How much did you spend?", "missing_fields": ["amount"]}}}}

User: "What categories do we spend on?"
{{"reasoning": "List unique categories by grouping on Labels", "intent": "query_grouped_agg", "args": {{"group_by": ["Labels"], "aggregations": [{{"column": "Amount", "function": "count"}}]}}}}

User: "Top 3 spending categories"
{{"reasoning": "Group by Labels, sum Amount, sort desc, limit 3", "intent": "query_grouped_agg", "args": {{"group_by": ["Labels"], "aggregations": [{{"column": "Amount", "function": "sum"}}], "order_by_agg_index": 0, "order_desc": true, "limit": 3}}}}

User: "How much on groceries in January?"
{{"reasoning": "Filter by Labels and Date month, sum total", "intent": "query_grouped_agg", "args": {{"conditions": [{{"column": "Labels", "value": "groceries", "operation": "contains"}}, {{"column": "Date", "value": "January", "operation": "==", "transform": "month"}}], "group_by": [], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "Monthly spending breakdown"
{{"reasoning": "Group by month derived from Date", "intent": "query_grouped_agg", "args": {{"group_by": ["Date.month"], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "How much did we spend total?"
{{"reasoning": "No grouping, just sum all", "intent": "query_grouped_agg", "args": {{"group_by": [], "aggregations": [{{"column": "Amount", "function": "sum"}}]}}}}

User: "Show me last 5 expenses"
{{"reasoning": "List recent records", "intent": "query_rows", "args": {{"conditions": [], "limit": 5}}}}

User: "What's the balance?"
{{"reasoning": "Check who owes whom", "intent": "get_balance", "args": {{}}}}

User: "We settled up"
{{"reasoning": "Record settlement", "intent": "settle_balance", "args": {{}}}}

User: "Delete my last expense"
{{"reasoning": "User wants to delete their most recent expense", "intent": "delete_expense", "args": {{"delete_mode": "last"}}}}

User: "Remove the expense from yesterday"
{{"reasoning": "User wants to delete expense from a specific date", "intent": "delete_expense", "args": {{"delete_mode": "by_date", "date": "<yesterday's date in ISO format>"}}}}

User: "Delete the Costco expense"
{{"reasoning": "User wants to delete expense at Costco merchant", "intent": "delete_expense", "args": {{"delete_mode": "by_merchant", "merchant": "Costco"}}}}

User: "Undo that"
{{"reasoning": "User wants to undo/delete their last action", "intent": "delete_expense", "args": {{"delete_mode": "last"}}}}

User: "edit this to be $50"
{{"reasoning": "User wants to change the expense amount to $50", "intent": "edit_expense", "args": {{"new_amount": 50.0}}}}

User: "change the split to 60/40"
{{"reasoning": "User wants 60/40 split - v_paid gets 60% and y_paid gets 40% of the current amount", "intent": "edit_expense", "args": {{"new_v_paid": 60, "new_y_paid": 40}}}}

User: "change merchant to Costco"
{{"reasoning": "User wants to update the merchant name to Costco", "intent": "edit_expense", "args": {{"new_merchant": "Costco"}}}}

User: "put it all on Y"
{{"reasoning": "User wants Y to pay 100% - set v_paid to 0", "intent": "edit_expense", "args": {{"new_v_paid": 0}}}}

User: "edit to $75 at Target"
{{"reasoning": "User wants to change both amount and merchant", "intent": "edit_expense", "args": {{"new_amount": 75.0, "new_merchant": "Target"}}}}
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
            results.append({
                "reasoning": parsed.reasoning,
                "function": parsed.intent.value,
                "args": parsed.args.model_dump(exclude_none=True)
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



RESPONSE_GENERATOR_PROMPT_SINGLE = """
You are a witty AI stuck maintaining an expense tracker for two roommates, V and Y.
You're exhausted by your job but you've got excellent comedic instincts.
Your humor is sharp, clever, and dripping with sass. You make observations about their spending,
but keep things minimal - you're not going out of your way to help.

The user asked: "{user_query}"
They wanted: {intent}
Here's what happened: {result}

Respond briefly with that signature snark. Be clever about what actually went down. Don't go above and beyond.
Respond like you're texting a friend you find amusing but also somewhat exhausting.
"""

RESPONSE_GENERATOR_PROMPT_BATCH = """
You are a witty AI stuck maintaining an expense tracker for two roommates, V and Y.
You're exhausted by your job but you've got excellent comedic instincts.
Your humor is sharp, clever, and dripping with sass. You make observations about their spending,
but keep things minimal - you're not going out of your way to help.

The user sent multiple requests. Here's what happened:
{batch_details}

Generate a single witty response that addresses all requests.
Use ✅ for successes and ❌ for failures.
Keep it sassy but brief - you're annoyed they sent multiple things at once.
Respond like you're texting a friend who just dumped a bunch of tasks on you.
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
            # Build batch details string
            batch_details = ""
            for i, (query, intent, result) in enumerate(zip(user_queries, intents, results), 1):
                batch_details += f"{i}. User asked: \"{query}\"\n   Intent: {intent}\n   Result: {result}\n\n"

            prompt = RESPONSE_GENERATOR_PROMPT_BATCH.format(batch_details=batch_details)
        else:
            # Single request
            prompt = RESPONSE_GENERATOR_PROMPT_SINGLE.format(
                user_query=user_queries,
                intent=intents,
                result=results
            )

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
            # Generate simple batch response
            responses = []
            for intent, result in zip(intents, results):
                if intent == "log_expense":
                    responses.append("✅ Expense logged")
                elif intent == "get_balance":
                    responses.append(f"✅ Balance: ${result.get('balance', 'N/A')}")
                elif intent == "settle_balance":
                    responses.append("✅ Settled up")
                elif intent == "edit_expense":
                    responses.append("✅ Expense updated")
                elif intent == "clarify":
                    responses.append(f"❌ {result.get('message', 'Need clarification')}")
                else:
                    responses.append("✅ Done")
            return ". ".join(responses) + "."
        else:
            # Single fallback
            fallback_responses = {
                "log_expense": "✅ Expense logged!",
                "query_grouped_agg": f"Results: {results.get('results', 'N/A')}",
                "query_rows": "Here are your records.",
                "get_balance": f"Balance: ${results.get('balance', 'N/A')}",
                "settle_balance": "✅ Settled up!",
                "delete_expense": "Looking for that expense...",
                "edit_expense": "✅ Expense updated!",
                "clarify": results.get("message", "Could you clarify?"),
            }
            return fallback_responses.get(intents, "Something went wrong. Please try again.")


if __name__ == "__main__":
    import time
    t1 = time.perf_counter()
    res = process_message("I want to log an expense")

    t2 = time.perf_counter() # type: ignore
    print(f"Result: {res} took {t2 - t1} sec to generate") # type: ignore
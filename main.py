import os
import json
import hmac
import hashlib
import random
import string
from datetime import datetime

import functions_framework
import requests

from src.llm.gemini_handler import process_message, generate_response
from src.llm.message_preprocessor import MessagePreprocessor
from src.llm.sheets_handler import SheetsHandler, Condition

# --- Configuration ---
META_TOKEN = os.environ.get("META_TOKEN")
META_PHONE_ID = os.environ.get("META_PHONE_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
META_APP_SECRET = os.environ.get("META_APP_SECRET")
SHEET_NAME = os.environ.get("SHEET_NAME", "Test Expenses")
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "Settle_up")
CREDS_PATH = os.environ.get("CREDS_PATH", "src/creds/service_account.json")
PHONE_MAPPING_PATH = os.environ.get("PHONE_MAPPING_PATH", "src/mapping.json")

# Load phone number to person mapping
with open(PHONE_MAPPING_PATH) as f:
    PHONE_TO_PERSON: dict[str, str] = json.load(f)

# Initialize sheets handler
sheets = SheetsHandler(
    spreadsheet_name=SPREADSHEET_NAME,
    sheet_name=SHEET_NAME,
    credentials_path=CREDS_PATH
)


def get_person_from_phone(phone_number: str) -> str | None:
    """Map a phone number to a person identifier (V or Y).

    Returns None if the phone number is not recognized.
    """
    return PHONE_TO_PERSON.get(phone_number)


def verify_signature(request) -> bool:
    """Verify that the request is actually from Meta using HMAC signature."""
    if not META_APP_SECRET:
        # Skip validation if no secret configured (dev mode)
        return True

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        META_APP_SECRET.encode(),
        request.data,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature[7:], expected)


def reply_to_whatsapp(to_number: str, text: str) -> str | None:
    """Send a reply message via WhatsApp API.

    Returns:
        The message ID (wamid) if successful, None otherwise.
    """
    url = f"https://graph.facebook.com/v21.0/{META_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        print(f"Reply status: {r.status_code} - {r.text}")
        if r.status_code == 200:
            response_data = r.json()
            messages = response_data.get("messages", [])
            if messages:
                return messages[0].get("id")
        return None
    except Exception as e:
        print(f"Failed to reply: {e}")
        return None


def generate_short_code(length: int = 3) -> str:
    """Generate a random short code for delete confirmation."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def is_short_code_format(text: str) -> bool:
    """Check if text looks like a short confirmation code (3 lowercase letters)."""
    return len(text) == 3 and text.isalpha()


def is_edit_request(text: str) -> bool:
    """Check if text looks like an edit request."""
    edit_keywords = ["edit", "change", "update", "modify"]
    return any(kw in text.lower() for kw in edit_keywords)


def extract_reply_context(message: dict) -> str | None:
    """Extract the message ID being replied to, if this is a reply."""
    context = message.get("context")
    if context:
        return context.get("id")
    return None


def execute_intent(function_name: str, args: dict) -> dict:
    """Execute the appropriate sheets operation and return the raw result."""

    if function_name == "log_expense":
        date = args.get("date")
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        elif date is None:
            date = datetime.now()

        return sheets.log_expense(
            date=date,
            name=args.get("name", "Unknown"),
            amount=args.get("amount", 0),
            v_paid=args.get("v_paid", 0),
            y_paid=args.get("y_paid", 0),
            split=args.get("share", 0.5),
            labels=args.get("labels", []),
            notes=args.get("notes"),
            message_id=args.get("_message_id"),
        )

    elif function_name == "query_grouped_agg":
        conditions = [
            Condition(
                column=c["column"],
                value=c["value"],
                operation=c["operation"],
                is_inverse=c.get("is_inverse", False),
                since_last_settle_up=c.get("since_last_settle_up", False),
                transform=c.get("transform"),
            )
            for c in args.get("conditions", [])
        ]
        aggregations = [
            {"column": a["column"], "function": a["function"]}
            for a in args.get("aggregations", [])
        ]
        return sheets.query_grouped_agg(
            conditions=conditions,
            group_by=args.get("group_by", []),
            aggregations=aggregations,
            order_by_agg_index=args.get("order_by_agg_index"),
            order_desc=args.get("order_desc", True),
            limit=args.get("limit"),
        )

    elif function_name == "query_rows":
        conditions = [
            Condition(
                column=c["column"],
                value=c["value"],
                operation=c["operation"],
                is_inverse=c.get("is_inverse", False),
                since_last_settle_up=c.get("since_last_settle_up", False),
                transform=c.get("transform"),
            )
            for c in args.get("conditions", [])
        ]
        return sheets.query_rows(
            conditions=conditions,
            limit=args.get("limit"),
        )

    elif function_name == "get_balance":
        return sheets.get_balance()

    elif function_name == "settle_balance":
        return sheets.settle_balance()

    elif function_name == "clarify":
        return {"message": args.get("message", "Could you clarify what you mean?")}

    elif function_name == "delete_expense":
        # This returns candidates for deletion, not the actual deletion
        # The actual deletion happens via the short-code bypass
        delete_mode = args.get("delete_mode", "last")
        person = args.get("_person")  # Passed in from webhook handler

        # Build conditions based on delete mode
        conditions = []

        if delete_mode == "by_date":
            date = args.get("date")
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            if date:
                date_str = date.strftime("%Y-%m-%d")
                conditions.append(Condition(
                    column="Date",
                    value=date_str,
                    operation="contains",
                ))

        elif delete_mode == "by_merchant":
            merchant = args.get("merchant")
            if merchant:
                conditions.append(Condition(
                    column="Name",
                    value=merchant,
                    operation="contains",
                ))

        # Query with row indices
        result = sheets.query_rows(
            conditions=conditions,
            limit=3 if delete_mode != "last" else 1,
            include_row_index=True,
        )

        # Filter by person if needed (they should only delete their own expenses)
        rows = result.get("rows", [])
        if person:
            person_col = "v_paid" if person.upper() == "V" else "y_paid"
            rows = [r for r in rows if float(r.get(person_col) or 0) > 0]

        if not rows:
            return {"success": False, "error": "No matching expenses found", "candidates": []}

        # Return candidates
        return {
            "success": True,
            "candidates": rows[:3],  # Max 3 candidates
            "delete_mode": delete_mode,
        }

    else:
        return {"error": "Unknown intent"}


@functions_framework.http
def whatsapp_webhook(request):
    """Main webhook handler for WhatsApp messages."""

    # --- GET: Webhook Verification ---
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("Webhook verified successfully")
            return challenge, 200
        print(f"Webhook verification failed. Token mismatch: {token} != {VERIFY_TOKEN}")
        return "Forbidden", 403

    # --- POST: Message Handling ---
    if request.method == "POST":
        # Verify request is from Meta (skip in dev if no secret configured)
        if not verify_signature(request):
            print("Invalid signature - rejecting request")
            return "Invalid signature", 403

        data = request.get_json()
        print(f"Incoming webhook: {json.dumps(data)}")

        try:
            entry = data["entry"][0]["changes"][0]["value"]

            # Only process actual messages (not status updates)
            if "messages" not in entry:
                return "OK", 200

            message = entry["messages"][0]
            sender = message["from"]

            # Validate phone number is authorized
            person = get_person_from_phone(sender)
            if person is None:
                print(f"Unauthorized phone number: {sender}")
                reply_to_whatsapp(sender, "Sorry, this number is not authorized to use this bot.")
                return "OK", 200

            # Handle text messages only
            if message.get("type") != "text":
                reply_to_whatsapp(sender, "I can only process text messages for now.")
                return "OK", 200

            msg_body = message["text"]["body"]
            incoming_msg_id = message.get("id")  # The user's message ID
            reply_to_message_id = extract_reply_context(message)
            print(f"Message from {sender} ({person}): {msg_body}")

            # --- Reply-based delete: reply "delete" to an expense message ---
            if reply_to_message_id and msg_body.strip().lower() == "delete":
                delete_result = sheets.delete_expense_by_message_id(reply_to_message_id)
                if delete_result["success"]:
                    deleted_row = delete_result["deleted_row"]
                    amount = float(deleted_row.get("v_paid") or 0) + float(deleted_row.get("y_paid") or 0)
                    response_text = generate_response(
                        user_queries="delete expense",
                        intents="delete_expense",
                        results={"deleted": True, "expense": deleted_row.get("Name"), "amount": amount}
                    )
                else:
                    response_text = delete_result.get("error", "Could not delete that expense")
                reply_to_whatsapp(sender, response_text)
                return "OK", 200

            # --- Reply-based edit: reply with edit request to an expense message ---
            if reply_to_message_id and is_edit_request(msg_body):
                # Look up the expense by the message being replied to
                expense_data = sheets.get_expense_by_message_id(reply_to_message_id)

                if not expense_data:
                    reply_to_whatsapp(sender, "Could not find that expense. Make sure you're replying to an expense log message.")
                    return "OK", 200

                original_expense = expense_data["expense"]
                row_index = expense_data["row_index"]

                # Process the edit request through Gemini to extract changes
                edit_result = process_message(f"{person} said: {msg_body}")

                if edit_result.get("function") != "edit_expense":
                    reply_to_whatsapp(sender, "I couldn't understand what you want to edit. Try something like 'edit to $50' or 'change split to 60/40'")
                    return "OK", 200

                edit_args = edit_result.get("args", {})

                # Build the updates dict
                updates = {}
                orig_v = float(original_expense.get("v_paid") or 0)
                orig_y = float(original_expense.get("y_paid") or 0)
                orig_v_owes = float(original_expense.get("v_owes") or 0)
                current_amount = orig_v + orig_y

                # Derive original share (V's fraction of the expense)
                # from existing owes data. Default 0.5 if we can't determine.
                share = orig_v_owes / current_amount if current_amount > 0 else 0.5

                # Handle amount change
                if edit_args.get("new_amount") is not None:
                    new_amount = float(edit_args["new_amount"])
                    updates["Amount"] = new_amount
                    # Recalculate who paid proportionally
                    if current_amount > 0:
                        v_ratio = orig_v / current_amount
                        updates["v_paid"] = round(new_amount * v_ratio, 2)
                        updates["y_paid"] = round(new_amount * (1 - v_ratio), 2)
                    else:
                        updates["v_paid"] = round(new_amount / 2, 2)
                        updates["y_paid"] = round(new_amount / 2, 2)
                    current_amount = new_amount

                # Handle split change
                if edit_args.get("new_v_paid") is not None or edit_args.get("new_y_paid") is not None:
                    new_v = edit_args.get("new_v_paid")
                    new_y = edit_args.get("new_y_paid")

                    # Check if these look like percentages (values that sum to ~100)
                    if new_v is not None and new_y is not None:
                        if abs((new_v + new_y) - 100) < 1:  # Looks like percentages
                            share = new_v / 100
                            updates["v_paid"] = round(current_amount * new_v / 100, 2)
                            updates["y_paid"] = round(current_amount * new_y / 100, 2)
                        else:
                            updates["v_paid"] = float(new_v)
                            updates["y_paid"] = float(new_y)
                    elif new_v is not None:
                        new_v_val = float(new_v)
                        if new_v_val == 0:
                            updates["v_paid"] = 0
                            updates["y_paid"] = current_amount
                        else:
                            updates["v_paid"] = new_v_val
                            updates["y_paid"] = current_amount - new_v_val
                    elif new_y is not None:
                        new_y_val = float(new_y)
                        if new_y_val == 0:
                            updates["y_paid"] = 0
                            updates["v_paid"] = current_amount
                        else:
                            updates["y_paid"] = new_y_val
                            updates["v_paid"] = current_amount - new_y_val

                # Handle merchant change
                if edit_args.get("new_merchant"):
                    updates["Name"] = edit_args["new_merchant"]

                # Recalculate v_owes/y_owes based on final values
                if "v_paid" in updates or "y_paid" in updates or "Amount" in updates:
                    final_v_paid = updates.get("v_paid", orig_v)
                    final_y_paid = updates.get("y_paid", orig_y)
                    final_amount = updates.get("Amount", current_amount)
                    updates["v_owes"] = round(max(0, final_amount * share - final_v_paid), 2)
                    updates["y_owes"] = round(max(0, final_amount * (1 - share) - final_y_paid), 2)

                if not updates:
                    reply_to_whatsapp(sender, "I couldn't determine what to change. Please be specific, e.g., 'edit to $50' or 'change merchant to Costco'")
                    return "OK", 200

                # Build confirmation message
                code = generate_short_code()
                confirm_lines = ["Edit this expense?\n"]

                # Show original
                orig_merchant = original_expense.get("Name", "Unknown")

                confirm_lines.append(f"Original: ${current_amount if 'v_paid' not in updates else (orig_v + orig_y):.2f} at {orig_merchant}")
                if orig_v > 0 or orig_y > 0:
                    orig_total = orig_v + orig_y
                    v_pct = int(orig_v / orig_total * 100) if orig_total > 0 else 0
                    y_pct = 100 - v_pct
                    confirm_lines.append(f"  Split: {v_pct}/{y_pct} (V: ${orig_v:.2f}, Y: ${orig_y:.2f})")

                confirm_lines.append("    ↓")

                # Show new values
                new_v = updates.get("v_paid", orig_v)
                new_y = updates.get("y_paid", orig_y)
                new_amount = new_v + new_y
                new_merchant = updates.get("Name", orig_merchant)

                confirm_lines.append(f"New: ${new_amount:.2f} at {new_merchant}")
                if new_v != orig_v or new_y != orig_y or "v_paid" in updates or "y_paid" in updates:
                    new_v_pct = int(new_v / new_amount * 100) if new_amount > 0 else 0
                    new_y_pct = 100 - new_v_pct
                    confirm_lines.append(f"  Split: {new_v_pct}/{new_y_pct} (V: ${new_v:.2f}, Y: ${new_y:.2f})")

                confirm_lines.append(f"\nReply '{code}' to confirm")

                confirm_msg = "\n".join(confirm_lines)
                bot_msg_id = reply_to_whatsapp(sender, confirm_msg)

                if bot_msg_id:
                    sheets.store_pending_edit(bot_msg_id, code, row_index, updates)
                    print(f"Stored pending edit: {bot_msg_id} -> {code} -> row {row_index}")

                return "OK", 200

            # --- Short-code bypass for delete/edit confirmation ---
            if reply_to_message_id and is_short_code_format(msg_body.strip().lower()):
                code = msg_body.strip().lower()

                # Check for pending edit first
                pending_edit = sheets.get_pending_edit(reply_to_message_id)
                if pending_edit and code == pending_edit["code"]:
                    row_index = pending_edit["row_index"]
                    edit_data = pending_edit["edit_data"]

                    edit_result = sheets.update_row(row_index, edit_data)

                    if edit_result["success"]:
                        sheets.clear_pending_edit(reply_to_message_id)
                        updated_row = edit_result["updated_row"]
                        amount = float(updated_row.get("v_paid") or 0) + float(updated_row.get("y_paid") or 0)
                        response_text = generate_response(
                            user_queries=f"edit expense {code}",
                            intents="edit_expense",
                            results={"edited": True, "expense": updated_row.get("Name"), "amount": amount}
                        )
                    else:
                        response_text = f"Failed to edit: {edit_result.get('error', 'Unknown error')}"

                    reply_to_whatsapp(sender, response_text)
                    return "OK", 200

                # Check for pending delete
                pending_delete = sheets.get_pending_delete(reply_to_message_id)
                if pending_delete and code in pending_delete["code_mapping"]:
                    row_index = pending_delete["code_mapping"][code]
                    delete_result = sheets.delete_row(row_index)

                    if delete_result["success"]:
                        sheets.clear_pending_delete(reply_to_message_id)
                        deleted_row = delete_result["deleted_row"]
                        amount = float(deleted_row.get("v_paid") or 0) + float(deleted_row.get("y_paid") or 0)
                        response_text = generate_response(
                            user_queries=f"delete expense {code}",
                            intents="delete_expense",
                            results={"deleted": True, "expense": deleted_row.get("Name"), "amount": amount}
                        )
                    else:
                        response_text = f"Failed to delete: {delete_result.get('error', 'Unknown error')}"

                    reply_to_whatsapp(sender, response_text)
                    return "OK", 200

                # Looks like a short code but no matching pending action
                if pending_edit is None and pending_delete is None:
                    response_text = "This request has expired. Please try again."
                else:
                    response_text = f"Invalid code '{code}'. Please use the code shown above."
                reply_to_whatsapp(sender, response_text)
                return "OK", 200

            # Check for "list all modes" command
            msg_lower = msg_body.lower()
            if msg_lower in ('/list', "list"):
                response_text = "\n".join([
                    "Here's what I can do:\n",
                    "Log expense — \"$50 at Costco\"",
                    "Query totals — \"How much on groceries?\"",
                    "List expenses — \"Show last 5 expenses\"",
                    "Check balance — \"Who owes whom?\"",
                    "Settle up — \"We settled up\"",
                    "Delete expense — \"Delete the Costco expense\"",
                    "Edit expense — Reply to a logged expense with \"edit to $50\"",
                ])
                reply_to_whatsapp(sender, response_text)
                return "OK", 200

            # Step 1: Preprocess message (LLM call #1)
            preprocessor = MessagePreprocessor()
            preprocess_result = preprocessor.preprocess_message(msg_body)
            print(f"Preprocess result: {preprocess_result}")

            # Handle preprocessing errors
            if not preprocess_result.is_valid:
                if not preprocess_result.is_in_domain:
                    # Out-of-domain: use generate_response for witty reply
                    response_text = generate_response(
                        user_queries=msg_body,
                        intents="clarify",
                        results={"message": "This is not related to expense tracking"}
                    )
                else:
                    # Too many asks: direct error message
                    response_text = preprocess_result.error_message
                reply_to_whatsapp(sender, response_text)
                return "OK", 200

            # Step 2: Prefix each ask with person identifier
            asks = preprocess_result.asks
            prefixed_asks = [f"{person} said: {ask}" for ask in asks]
            print(f"Processing {len(asks)} ask(s): {asks}")

            # Step 3: Batch process all messages (LLM call #2)
            gemini_results = process_message(prefixed_asks)
            print(f"Gemini results: {gemini_results}")

            # Handle single vs. list return
            if not isinstance(gemini_results, list):
                gemini_results = [gemini_results]

            # Step 4: Execute intents
            intent_results = []
            delete_handled = False  # Track if we handled a delete specially

            for gemini_result in gemini_results:
                try:
                    if gemini_result.get("function"):
                        func_name = gemini_result["function"]
                        args = gemini_result["args"]

                        # For delete_expense, inject the person identifier
                        if func_name == "delete_expense":
                            args["_person"] = person

                        # For log_expense, inject the incoming message ID
                        if func_name == "log_expense":
                            args["_message_id"] = incoming_msg_id

                        intent_result = execute_intent(func_name, args)

                        # Special handling for delete_expense: send confirmation message
                        if func_name == "delete_expense" and intent_result.get("success"):
                            candidates = intent_result.get("candidates", [])
                            if candidates:
                                # Generate short codes for each candidate
                                code_mapping = {}
                                confirmation_lines = []

                                for candidate in candidates:
                                    code = generate_short_code()
                                    row_idx = candidate.get("row_index")
                                    code_mapping[code] = row_idx

                                    # Format display
                                    name = candidate.get("Name", "Unknown")
                                    amount = float(candidate.get("v_paid") or 0) + float(candidate.get("y_paid") or 0)
                                    date_str = candidate.get("Date", "")[:10]
                                    confirmation_lines.append(f"• ${amount:.2f} at {name} ({date_str}) → reply '{code}'")

                                # Build confirmation message
                                if len(candidates) == 1:
                                    confirm_msg = f"Delete this expense?\n{confirmation_lines[0]}"
                                else:
                                    confirm_msg = "Which expense to delete?\n" + "\n".join(confirmation_lines)

                                # Send confirmation and store pending
                                bot_msg_id = reply_to_whatsapp(sender, confirm_msg)
                                if bot_msg_id:
                                    sheets.store_pending_delete(bot_msg_id, code_mapping)
                                    print(f"Stored pending delete: {bot_msg_id} -> {code_mapping}")

                                delete_handled = True
                                continue  # Skip adding to intent_results

                        intent_results.append({
                            "intent": func_name,
                            "result": intent_result,
                            "success": True
                        })
                    else:
                        intent_results.append({
                            "intent": "error",
                            "result": {"error": "Could not understand request"},
                            "success": False
                        })
                except Exception as e:
                    print(f"Error executing intent: {e}")
                    intent_results.append({
                        "intent": "error",
                        "result": {"error": str(e)},
                        "success": False
                    })

            # If delete was handled specially, we already sent a response
            if delete_handled and not intent_results:
                return "OK", 200

            # Step 5: Generate aggregated response (LLM call #3)
            response_text = generate_response(
                user_queries=asks,
                intents=[r["intent"] for r in intent_results],
                results=[r["result"] for r in intent_results]
            )
            print(f"Final response: {response_text}")

            # Reply to user
            reply_to_whatsapp(sender, response_text)

        except KeyError as e:
            print(f"Error parsing webhook data: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")
            # Optionally notify user of error
            # reply_to_whatsapp(sender, "❌ Something went wrong. Please try again.")

        return "OK", 200

    return "Method not allowed", 405


# For local testing without functions-framework
if __name__ == "__main__":
    from flask import Flask, request as flask_request
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        return whatsapp_webhook(flask_request)

    app.run(host="0.0.0.0", port=8080, debug=True)

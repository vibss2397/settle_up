This is the engineering specification for the **WhatsApp-First Expense Tracker**.

This architecture creates a two-way chat bot where you text natural language to a dedicated number, and it replies with confirmation or reports. It uses the **Meta Cloud API** as the interface and **Google Cloud Functions** as the brain.

### 1. System Architecture

* **Pattern:** Webhook-based Chatbot.
* **Primary Interface:** WhatsApp (Text).
* **Latency:** < 3 seconds per turn.
* **Data Flow:** `User (WhatsApp)`  `Meta Cloud API`  `GCP Cloud Function`  `Gemini + Sheets`.

### 2. Bill of Materials

* **Google Cloud Project:**
* **Cloud Run Functions:** To host the Python code.
* **Service Account:** With "Editor" access to the specific Google Sheet.


* **Meta (Facebook) Developer Account:**
* **WhatsApp Business API:** Provides the test phone number and API keys.


* **Gemini API Key:** From Google AI Studio.

---

### 3. Component Specifications

#### Component A: The Interface (Meta Cloud API)

* **Role:** Handles sending/receiving messages.
* **Configuration:**
* **Product:** WhatsApp.
* **Webhook:** Points to your Cloud Function URL.
* **Events:** Subscribed to `messages`.


* **Security:**
* **Verify Token:** A custom string (e.g., `expense_bot_secret_2026`) used to handshake between Meta and your code.
* **Access Token:** A permanent (or long-lived) token from Meta to allow your code to reply.



#### Component B: The Brain (Cloud Function)

* **Runtime:** Python 3.10+
* **Dependencies:** `flask`, `functions-framework`, `google-generativeai`, `gspread`, `requests`.
* **Environment Variables:**
* `GEMINI_API_KEY`: Your AI Studio key.
* `META_PHONE_ID`: The ID of the phone number (from Meta dashboard).
* `META_TOKEN`: The API token (from Meta dashboard).
* `VERIFY_TOKEN`: Your chosen password for the webhook handshake.
* `SHEET_NAME`: "Household Expenses".



#### Component C: The Logic (Gemini Prompts)

We will use a **Two-Step Logic Flow** inside the Python code:

1. **Router (Intent Classification):**
* *Input:* User's message text
* *Prompt:* "Classify intent: LOG, QUERY, BALANCE, or SETTLE."
* *Output:* One of the four intents

| Intent | Trigger Examples | Action |
|--------|------------------|--------|
| `LOG` | "Spent $50 at Costco", "y paid $30 for groceries" | Record expense to sheet |
| `QUERY` | "How much on groceries?", "What did we spend in June?" | Return filtered totals |
| `BALANCE` | "What's the balance?", "Who owes whom?" | Show current settlement balance |
| `SETTLE` | "We settled up", "Cleared the balance" | Auto-calculate & record settlement payment |


2. **Worker (Entity Extraction):**
* *If LOG:* Extract `merchant`, `amount`, `date`, `label`, `v_share`, `y_share`.
* *If QUERY:* Extract filters - `month`, `label`, `merchant`, `person` (v/y). Apply all specified filters.
* *If BALANCE:* No extraction needed - calculate from all records.
* *If SETTLE:* No extraction needed - auto-calculate balance and record settlement row.



#### Component D: The Database (Google Sheets)

Same schema as before, ensuring robust tracking.

* **Headers (Row 1):**
`Timestamp` | `Name` | `Month` | `Labels` | `Amount` | `v Paid` | `y Paid`

---

### 4. Implementation Steps (The "How-To")

#### Step 1: Meta Developer Setup

1. Go to [developers.facebook.com](https://www.google.com/search?q=https://developers.facebook.com)  My Apps  Create App  **Other**  **Business**.
2. On the App Dashboard, scroll down to "Add products to your app" and select **WhatsApp**  **Set up**.
3. On the **API Setup** panel, you will see:
* **Temporary Access Token:** Copy this (you'll need it for `META_TOKEN`).
* **Phone Number ID:** Copy this (you'll need it for `META_PHONE_ID`).
* **Test Number:** Save this number; this is who you will text.


4. **Crucial:** Add your *real* personal phone number to the "To" field on that page and click "Manage phone number list" to verify your own number so you can receive messages.

#### Step 2: The Code (main.py)

This script handles the verification handshake *and* the message processing.

```python
import os
import json
import requests
import functions_framework
import google.generativeai as genai
import gspread
from datetime import datetime

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SHEET_NAME = os.environ.get("SHEET_NAME", "Household Expenses")
META_TOKEN = os.environ.get("META_TOKEN")
META_PHONE_ID = os.environ.get("META_PHONE_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")

genai.configure(api_key=GEMINI_KEY)
# Ensure service_account.json is in the root directory
gc = gspread.service_account(filename='service_account.json')

def reply_to_whatsapp(to_number, text):
    url = f"https://graph.facebook.com/v18.0/{META_PHONE_ID}/messages"
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
    except Exception as e:
        print(f"Failed to reply: {e}")

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {}

def calculate_balance(records):
    """Calculate who owes whom based on all records."""
    v_paid = sum(float(r.get('v Paid') or 0) for r in records)
    y_paid = sum(float(r.get('y Paid') or 0) for r in records)
    total = v_paid + y_paid
    # Each person's fair share is half the total
    v_owes = (total / 2) - v_paid  # positive = v owes y
    return v_owes

@functions_framework.http
def whatsapp_webhook(request):
    # --- PART 1: VERIFICATION HANDSHAKE (Meta checks this when saving URL) ---
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == VERIFY_TOKEN:
            return challenge, 200
        return "Forbidden", 403

    # --- PART 2: MESSAGE HANDLING ---
    if request.method == "POST":
        data = request.get_json()
        print(f"Incoming: {json.dumps(data)}") # Useful for debugging logs

        try:
            # Parse deeply nested WhatsApp JSON
            entry = data['entry'][0]['changes'][0]['value']

            # Check if it's a message (not a status update)
            if 'messages' in entry:
                message = entry['messages'][0]
                sender = message['from']
                msg_body = message['text']['body']

                # --- CORE LOGIC START ---

                # 1. Intent Classification
                intent_prompt = f"""Analyze '{msg_body}'. Return JSON with 'intent' key.
                - LOG: Recording a new expense (spending money)
                - QUERY: Asking about expenses (how much, totals, etc.)
                - BALANCE: Asking who owes whom, what's the balance
                - SETTLE: Recording that a settlement payment was made
                """
                intent = get_gemini_response(intent_prompt).get('intent', 'LOG')

                sh = gc.open(SHEET_NAME).sheet1

                if intent == 'LOG':
                    log_prompt = f"""
                    Extract data from: "{msg_body}". Today is {datetime.now().strftime("%Y-%m-%d")}.
                    JSON keys: merchant, amount (float), month, label, v_share, y_share.
                    Logic: "Split/Half" -> 50/50. "y" -> y pays 100%. Default -> v pays 100%.
                    """
                    d = get_gemini_response(log_prompt)

                    sh.append_row([
                        datetime.now().isoformat(), d.get('merchant'), d.get('month'),
                        d.get('label'), d.get('amount'), d.get('v_share'), d.get('y_share')
                    ])
                    reply_to_whatsapp(sender, f"✅ Logged ${d.get('amount')} at {d.get('merchant')}")

                elif intent == 'QUERY':
                    query_prompt = f"""
                    Extract filters from '{msg_body}'. Today is {datetime.now().strftime("%Y-%m-%d")}.
                    JSON keys (use null if not specified): month, label, merchant, person (v or y).
                    """
                    q = get_gemini_response(query_prompt)

                    records = sh.get_all_records()
                    filtered = records
                    filters_applied = []

                    # Apply filters dynamically
                    if q.get('month'):
                        filtered = [r for r in filtered if q['month'].lower() in str(r.get('Month', '')).lower()]
                        filters_applied.append(q['month'])
                    if q.get('label'):
                        filtered = [r for r in filtered if q['label'].lower() in str(r.get('Labels', '')).lower()]
                        filters_applied.append(q['label'])
                    if q.get('merchant'):
                        filtered = [r for r in filtered if q['merchant'].lower() in str(r.get('Name', '')).lower()]
                        filters_applied.append(q['merchant'])

                    total = sum(float(r.get('Amount') or 0) for r in filtered)
                    filter_desc = ', '.join(filters_applied) if filters_applied else 'All'
                    reply_to_whatsapp(sender, f"📊 Total for {filter_desc}: ${total:.2f}")

                elif intent == 'BALANCE':
                    records = sh.get_all_records()
                    v_owes = calculate_balance(records)

                    if v_owes > 0.01:
                        reply_to_whatsapp(sender, f"💰 v owes y ${v_owes:.2f}")
                    elif v_owes < -0.01:
                        reply_to_whatsapp(sender, f"💰 y owes v ${-v_owes:.2f}")
                    else:
                        reply_to_whatsapp(sender, "✅ All settled up!")

                elif intent == 'SETTLE':
                    records = sh.get_all_records()
                    v_owes = calculate_balance(records)

                    if abs(v_owes) < 0.01:
                        reply_to_whatsapp(sender, "✅ Already settled - no balance to clear!")
                    else:
                        # Record the settlement payment
                        if v_owes > 0:
                            # v owes y, so v pays
                            sh.append_row([
                                datetime.now().isoformat(), "Settlement",
                                datetime.now().strftime("%B"), "Settlement",
                                0, v_owes, 0
                            ])
                            reply_to_whatsapp(sender, f"✅ Recorded: v paid y ${v_owes:.2f}. Balance cleared!")
                        else:
                            # y owes v, so y pays
                            sh.append_row([
                                datetime.now().isoformat(), "Settlement",
                                datetime.now().strftime("%B"), "Settlement",
                                0, 0, -v_owes
                            ])
                            reply_to_whatsapp(sender, f"✅ Recorded: y paid v ${-v_owes:.2f}. Balance cleared!")

                # --- CORE LOGIC END ---

        except Exception as e:
            print(f"Error processing message: {e}")
            # Optional: reply_to_whatsapp(sender, "Error processing request.")

        return "EVENT_RECEIVED", 200

```

#### Step 3: Deployment

1. **Deploy to Google Cloud:**
```bash
gcloud functions deploy expense-bot-v2 \
--runtime python310 \
--trigger-http \
--allow-unauthenticated \
--set-env-vars GEMINI_API_KEY=...,META_TOKEN=...,META_PHONE_ID=...,VERIFY_TOKEN=my_secret_123

```


*(Remember to upload `service_account.json` and `requirements.txt` with it)*
2. **Connect Webhook:**
* Copy the URL (`https://...cloudfunctions.net/expense-bot-v2`).
* Go back to Meta Dashboard  WhatsApp  Configuration.
* Click **Edit** next to Webhook.
* Paste the URL.
* Enter `my_secret_123` (or whatever you set for `VERIFY_TOKEN`).
* **Verify and Save**. (If this fails, check Cloud Function logs for "Verify Token Mismatch").


3. **Subscribe:**
* On the same Meta page, under "Webhook fields", click **Manage**.
* Check the box for `messages`.
* Click **Subscribe**.


#### Step 4: Permanent Access Token (Required for Production)

The temporary token from Step 1 expires in 24 hours. For a reliable bot, create a permanent System User token:

1. Go to **Meta Business Suite** → **Settings** → **Business Settings**
2. Navigate to **Users** → **System Users** → **Add**
3. Create a System User with **Admin** role
4. Click **Add Assets** → Select your WhatsApp app → Grant **full_access**
5. Click **Generate New Token** → Select permissions:
   * `whatsapp_business_messaging`
   * `whatsapp_business_management`
6. Copy the generated token
7. Update your Cloud Function's `META_TOKEN` environment variable with this new token
8. Redeploy the function:
```bash
gcloud functions deploy expense-bot-v2 \
--update-env-vars META_TOKEN=your_new_permanent_token
```

---

### 5. Future "Voice" Add-on (The Hybrid Model)

Later, when you want to use your Google Home:

1. Set up the **IFTTT** applet exactly as discussed before ("Add expense $").
2. Point the IFTTT Webhook to this **SAME** Cloud Function URL.
3. **Update the Code:** Modify the `whatsapp_webhook` function to check the payload structure.
* If it has `entry`  It's WhatsApp.
* If it has `raw_text`  It's IFTTT.


4. This way, both Voice and Text feed into the exact same Logic and Database.

### 6. Next Action

Do you have your **Meta Developer Portal** open? I can guide you through generating the permanent token (System User) so your bot doesn't stop working after 24 hours (the default temporary token limit).
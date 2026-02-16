# Settle Up - WhatsApp Expense Tracker

A WhatsApp-first expense tracking chatbot that lets you manage household expenses through natural language messages. Simply text your expenses and get instant confirmations, expense summaries, and balance calculations.

## Overview

Settle Up is a serverless chatbot that uses WhatsApp as its primary interface, Google Gemini for natural language processing, and Google Sheets as its database. It's designed for couples or roommates who want to track shared expenses and settlements without dealing with complex apps.

## Key Features

- **Natural Language Logging**: Text expenses in plain English - "Spent $50 at Costco" or "Y paid $30 for groceries"
- **Expense Queries**: Ask questions like "How much on groceries this month?" or "Show last 5 expenses"
- **Balance Tracking**: Check who owes whom with "What's the balance?"
- **Auto-Settlement**: Record settlements automatically with "We settled up"
- **Edit & Delete**: Modify or remove expenses - "Delete last expense", "Edit this to $50", "Change split to 60/40"
- **Real-time Processing**: Get responses in under 3 seconds
- **Witty Responses**: AI-generated sassy replies that make expense tracking more fun
- **Secure**: Uses Meta signature verification and service account authentication

## Architecture

```
User (WhatsApp) → Meta Cloud API → GCP Cloud Function → Gemini + Google Sheets
```

**Components:**
- **Interface**: WhatsApp via Meta Cloud API
- **Backend**: Google Cloud Functions (Python 3.10+)
- **AI Processing**: Google Gemini 1.5 Flash for intent classification and entity extraction
- **Database**: Google Sheets with structured expense tracking
- **Authentication**: Meta webhook verification + Google service account

## Intent Types

The bot handles six types of interactions:

| Intent | Examples | Action |
|--------|----------|--------|
| **LOG** | "Spent $50 at Costco", "Y paid $30 for groceries" | Records expense to sheet |
| **QUERY** | "How much on groceries?", "What did we spend in June?", "Show last 5 expenses" | Returns filtered totals or individual records |
| **BALANCE** | "What's the balance?", "Who owes whom?" | Calculates current settlement balance |
| **SETTLE** | "We settled up", "Cleared the balance" | Auto-calculates & records settlement payment |
| **DELETE** | "Delete last expense", "Remove the Costco expense", "Undo that" | Deletes expenses by recency, date, or merchant |
| **EDIT** | "Edit this to $50", "Change split to 60/40", "Put it all on Y" | Modifies existing expense amount, split, or merchant |

### Advanced Features

**DELETE Intent** supports three modes:
- **Last**: Delete most recent expense - "delete last expense", "undo my last expense", "remove that"
- **By Date**: Delete expense from specific date - "delete expense from yesterday", "remove the Jan 5 expense"
- **By Merchant**: Delete expense at merchant - "delete costco expense", "remove the starbucks charge"

**EDIT Intent** allows modifying existing expenses (typically when replying to an expense confirmation):
- **Amount**: "edit this to $50", "change amount to 75"
- **Split**: "change the split to 60/40", "put it all on Y"
- **Merchant**: "change merchant to Costco"
- **Multiple fields**: "edit to $75 at Target"

**QUERY Intent** supports two modes:
- **Grouped Aggregations**: Get totals with optional grouping - "How much on groceries?", "Monthly spending breakdown", "Top 3 categories"
- **Individual Records**: List specific expenses - "Show me last 5 expenses", "What did we buy at Costco?"

## Project Structure

```
settle_up/
├── main.py                          # Cloud Function entry point
├── requirements.txt                 # Python dependencies
├── src/
│   ├── llm/
│   │   ├── gemini_handler.py       # Gemini AI integration
│   │   ├── message_preprocessor.py # Message parsing & intent detection
│   │   └── sheets_handler.py       # Google Sheets operations
│   ├── creds/
│   │   └── service_account.json    # GCP service account credentials
│   ├── mapping.json                # WhatsApp phone → person mapping
│   └── tests/                      # Unit tests
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Local development setup
├── .env.example                    # Environment variable template
└── spec.md                         # Detailed engineering specification
```

## Google Sheets Schema

The bot writes to a Google Sheet with the following structure:

| Timestamp | Name | Month | Labels | Amount | v Paid | y Paid |
|-----------|------|-------|--------|--------|---------|---------|
| ISO 8601 | Merchant | Month name | Category | Total | V's share | Y's share |

## Setup & Deployment

### Prerequisites

1. Google Cloud Platform account
2. Meta Developer account with WhatsApp Business API access
3. Google Gemini API key
4. Google Sheet with proper sharing permissions

### Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key
META_TOKEN=your_meta_access_token
META_PHONE_ID=your_whatsapp_phone_number_id
META_APP_SECRET=your_app_secret_optional
VERIFY_TOKEN=your_webhook_verify_secret
SHEET_NAME=Household Expenses
SPREADSHEET_NAME=Settle_up
```

### Deploy to Google Cloud

```bash
gcloud functions deploy expense-bot-v2 \
  --runtime python310 \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=...,META_TOKEN=...,META_PHONE_ID=...,VERIFY_TOKEN=...,META_APP_SECRET=...,SHEET_NAME=...,SPREADSHEET_NAME=...
```

### Local Development

1. Copy `.env.example` to `.env` and fill in your credentials
2. Place your service account JSON in `src/creds/service_account.json`
3. Update `src/mapping.json` with your WhatsApp phone numbers
4. Run with Docker Compose:

```bash
docker-compose up
```

Or run locally:

```bash
pip install -r requirements.txt
python -m flask --app main run
```

## Testing

Run the test suite:

```bash
pytest src/tests/
```

Tests cover:
- Message preprocessing and intent classification
- Gemini API integration
- Google Sheets operations

## Security

- **Webhook Verification**: Custom verify token for Meta webhook handshake
- **Signature Validation**: HMAC-SHA256 verification of Meta requests
- **Service Account**: Scoped Google Sheets access
- **Phone Mapping**: Restricts access to known phone numbers

## Tech Stack

- **Runtime**: Python 3.10+
- **Frameworks**: Flask, Functions Framework
- **AI/ML**: Google Gemini 1.5 Flash
- **Database**: Google Sheets (gspread)
- **Messaging**: Meta WhatsApp Cloud API
- **Infrastructure**: Google Cloud Functions
- **Validation**: Pydantic

## How It Works

1. User sends WhatsApp message to bot number
2. Meta forwards message to Cloud Function webhook
3. Function validates signature and extracts message
4. Message preprocessor identifies intent using Gemini
5. Based on intent:
   - **LOG**: Extract expense details and append to sheet
   - **QUERY**: Filter sheet records and calculate totals or return individual records
   - **BALANCE**: Calculate net balance from all records
   - **SETTLE**: Calculate balance and record settlement payment
   - **DELETE**: Find and remove matching expense(s) from sheet
   - **EDIT**: Update existing expense with new values
6. Bot generates witty response using Gemini and replies via WhatsApp API

## Future Enhancements

- Voice interface via Google Home + IFTTT
- Support for more than 2 people
- Expense categories and budgets
- Monthly reports and analytics
- Receipt photo parsing
- Multi-currency support

## Documentation

- [spec.md](spec.md) - Detailed engineering specification
- [deploy_command.md](deploy_command.md) - Deployment guide

## License

Private project for personal use.

## Contributing

This is a personal project, but suggestions and improvements are welcome via issues.

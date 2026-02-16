from datetime import datetime, timedelta
from typing import Optional, Literal
from dataclasses import dataclass
import gspread
import json

@dataclass
class Condition:
    column: str
    value: any
    operation: Literal[
        "==", ">", "<", ">=", "<=", "!=",
        "in",
        "contains", "substr"
    ]
    # Setting this to true means we want the opposite of the condition to happen.
    is_inverse: bool = False
    # Setting this to true only filters records since the last settle up event if it exists
    since_last_settle_up: bool = False
    # Transform to apply to column value before comparison (for Date column)
    transform: Optional[Literal["month", "year", "weekday"]] = None


class SheetsHandler:
    """Handles all Google Sheets operations for expense tracking."""

    def __init__(self, spreadsheet_name: str, sheet_name: str, credentials_path: str = "service_account.json"):
        self.gc = gspread.service_account(filename=credentials_path)
        self.sheet = self.gc.open(spreadsheet_name).worksheet(sheet_name)


    def log_expense(
        self,
        date: datetime,
        name: str,
        amount: float,
        v_paid: float,
        y_paid: float,
        notes: Optional[str],
        labels: Optional[list] = [],
        message_id: Optional[str] = None,
    ) -> dict:
        """Log a new expense to the sheet.

        Args:
            message_id: The WhatsApp message ID of the user's message (for reply-based delete)
        """
        row = [
            date.isoformat(),
            name,
            amount,
            v_paid,
            y_paid,
            ", ".join(labels),
            notes,
            message_id,
        ]
        self.sheet.append_row(row)
        return {
            "merchant": name,
            "amount": amount,
            "labels": labels,
            "notes": notes,
            "v_paid": v_paid,
            "y_paid": y_paid,
        }

    def delete_expense_by_message_id(self, message_id: str) -> dict:
        """Find and delete an expense by its WhatsApp message_id.

        Args:
            message_id: The WhatsApp message ID to look up

        Returns:
            {"success": True, "deleted_row": dict} or {"success": False, "error": str}
        """
        records = self.sheet.get_all_records()
        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                row_index = i + 2  # +2 for header and 0-indexing
                return self.delete_row(row_index)
        return {"success": False, "error": "Expense not found for this message"}

    def get_expense_by_message_id(self, message_id: str) -> dict | None:
        """Find an expense by its WhatsApp message_id.

        Args:
            message_id: The WhatsApp message ID to look up

        Returns:
            {"row_index": int, "expense": dict} or None if not found
        """
        records = self.sheet.get_all_records()
        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                row_index = i + 2  # +2 for header and 0-indexing
                return {"row_index": row_index, "expense": record}
        return None

    def update_row(self, row_index: int, updates: dict) -> dict:
        """Update specific fields in a row by index.

        Args:
            row_index: The 1-based row number in the sheet (row 1 is header, row 2+ are data)
            updates: Dict of column names to new values. Supported columns:
                     Name, Amount, v_paid, y_paid, Labels, Notes

        Returns:
            {"success": True, "original_row": dict, "updated_row": dict} or {"success": False, "error": str}
        """
        try:
            records = self.sheet.get_all_records()
            record_idx = row_index - 2  # Convert to 0-based index

            if record_idx < 0 or record_idx >= len(records):
                return {"success": False, "error": f"Row {row_index} not found"}

            original_record = dict(records[record_idx])

            # Column mapping: column name -> column index (1-based for gspread)
            column_indices = {
                "Date": 1,
                "Name": 2,
                "Amount": 3,
                "v_paid": 4,
                "y_paid": 5,
                "Labels": 6,
                "Notes": 7,
                "message_id": 8,
            }

            updated_record = dict(original_record)
            for col_name, new_value in updates.items():
                if col_name in column_indices:
                    col_idx = column_indices[col_name]
                    # Handle special formatting
                    if col_name == "Labels" and isinstance(new_value, list):
                        new_value = ", ".join(new_value)

                    # Update the cell
                    self.sheet.update_cell(row_index, col_idx, new_value)
                    updated_record[col_name] = new_value

            return {
                "success": True,
                "original_row": original_record,
                "updated_row": updated_record
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_transform(self, value: any, transform: Optional[str]) -> any:
        """Apply a transform to a value (currently only for Date column)."""
        if transform is None:
            return value

        # Parse ISO date string
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                return value
        elif isinstance(value, datetime):
            dt = value
        else:
            return value

        if transform == "month":
            return dt.strftime("%B")  # January, February, ...
        elif transform == "year":
            return str(dt.year)
        elif transform == "weekday":
            return dt.strftime("%A")  # Monday, Tuesday, ...
        return value

    def _filter_records(
        self,
        conditions: list[Condition],
        include_row_index: bool = False,
    ) -> tuple[list[dict], list[str]]:
        """
        Internal method to filter records based on conditions.
        Returns (filtered_records, filters_applied).

        If include_row_index=True, each record will have a '_row_index' key
        with its 1-based row number in the sheet (for use with delete operations).
        """
        records = self.sheet.get_all_records()

        # Add row indices: get_all_records() starts at row 2 (row 1 is header)
        # So record at index i has row_index = i + 2
        indexed_records = [
            {**r, "_row_index": i + 2} for i, r in enumerate(records)
        ]

        # If any condition has since_last_settle_up=True, filter to records after last settle-up
        if any(c.since_last_settle_up for c in conditions):
            last_settle_idx = -1
            for i, r in enumerate(indexed_records):
                labels = str(r.get("Labels", "")).lower()
                if "settle-up" in labels:
                    last_settle_idx = i
            if last_settle_idx >= 0:
                indexed_records = indexed_records[last_settle_idx + 1:]

        filters_applied = []

        def matches_condition(record: dict, cond: Condition) -> bool:
            col_value = record.get(cond.column)
            # Apply transform if specified (e.g., extract month from Date)
            col_value = self._apply_transform(col_value, cond.transform)
            target = cond.value
            op = cond.operation

            if op == "==":
                # Compare as strings to handle type mismatches (e.g., "2026" vs 2026)
                result = str(col_value) == str(target)
            elif op == "!=":
                result = str(col_value) != str(target)
            elif op == ">":
                result = float(col_value or 0) > float(target)
            elif op == "<":
                result = float(col_value or 0) < float(target)
            elif op == ">=":
                result = float(col_value or 0) >= float(target)
            elif op == "<=":
                result = float(col_value or 0) <= float(target)
            elif op == "in":
                result = col_value in target  # target is a list
            elif op in ("contains", "substr"):
                # Case-insensitive substring match
                result = str(target).lower() in str(col_value or "").lower()
            else:
                result = False

            return not result if cond.is_inverse else result

        # Apply all conditions (AND logic)
        filtered = indexed_records
        for cond in conditions:
            filtered = [r for r in filtered if matches_condition(r, cond)]
            inv_prefix = "NOT " if cond.is_inverse else ""
            filters_applied.append(f"{inv_prefix}{cond.column} {cond.operation} {cond.value}")

        # Remove _row_index from results if not requested
        if not include_row_index:
            filtered = [{k: v for k, v in r.items() if k != "_row_index"} for r in filtered]

        return filtered, filters_applied

    def query_grouped_agg(
        self,
        conditions: list[Condition],
        group_by: list[str],
        aggregations: list[dict],
        order_by_agg_index: Optional[int] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Query with grouping and aggregation.
        group_by supports dot notation for Date transforms: Date.month, Date.year, Date.weekday
        aggregations: list of {"column": str, "function": str} where function is count|sum|avg|median
        """
        import statistics
        from collections import defaultdict

        filtered, filters_applied = self._filter_records(conditions)

        def get_group_key(record: dict) -> tuple:
            """Extract group key from record, applying transforms for dot notation."""
            key_parts = []
            for gb in group_by:
                if "." in gb:
                    col, transform = gb.split(".", 1)
                    val = self._apply_transform(record.get(col), transform)
                else:
                    val = record.get(gb)
                key_parts.append(val)
            return tuple(key_parts)

        def compute_agg(records: list, agg: dict) -> float:
            """Compute aggregation for a group of records."""
            col = agg["column"]
            fn = agg["function"]

            # For "Amount", compute v_paid + y_paid
            if col == "Amount":
                values = [float(r.get("v_paid") or 0) + float(r.get("y_paid") or 0) for r in records]
            else:
                values = [float(r.get(col) or 0) for r in records]

            if fn == "count":
                return len(records)
            elif fn == "sum":
                return sum(values)
            elif fn == "avg":
                return statistics.mean(values) if values else 0
            elif fn == "median":
                return statistics.median(values) if values else 0
            return 0

        # Group records
        groups = defaultdict(list)
        for record in filtered:
            key = get_group_key(record)
            groups[key].append(record)

        # Build results
        results = []

        # Handle empty group_by (aggregate all records)
        if not group_by:
            if filtered:
                row = {}
                for agg in aggregations:
                    agg_name = f"{agg['function']}_{agg['column']}"
                    row[agg_name] = compute_agg(filtered, agg)
                results = [row]
        else:
            for key, records in groups.items():
                row = {}
                for i, gb in enumerate(group_by):
                    row[gb] = key[i]
                for agg in aggregations:
                    agg_name = f"{agg['function']}_{agg['column']}"
                    row[agg_name] = compute_agg(records, agg)
                results.append(row)

        # Sort by aggregation if specified
        if order_by_agg_index is not None and aggregations:
            agg = aggregations[order_by_agg_index]
            agg_name = f"{agg['function']}_{agg['column']}"
            results.sort(key=lambda x: x.get(agg_name, 0), reverse=order_desc)

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return {
            "results": results,
            "filters_applied": filters_applied,
            "group_by": group_by,
            "record_count": len(filtered),
        }

    def query_rows(
        self,
        conditions: list[Condition],
        limit: Optional[int] = None,
        include_row_index: bool = False,
    ) -> dict:
        """
        Query expenses and return matching rows.
        Conditions are applied in AND fashion.
        Returns most recent records first (reversed order).

        If include_row_index=True, each row will have 'row_index' with its
        1-based sheet row number (useful for delete operations).
        """
        filtered, filters_applied = self._filter_records(conditions, include_row_index=include_row_index)

        # Return most recent first
        filtered = list(reversed(filtered))

        # Rename _row_index to row_index for cleaner API
        if include_row_index:
            for row in filtered:
                if "_row_index" in row:
                    row["row_index"] = row.pop("_row_index")

        # Apply limit if specified
        if limit is not None:
            filtered = filtered[:limit]

        return {
            "rows": filtered,
            "filters_applied": filters_applied,
            "record_count": len(filtered),
        }

    def get_balance(self) -> dict:
        """Calculate who owes whom based on records since last settle-up."""
        records = self.sheet.get_all_records()

        # Find the index of the last "settle-up" row
        last_settle_idx = -1
        for i, r in enumerate(records):
            labels = str(r.get("Labels", "")).lower()
            if "settle-up" in labels:
                last_settle_idx = i

        # Use only records after the last settle-up (or all if none found)
        if last_settle_idx >= 0:
            records = records[last_settle_idx + 1 :]

        v_paid = sum(float(r.get("v_paid") or 0) for r in records)
        y_paid = sum(float(r.get("y_paid") or 0) for r in records)
        diff = abs(v_paid - y_paid)
        total = v_paid + y_paid

        if v_paid == y_paid:
            who_owes = None
        elif v_paid > y_paid:
            who_owes = "y"
        else:
            who_owes = "v"

        return {
            "v_paid_total": v_paid,
            "y_paid_total": y_paid,
            "total": total,
            "amount_owed": diff,
            "who_owes": who_owes,
        }



    def settle_balance(self) -> dict:
        """Record a settlement payment to clear the balance."""
        balance = self.get_balance()
        who_owes = balance.get("who_owes")
        amount_owed = balance.get("amount_owed")
        if not who_owes:
            return {"settled": False, "message": "No balance to settle!"}

        payer, payee = None, None
        if who_owes == "v":
            row = [
                datetime.now().isoformat(),
                "Settlement",
                amount_owed,
                amount_owed,
                0,
                "settle-up",
                "settling up from last batch of payments"
            ]
            payer = "v"
            payee = "y"

        else:
            row = [
                datetime.now().isoformat(),
                "Settlement",
                amount_owed,
                0,
                amount_owed,
                "settle-up",
                "settling up from last batch of payments"
            ]
            payer = "y"
            payee = "v"
             
        self.sheet.append_row(row)
        return {
            "settled": True,
            "payer": payer,
            "payee": payee,
            "amount": amount_owed
        }

    def delete_row(self, row_index: int) -> dict:
        """
        Delete a row from the sheet by its 1-based row index.

        Args:
            row_index: The 1-based row number in the sheet (row 1 is header, row 2+ are data)

        Returns:
            {"success": True, "deleted_row": dict} or {"success": False, "error": str}
        """
        try:
            # Get the row data before deleting (for confirmation message)
            records = self.sheet.get_all_records()
            record_idx = row_index - 2  # Convert to 0-based index (accounting for header)

            if record_idx < 0 or record_idx >= len(records):
                return {"success": False, "error": f"Row {row_index} not found"}

            deleted_record = records[record_idx]
            self.sheet.delete_rows(row_index)

            return {"success": True, "deleted_row": deleted_record}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_pending_sheet(self):
        """Get or create the PendingDeletes worksheet."""
        spreadsheet = self.sheet.spreadsheet
        try:
            return spreadsheet.worksheet("PendingDeletes")
        except gspread.WorksheetNotFound:
            pending_sheet = spreadsheet.add_worksheet(
                title="PendingDeletes",
                rows=100,
                cols=3
            )
            pending_sheet.append_row(["message_id", "code_mapping", "expires_at"])
            return pending_sheet

    def store_pending_delete(
        self,
        message_id: str,
        code_mapping: dict[str, int],
        expires_minutes: int = 30,
    ) -> dict:
        """
        Store a pending delete action.

        Args:
            message_id: The WhatsApp message ID of the bot's confirmation message
            code_mapping: Dict mapping short codes to row indices, e.g. {"xyz": 47, "abc": 52}
            expires_minutes: How long until this pending action expires

        Returns:
            {"stored": True, "expires_at": str}
        """
        pending_sheet = self._get_pending_sheet()
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)

        row = [
            message_id,
            json.dumps(code_mapping),
            expires_at.isoformat(),
        ]
        pending_sheet.append_row(row)

        return {"stored": True, "expires_at": expires_at.isoformat()}

    def get_pending_delete(self, message_id: str) -> dict | None:
        """
        Retrieve a pending delete action by message ID.

        Args:
            message_id: The WhatsApp message ID to look up

        Returns:
            {"code_mapping": dict, "expires_at": str, "row_index": int} or None if not found/expired
        """
        pending_sheet = self._get_pending_sheet()
        records = pending_sheet.get_all_records()

        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                expires_at_str = record.get("expires_at", "")
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                except ValueError:
                    continue

                # Check if expired
                if datetime.now() > expires_at:
                    return None

                code_mapping_str = record.get("code_mapping", "{}")
                try:
                    code_mapping = json.loads(code_mapping_str)
                except json.JSONDecodeError:
                    code_mapping = {}

                return {
                    "code_mapping": code_mapping,
                    "expires_at": expires_at_str,
                    "pending_row_index": i + 2,  # For clearing later
                }

        return None

    def clear_pending_delete(self, message_id: str) -> bool:
        """
        Remove a pending delete action after it's been used or cancelled.

        Args:
            message_id: The WhatsApp message ID to clear

        Returns:
            True if cleared, False if not found
        """
        pending_sheet = self._get_pending_sheet()
        records = pending_sheet.get_all_records()

        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                pending_sheet.delete_rows(i + 2)  # +2 for header and 0-indexing
                return True

        return False

    def _get_pending_edits_sheet(self):
        """Get or create the PendingEdits worksheet."""
        spreadsheet = self.sheet.spreadsheet
        try:
            return spreadsheet.worksheet("PendingEdits")
        except gspread.WorksheetNotFound:
            pending_sheet = spreadsheet.add_worksheet(
                title="PendingEdits",
                rows=100,
                cols=5
            )
            pending_sheet.append_row(["message_id", "code", "row_index", "edit_data", "expires_at"])
            return pending_sheet

    def store_pending_edit(
        self,
        message_id: str,
        code: str,
        row_index: int,
        edit_data: dict,
        expires_minutes: int = 30,
    ) -> dict:
        """
        Store a pending edit action.

        Args:
            message_id: The WhatsApp message ID of the bot's confirmation message
            code: The 3-letter confirmation code
            row_index: The row to edit
            edit_data: Dict of field updates to apply
            expires_minutes: How long until this pending action expires

        Returns:
            {"stored": True, "expires_at": str}
        """
        pending_sheet = self._get_pending_edits_sheet()
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)

        row = [
            message_id,
            code,
            row_index,
            json.dumps(edit_data),
            expires_at.isoformat(),
        ]
        pending_sheet.append_row(row)

        return {"stored": True, "expires_at": expires_at.isoformat()}

    def get_pending_edit(self, message_id: str) -> dict | None:
        """
        Retrieve a pending edit action by message ID.

        Args:
            message_id: The WhatsApp message ID to look up

        Returns:
            {"code": str, "row_index": int, "edit_data": dict, "expires_at": str} or None if not found/expired
        """
        pending_sheet = self._get_pending_edits_sheet()
        records = pending_sheet.get_all_records()

        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                expires_at_str = record.get("expires_at", "")
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                except ValueError:
                    continue

                # Check if expired
                if datetime.now() > expires_at:
                    return None

                edit_data_str = record.get("edit_data", "{}")
                try:
                    edit_data = json.loads(edit_data_str)
                except json.JSONDecodeError:
                    edit_data = {}

                return {
                    "code": record.get("code", ""),
                    "row_index": int(record.get("row_index", 0)),
                    "edit_data": edit_data,
                    "expires_at": expires_at_str,
                }

        return None

    def clear_pending_edit(self, message_id: str) -> bool:
        """
        Remove a pending edit action after it's been used or cancelled.

        Args:
            message_id: The WhatsApp message ID to clear

        Returns:
            True if cleared, False if not found
        """
        pending_sheet = self._get_pending_edits_sheet()
        records = pending_sheet.get_all_records()

        for i, record in enumerate(records):
            if record.get("message_id") == message_id:
                pending_sheet.delete_rows(i + 2)  # +2 for header and 0-indexing
                return True

        return False

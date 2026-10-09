"""
Expense Reimbursement Workflow
Implements a multi-stage approval process using TypeSafe AI (Jev) API
"""

import os
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Noul, Choice, Score

load_dotenv()


# ==================== Data Models ====================


class WorkflowStatus(Enum):
    """Status of the expense in the workflow"""

    SUBMITTED = "submitted"
    RECEIPTS_PENDING = "receipts_pending"
    AUTO_APPROVED = "auto_approved"
    MANAGER_REJECTED = "manager_rejected"
    EXECUTIVE_REVIEW = "executive_review"
    COMPLETED = "completed"


@dataclass
class Receipt:
    """Receipt information"""

    id: str
    name: str
    category: str
    timestamp: str


@dataclass
class ExpenseReport:
    """Expense report structure"""

    id: str
    employee_id: str
    employee_name: str
    amount: float
    description: str
    receipts: List[Receipt]
    submission_date: str
    status: WorkflowStatus = WorkflowStatus.SUBMITTED
    manager_notes: Optional[str] = None
    vp_notes: Optional[str] = None


# ==================== Decision Gates ====================


class ReceiptValidator:
    """Gate 1: Validates receipt completeness"""

    def __init__(self, client: TypeSafeClient):
        self.client = client

    def validate(self, expense: ExpenseReport) -> Tuple[bool, str]:
        """Validate receipts for the expense using TypeSafe API"""
        receipts_info = ", ".join(
            [f"{r.name} ({r.category})" for r in expense.receipts]
        )

        state = f"""Expense Report Validation:
Amount: {expense.amount} INR
Number of Receipts: {len(expense.receipts)}
Receipt Details: {receipts_info}

Receipt Requirements:
- For amounts < ₹2,000: At least 1 receipt required
- For amounts ₹2,000-₹10,000: All individual items need receipts
- For amounts > ₹10,000: Itemized receipt AND payment proof required"""

        response = self.client.system_one(
            state=state,
            questions={
                "is_complete": Noul(
                    instructions="Does the expense have all required receipts based on the amount?"
                ),
                "missing_items": Noul(
                    instructions="Are any required receipt items missing?"
                ),
            },
        )

        is_complete = response.answers["is_complete"].noul == 1.0
        has_missing = response.answers["missing_items"].noul == 1.0

        reason = (
            "Receipt validation passed" if is_complete else "Missing required receipts"
        )

        return is_complete, reason


class ExpenseLimitGate:
    """Gate 2 & 4: Routes expenses based on amount thresholds"""

    TIER_1_LIMIT = 5000  # Auto-approve if under this
    TIER_2_LIMIT = 50000  # Finance if under this, VP if above

    @staticmethod
    def check_auto_approval(amount: float) -> bool:
        """Check if amount qualifies for auto-approval (< ₹5,000)"""
        return amount < ExpenseLimitGate.TIER_1_LIMIT

    @staticmethod
    def requires_vp_approval(amount: float) -> bool:
        """Check if amount requires VP approval (>= ₹50,000)"""
        return amount >= ExpenseLimitGate.TIER_2_LIMIT

    @staticmethod
    def get_routing(amount: float) -> str:
        """Determine routing based on amount"""
        if amount < ExpenseLimitGate.TIER_1_LIMIT:
            return "AUTO_APPROVE"
        elif amount < ExpenseLimitGate.TIER_2_LIMIT:
            return "FINANCE"
        else:
            return "VP_APPROVAL"


class ManagerReviewer:
    """Gate 3: Manager review and decision"""

    def __init__(self, client: TypeSafeClient):
        self.client = client

    def review(self, expense: ExpenseReport) -> Tuple[bool, str]:
        """Review expense and return decision using TypeSafe API"""
        state = f"""Manager Review for Expense:
Employee: {expense.employee_name}
Amount: {expense.amount} INR
Description: {expense.description}
Number of Receipts: {len(expense.receipts)}

Review Criteria:
1. Is the business justification clear?
2. Are the receipts valid and legitimate?
3. Is the amount reasonable for this expense?
4. Does it follow company policy?"""

        response = self.client.system_one(
            state=state,
            questions={
                "approve": Noul(
                    instructions="Should this expense be approved by the manager?"
                ),
                "justification_clarity": Score(
                    instructions="How clear is the business justification?",
                    criteria=[
                        "Unclear or missing justification",
                        "Somewhat clear justification",
                        "Clear and compelling justification",
                    ],
                ),
                "receipt_validity": Score(
                    instructions="How valid and legitimate are the receipts?",
                    criteria=[
                        "Receipts appear incomplete or questionable",
                        "Receipts are adequate",
                        "Receipts are complete and legitimate",
                    ],
                ),
                "amount_reasonableness": Noul(
                    instructions="Is the amount reasonable for the stated purpose?"
                ),
            },
        )

        is_approved = response.answers["approve"].noul == 1.0
        notes = f"Manager review: Justification clarity {response.answers['justification_clarity'].score}/2, Receipt validity {response.answers['receipt_validity'].score}/2"

        return is_approved, notes


class ExecutiveReviewer:
    """Gate 5: Executive (VP) review for high-value expenses"""

    def __init__(self, client: TypeSafeClient):
        self.client = client

    def review(self, expense: ExpenseReport, manager_notes: str) -> Tuple[bool, str]:
        """Executive review of high-value expenses using TypeSafe API"""
        state = f"""Executive (VP) Review for High-Value Expense:
Employee: {expense.employee_name}
Amount: {expense.amount} INR (HIGH VALUE EXPENSE)
Description: {expense.description}
Manager Approval: Yes (already approved by direct manager)
Manager Notes: {manager_notes}

Executive Review Criteria:
1. Alignment with departmental budget
2. Strategic business value
3. No policy violations
4. Amount seems reasonable for the business purpose"""

        response = self.client.system_one(
            state=state,
            questions={
                "executive_approve": Noul(
                    instructions="Should this high-value expense be approved by the VP?"
                ),
                "budget_alignment": Score(
                    instructions="How well does this align with departmental budget?",
                    criteria=[
                        "Poor alignment, potential budget concerns",
                        "Adequate alignment",
                        "Strong alignment with budget",
                    ],
                ),
                "business_value": Score(
                    instructions="What is the strategic business value?",
                    criteria=[
                        "Low strategic value",
                        "Moderate business value",
                        "High strategic importance",
                    ],
                ),
                "policy_violations": Noul(
                    instructions="Are there any policy violations?"
                ),
            },
        )

        is_approved = response.answers["executive_approve"].noul == 1.0
        has_violations = response.answers["policy_violations"].noul == 1.0
        notes = f"VP Review: Budget alignment {response.answers['budget_alignment'].score}/2, Business value {response.answers['business_value'].score}/2, Policy violations: {has_violations}"

        return is_approved, notes


class FinanceProcessor:
    """Gate 6: Finance processing and payment"""

    def __init__(self, client: TypeSafeClient):
        self.client = client

    def process(self, expense: ExpenseReport) -> Dict:
        """Process payment for approved expense using TypeSafe API"""
        state = f"""Finance Processing for Approved Expense:
Employee: {expense.employee_name}
Amount: {expense.amount} INR
Description: {expense.description}
Approval Status: Approved

Processing Verification:
1. All required documentation present
2. Amount is reasonable and justified
3. No processing concerns"""

        response = self.client.system_one(
            state=state,
            questions={
                "ready_for_payment": Noul(
                    instructions="Is this expense ready for payment processing?"
                ),
                "processing_priority": Choice(
                    instructions="What processing priority should this have?",
                    criteria={
                        "standard": "Standard processing (5-7 business days)",
                        "expedited": "Expedited processing (2-3 business days)",
                        "urgent": "Urgent processing (1 business day)",
                    },
                ),
            },
        )

        is_ready = response.answers["ready_for_payment"].noul == 1.0
        priority = response.answers["processing_priority"].choice

        # Generate payment confirmation
        return {
            "bank_transfer_initiated": is_ready,
            "transaction_id": f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "payment_date": (datetime.now()).strftime("%Y-%m-%d"),
            "confirmation_message": f"Payment will be processed with {priority} priority",
        }


# ==================== Workflow Orchestrator ====================


class ExpenseWorkflow:
    """Main workflow orchestrator"""

    def __init__(self):
        """Initialize the workflow with TypeSafe AI (Jev) API"""
        self.client = TypeSafeClient()  # Reads TYPESAFE_API_KEY from environment
        self.receipt_validator = ReceiptValidator(self.client)
        self.manager_reviewer = ManagerReviewer(self.client)
        self.executive_reviewer = ExecutiveReviewer(self.client)
        self.finance_processor = FinanceProcessor(self.client)

    def process_expense(self, expense: ExpenseReport) -> ExpenseReport:
        """Main workflow process"""
        print(f"Processing expense: {expense.id}")

        # GATE 1: Receipt Check
        is_complete, message = self.receipt_validator.validate(expense)
        if not is_complete:
            expense.status = WorkflowStatus.RECEIPTS_PENDING
            print(f"GATE 1: Missing receipts - {message}")
            return expense

        # GATE 2: Amount Threshold
        routing = ExpenseLimitGate.get_routing(expense.amount)
        print(f"GATE 2: Amount {expense.amount} INR - Route: {routing}")

        if routing == "AUTO_APPROVE":
            expense.status = WorkflowStatus.AUTO_APPROVED
            print("GATE 2: Auto-approved (amount < 5000)")
            return self._finalize_payment(expense)

        # GATE 3: Manager Review
        approved, notes = self.manager_reviewer.review(expense)
        expense.manager_notes = notes
        if not approved:
            expense.status = WorkflowStatus.MANAGER_REJECTED
            print(f"GATE 3: Rejected - {notes}")
            return expense

        print(f"GATE 3: Approved - {notes}")

        # GATE 4: High-Value Audit
        requires_vp = ExpenseLimitGate.requires_vp_approval(expense.amount)
        if requires_vp:
            expense.status = WorkflowStatus.EXECUTIVE_REVIEW
            print("GATE 4: High-value expense - VP approval required")

            # GATE 5: VP Review
            vp_approved, vp_notes = self.executive_reviewer.review(expense, notes)
            expense.vp_notes = vp_notes
            if not vp_approved:
                print(f"GATE 5: VP rejected - {vp_notes}")
                return expense

            print(f"GATE 5: VP approved - {vp_notes}")
        else:
            print("GATE 4: Direct to finance")

        # GATE 6: Finance Processing
        return self._finalize_payment(expense)

    def _finalize_payment(self, expense: ExpenseReport) -> ExpenseReport:
        """Process final payment"""
        payment_result = self.finance_processor.process(expense)
        expense.status = WorkflowStatus.COMPLETED
        print(
            f"GATE 6: Payment processed - Transaction: {payment_result.get('transaction_id')}"
        )
        return expense


# ==================== Workflow Report ====================


def print_result(expense: ExpenseReport):
    """Print workflow result"""
    print("\n" + "=" * 50)
    print("EXPENSE WORKFLOW RESULT")
    print("=" * 50)
    print(f"Expense ID: {expense.id}")
    print(f"Employee: {expense.employee_name}")
    print(f"Amount: {expense.amount} INR")
    print(f"Status: {expense.status.value}")
    if expense.manager_notes:
        print(f"Manager Notes: {expense.manager_notes}")
    if expense.vp_notes:
        print(f"VP Notes: {expense.vp_notes}")
    print("=" * 50 + "\n")


# ==================== Example Usage ====================


def main():
    """Demonstrate the expense workflow with different scenarios"""

    # Scenario 1: Small expense < ₹2,000 (Auto-approved, only 1 receipt needed)
    print("\n" + "=" * 60)
    print("SCENARIO 1: Small Expense < ₹2,000")
    print("=" * 60)
    receipts_1 = [
        Receipt(
            id="R1", name="Coffee", category="Food", timestamp="2025-01-10T09:00:00Z"
        ),
    ]
    expense_1 = ExpenseReport(
        id="EXP-001",
        employee_id="EMP-123",
        employee_name="John Doe",
        amount=1500,
        description="Team lunch",
        receipts=receipts_1,
        submission_date=datetime.now().isoformat(),
    )
    workflow = ExpenseWorkflow()
    result_1 = workflow.process_expense(expense_1)
    print_result(result_1)

    # Scenario 2: Medium expense ₹2,000-₹10,000 (Multiple receipts required)
    print("\n" + "=" * 60)
    print("SCENARIO 2: Medium Expense ₹2,000-₹10,000 (Multiple receipts needed)")
    print("=" * 60)
    receipts_2 = [
        Receipt(
            id="R1",
            name="Restaurant Invoice",
            category="Food",
            timestamp="2025-01-10T19:00:00Z",
        ),
        Receipt(
            id="R2",
            name="Beverages Receipt",
            category="Drinks",
            timestamp="2025-01-10T19:15:00Z",
        ),
    ]
    expense_2 = ExpenseReport(
        id="EXP-002",
        employee_id="EMP-123",
        employee_name="John Doe",
        amount=5000,
        description="Client dinner with team",
        receipts=receipts_2,
        submission_date=datetime.now().isoformat(),
    )
    result_2 = workflow.process_expense(expense_2)
    print_result(result_2)

    # Scenario 3: High-value expense > ₹10,000 (Itemized + payment proof required)
    print("\n" + "=" * 60)
    print("SCENARIO 3: High-Value Expense > ₹10,000")
    print("=" * 60)
    receipts_3 = [
        Receipt(
            id="R1",
            name="Hotel Invoice",
            category="Accommodation",
            timestamp="2025-01-10T15:00:00Z",
        ),
        Receipt(
            id="R2",
            name="Payment Proof - Bank Transfer",
            category="Payment",
            timestamp="2025-01-10T16:00:00Z",
        ),
    ]
    expense_3 = ExpenseReport(
        id="EXP-003",
        employee_id="EMP-124",
        employee_name="Jane Smith",
        amount=35000,
        description="Business conference accommodation",
        receipts=receipts_3,
        submission_date=datetime.now().isoformat(),
    )
    result_3 = workflow.process_expense(expense_3)
    print_result(result_3)


if __name__ == "__main__":
    main()


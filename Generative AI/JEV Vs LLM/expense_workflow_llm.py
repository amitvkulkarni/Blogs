"""
Expense Reimbursement Workflow
Implements a multi-stage approval process using LangChain with Azure OpenAI
"""

import os
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

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

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expense receipt validator. Analyze the expense report and determine if all required receipts are present.
            
Expense Amount: {amount} INR
Number of Receipts: {receipt_count}
Receipt Details: {receipts}

Required receipts:
- For amounts < ₹2,000: At least 1 receipt required
- For amounts ₹2,000-₹10,000: All individual items need receipts
- For amounts > ₹10,000: Itemized receipt AND payment proof required

Is the receipt documentation complete? Respond with JSON: {{"complete": true/false, "missing": ["list of missing items"], "reason": "brief reason"}}"""
        )

    def validate(self, expense: ExpenseReport) -> Tuple[bool, str]:
        """Validate receipts for the expense"""
        receipts_info = json.dumps([asdict(r) for r in expense.receipts])
        message = self.prompt.format_prompt(
            amount=expense.amount,
            receipt_count=len(expense.receipts),
            receipts=receipts_info,
        )
        response = self.llm.invoke([HumanMessage(content=message.to_string())])
        result = self._parse_response(response.content)
        return result["complete"], result.get("reason", "Validation completed")

    @staticmethod
    def _parse_response(content: str) -> Dict:
        """Extract JSON from response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {
                "complete": False,
                "missing": ["Unable to parse"],
                "reason": "Response parsing failed",
            }


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

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are a direct manager reviewing an expense report for your team member.

Employee: {employee_name}
Amount: {amount} INR
Description: {description}
Receipts: {receipt_count}

Consider:
1. Is the business justification clear?
2. Are the receipts valid and legitimate?
3. Is the amount reasonable for this expense?
4. Does it follow company policy?

Provide your decision in JSON format: {{"approved": true/false, "notes": "your review notes"}}"""
        )

    def review(self, expense: ExpenseReport) -> Tuple[bool, str]:
        """Review expense and return decision"""
        message = self.prompt.format_prompt(
            employee_name=expense.employee_name,
            amount=expense.amount,
            description=expense.description,
            receipt_count=len(expense.receipts),
        )
        response = self.llm.invoke([HumanMessage(content=message.to_string())])
        result = self._parse_response(response.content)
        return result["approved"], result["notes"]

    @staticmethod
    def _parse_response(content: str) -> Dict:
        """Extract JSON from response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"approved": False, "notes": "Unable to parse manager response"}


class ExecutiveReviewer:
    """Gate 5: Executive (VP) review for high-value expenses"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are the VP reviewing a high-value expense for final approval.

Employee: {employee_name}
Amount: {amount} INR (HIGH VALUE EXPENSE)
Description: {description}
Manager Approval: Yes (already approved by direct manager)

This is a high-value expense requiring executive oversight. Verify:
1. Alignment with departmental budget
2. Strategic business value
3. No policy violations
4. Amount seems reasonable for the business purpose

Decision in JSON: {{"approved": true/false, "notes": "executive review notes"}}"""
        )

    def review(self, expense: ExpenseReport, manager_notes: str) -> Tuple[bool, str]:
        """Executive review of high-value expenses"""
        message = self.prompt.format_prompt(
            employee_name=expense.employee_name,
            amount=expense.amount,
            description=expense.description,
        )
        response = self.llm.invoke([HumanMessage(content=message.to_string())])
        result = self._parse_response(response.content)
        return result["approved"], result["notes"]

    @staticmethod
    def _parse_response(content: str) -> Dict:
        """Extract JSON from response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"approved": False, "notes": "Unable to parse VP response"}


class FinanceProcessor:
    """Gate 6: Finance processing and payment"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are the Finance team processing approved expenses.

Employee: {employee_name}
Amount: {amount} INR
Description: {description}
Approval Status: Approved

Generate a payment processing summary in JSON format:
{{"bank_transfer_initiated": true, "transaction_id": "TXN-12345", "payment_date": "2025-01-15", "confirmation_message": "Payment will be processed within 3-5 business days"}}"""
        )

    def process(self, expense: ExpenseReport) -> Dict:
        """Process payment for approved expense"""
        message = self.prompt.format_prompt(
            employee_name=expense.employee_name,
            amount=expense.amount,
            description=expense.description,
        )
        response = self.llm.invoke([HumanMessage(content=message.to_string())])
        result = self._parse_response(response.content)
        return result

    @staticmethod
    def _parse_response(content: str) -> Dict:
        """Extract JSON from response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {
                "bank_transfer_initiated": True,
                "transaction_id": "TXN-AUTO",
                "payment_date": datetime.now().strftime("%Y-%m-%d"),
                "confirmation_message": "Payment processing initiated",
            }


# ==================== Workflow Orchestrator ====================


class ExpenseWorkflow:
    """Main workflow orchestrator"""

    def __init__(self):
        """Initialize the workflow with Azure OpenAI"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.receipt_validator = ReceiptValidator(self.llm)
        self.manager_reviewer = ManagerReviewer(self.llm)
        self.executive_reviewer = ExecutiveReviewer(self.llm)
        self.finance_processor = FinanceProcessor(self.llm)

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


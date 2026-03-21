"""
Pydantic AI agent with typed action tools for Tripletex.
The LLM provides structured args; Pydantic validates; code handles the API contract.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel

from tripletex_client import TripletexClient
from actions import ACTIONS

log = logging.getLogger(__name__)

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-5.4")
log.info(f"Agent using model: {MODEL}")


# --- Dependencies (injected into tools) ---

@dataclass
class AgentDeps:
    client: TripletexClient


# --- Tool argument models (Pydantic enforces the schema) ---

class DiscoverSandboxArgs(BaseModel):
    """No args needed — discovers what exists in the sandbox."""
    pass


class CreateEmployeeArgs(BaseModel):
    firstName: str
    lastName: str
    email: Optional[str] = None
    userType: str = Field(default="STANDARD", description="STANDARD, EXTENDED (admin), or NO_ACCESS")
    dateOfBirth: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    phoneNumberMobile: Optional[str] = None
    nationalIdentityNumber: Optional[str] = Field(default=None, description="Norwegian national ID (personnummer), 11 digits")
    departmentId: Optional[int] = Field(default=None, description="Department ID if already known")
    departmentName: Optional[str] = Field(default=None, description="Department name to find before employee creation")
    address: Optional[dict] = Field(default=None, description='Employee address, e.g. {"addressLine1":"...", "postalCode":"...", "city":"..."}')
    startDate: Optional[str] = Field(default=None, description="Employment start date YYYY-MM-DD")
    endDate: Optional[str] = Field(default=None, description="Employment end date YYYY-MM-DD")
    employmentType: Optional[str] = Field(default=None, description="ORDINARY, MARITIME, FREELANCE")
    percentageOfFullTimeEquivalent: Optional[float] = Field(default=None, description="Employment percentage, e.g. 100.0 for full-time, 50.0 for half")
    annualSalary: Optional[float] = Field(default=None, description="Annual salary (årslønn) in NOK")
    occupationCode: Optional[str] = Field(default=None, description="STYRK occupation code, e.g. '2511' for system developer")
    hoursPerDay: Optional[float] = Field(default=None, description="Standard work hours per day, e.g. 7.5")


class CreateCustomerArgs(BaseModel):
    name: str
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    organizationNumber: Optional[str] = None
    addressLine1: Optional[str] = Field(default=None, description="Street address")
    postalCode: Optional[str] = Field(default=None, description="Postal/ZIP code")
    city: Optional[str] = Field(default=None, description="City name")
    isPrivateIndividual: Optional[bool] = None
    postalAddress: Optional[dict] = Field(default=None, description='{"addressLine1":"...", "postalCode":"...", "city":"..."}')


class CreateSupplierArgs(BaseModel):
    name: str
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    organizationNumber: Optional[str] = None


class CreateProductArgs(BaseModel):
    name: str
    number: Optional[str] = None
    priceExcludingVat: Optional[float] = None
    priceIncludingVat: Optional[float] = None
    vatTypeId: int = Field(default=3, description="3=25% standard, 31=15% food, 32=12% transport, 5=0%")


class CreateDepartmentArgs(BaseModel):
    name: str
    departmentNumber: str


class OrderLine(BaseModel):
    description: str
    count: float = 1
    unitPrice: float = Field(description="Price per unit excluding VAT in NOK")
    vatTypeId: int = Field(default=3, description="3=25%, 31=15%, 32=12%, 5=0%")
    productNumber: Optional[str] = None


class CreateInvoiceArgs(BaseModel):
    customerName: Optional[str] = Field(default=None, description="Customer name to find or create")
    customerId: Optional[int] = Field(default=None, description="Customer ID if already known")
    customerOrgNumber: Optional[str] = None
    invoiceDate: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")
    invoiceDueDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    currencyCode: Optional[str] = Field(default=None, description="Currency code e.g. EUR, USD, SEK. Omit for NOK.")
    orderLines: list[OrderLine]


class CreateOrderArgs(BaseModel):
    customerName: Optional[str] = None
    customerId: Optional[int] = None
    orderDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    deliveryDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    orderLines: list[OrderLine]


class RegisterPaymentArgs(BaseModel):
    invoiceId: Optional[int] = Field(default=None, description="Invoice ID. If not provided, searches by customer name.")
    invoiceNumber: Optional[str] = Field(default=None, description="Invoice number/reference from the prompt or bank statement. Prefer this when an external invoice number is provided.")
    customerName: Optional[str] = Field(default=None, description="Customer name to match the correct invoice")
    amount: float = Field(description="Payment amount in NOK. Use negative for reversal.")
    paidAmountCurrency: Optional[float] = Field(default=None, description="Payment amount in foreign currency (e.g. EUR). Required for foreign currency invoice payments.")
    paymentDate: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")
    paymentTypeId: Optional[int] = Field(default=None, description="Payment type ID. Auto-detected if not provided.")


class CreateCreditNoteArgs(BaseModel):
    invoiceId: Optional[int] = Field(default=None, description="Invoice ID. Searches if not provided.")
    invoiceNumber: Optional[str] = Field(default=None, description="Invoice number/reference from the prompt when available.")
    customerName: Optional[str] = Field(default=None, description="Customer name to match the correct invoice")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")
    comment: Optional[str] = None


class VoucherPosting(BaseModel):
    accountNumber: int = Field(description="Account number from chart of accounts (e.g. 5000 for salary, 1920 for bank)")
    amountGross: float = Field(description="Positive=debit, negative=credit. Postings must sum to zero.")
    vatTypeId: Optional[int] = None
    description: Optional[str] = None
    customerId: Optional[int] = Field(default=None, description="Customer ID if this posting must be linked to a customer")
    supplierId: Optional[int] = Field(default=None, description="Supplier ID if this posting must be linked to a supplier")


class CreateVoucherArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    description: str
    customerId: Optional[int] = Field(default=None, description="Customer ID to attach to customer-related voucher postings")
    customerName: Optional[str] = Field(default=None, description="Customer name to find or create")
    customerOrgNumber: Optional[str] = Field(default=None, description="Customer organization number")
    supplierId: Optional[int] = Field(default=None, description="Supplier ID to attach to supplier-related voucher postings")
    supplierName: Optional[str] = Field(default=None, description="Supplier name to find or create")
    supplierOrgNumber: Optional[str] = Field(default=None, description="Supplier organization number")
    postings: list[VoucherPosting]


class CreateProjectArgs(BaseModel):
    name: str
    number: Optional[str] = None
    customerId: Optional[int] = None
    customerName: Optional[str] = Field(default=None, description="Customer name to find or create")
    customerOrgNumber: Optional[str] = Field(default=None, description="Customer org number")
    projectManagerId: Optional[int] = Field(default=None, description="Employee ID. Uses first employee if not provided.")
    projectManagerEmail: Optional[str] = Field(default=None, description="Project manager email to find existing employee")
    projectManagerName: Optional[str] = Field(default=None, description="Project manager name to find existing employee")
    isInternal: bool = False
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    fixedPrice: Optional[float] = Field(default=None, description="Fixed price amount in NOK. Sets project as fixed-price.")


class ActivateModuleArgs(BaseModel):
    name: str = Field(description="Module name: PROJECT, SMART_PROJECT, TIME_TRACKING, LOGISTICS, OCR, etc.")


class DimensionVoucherPosting(BaseModel):
    accountNumber: int = Field(description="Account number (e.g. 7300, 1920)")
    amountGross: float = Field(description="Positive=debit, negative=credit")
    vatTypeId: Optional[int] = None
    dimensionValueName: Optional[str] = Field(default=None, description="Name of dimension value to link")


class CreateAccountingDimensionArgs(BaseModel):
    """Create a custom accounting dimension with values, optionally post a linked voucher."""
    dimensionName: str = Field(description="Name of the dimension (e.g. 'Marked', 'Produktlinje')")
    description: Optional[str] = None
    values: list[str] = Field(description="Dimension value names (e.g. ['Offentlig', 'Privat'])")
    voucherDate: Optional[str] = Field(default=None, description="YYYY-MM-DD for voucher, if posting requested")
    voucherDescription: Optional[str] = None
    voucherPostings: Optional[list[DimensionVoucherPosting]] = Field(default=None, description="If provided, creates a voucher linked to the dimension")


class UpdateEmployeeArgs(BaseModel):
    employeeId: Optional[int] = None
    firstName: Optional[str] = Field(default=None, description="Search by first name if no ID")
    lastName: Optional[str] = Field(default=None, description="Search by last name if no ID")
    email: Optional[str] = None
    phoneNumberMobile: Optional[str] = None
    dateOfBirth: Optional[str] = None
    userType: Optional[str] = None
    address: Optional[dict] = None


class RegisterTimesheetArgs(BaseModel):
    employeeName: Optional[str] = None
    employeeEmail: Optional[str] = None
    employeeId: Optional[int] = None
    projectName: Optional[str] = None
    projectId: Optional[int] = None
    activityName: Optional[str] = None
    activityId: Optional[int] = None
    hours: float
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    comment: Optional[str] = None


class TravelExpenseItem(BaseModel):
    description: str = Field(description="E.g. 'Flight ticket', 'Taxi', 'Hotel'")
    amount: float = Field(description="Amount in NOK including VAT")


class CreateTravelExpenseArgs(BaseModel):
    employeeName: Optional[str] = None
    employeeEmail: Optional[str] = None
    employeeId: Optional[int] = None
    title: str = Field(description="Trip description, e.g. 'Client visit Oslo'")
    departureDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    returnDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    perDiemDays: Optional[int] = Field(default=None, description="Number of days for per diem")
    perDiemRate: Optional[float] = Field(default=None, description="Daily per diem rate in NOK")
    departure: Optional[str] = Field(default=None, description="Departure city")
    destination: Optional[str] = Field(default=None, description="Destination city")
    expenses: Optional[list[TravelExpenseItem]] = Field(default=None, description="Individual expense items like flights, taxi, hotel")


class ProcessSalaryArgs(BaseModel):
    employeeName: Optional[str] = None
    employeeEmail: Optional[str] = None
    employeeId: Optional[int] = None
    baseSalary: float = Field(description="Monthly base salary in NOK")
    bonus: float = Field(default=0, description="One-time bonus in NOK")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")


class RegisterSupplierInvoiceArgs(BaseModel):
    supplierName: str
    supplierOrgNumber: Optional[str] = None
    supplierId: Optional[int] = None
    invoiceNumber: Optional[str] = None
    amountIncludingVat: float = Field(description="Total amount including VAT")
    accountNumber: Optional[int] = Field(default=None, description="Expense account number (e.g. 6300 for office, 6590 for office services, 7140 for travel)")
    description: Optional[str] = Field(default=None, description="Description of what the invoice is for")
    invoiceDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    dueDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")


class CreateFixedPriceProjectInvoiceArgs(BaseModel):
    projectName: str
    fixedPrice: float = Field(description="Total fixed price in NOK")
    invoicePercent: float = Field(description="Percent of the fixed price to invoice, e.g. 25, 50, 75")
    customerId: Optional[int] = None
    customerName: Optional[str] = None
    customerOrgNumber: Optional[str] = None
    projectId: Optional[int] = None
    projectNumber: Optional[str] = None
    projectManagerId: Optional[int] = None
    projectManagerEmail: Optional[str] = None
    projectManagerName: Optional[str] = None
    startDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    invoiceDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    invoiceDueDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    description: Optional[str] = None


class RegisterTimesheetAndInvoiceArgs(BaseModel):
    hours: float
    hourlyRate: float = Field(description="Hourly billing rate in NOK")
    employeeId: Optional[int] = None
    employeeName: Optional[str] = None
    employeeEmail: Optional[str] = None
    customerId: Optional[int] = None
    customerName: Optional[str] = None
    customerOrgNumber: Optional[str] = None
    projectId: Optional[int] = None
    projectName: Optional[str] = None
    projectManagerId: Optional[int] = None
    projectManagerEmail: Optional[str] = None
    projectManagerName: Optional[str] = None
    activityId: Optional[int] = None
    activityName: Optional[str] = None
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    comment: Optional[str] = None
    invoiceDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    invoiceDueDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    description: Optional[str] = None


class GenericApiCallArgs(BaseModel):
    """Fallback for any API call not covered by typed tools."""
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE")
    path: str = Field(description="API path, e.g. /employee/123")
    params: Optional[dict] = Field(default=None, description="Query parameters")
    body: Optional[dict] = Field(default=None, description="JSON request body")


class RunPythonArgs(BaseModel):
    """Execute Python code for data processing, CSV parsing, calculations, etc."""
    code: str = Field(description="Python code to execute. Has access to 'client' (TripletexClient with auth) and 'httpx'. Print results to stdout.")


# --- Build the agent ---

agent = Agent(
    OpenRouterModel(MODEL),
    deps_type=AgentDeps,
    system_prompt="",  # Set dynamically per run
    retries=1,
    model_settings={"max_tokens": 4096},
)


# --- Register tools ---

async def _safe_action(action_name: str, client, args_dict: dict, max_len: int = 3000) -> str:
    """Run an action with error handling. Never raises — returns error as string."""
    # Circuit breaker: if too many errors already, warn the agent
    if client.error_count >= 10:
        return json.dumps({"error": f"Circuit breaker: {client.error_count} API errors so far. Check if auth is valid or adjust approach.", "hint": "Stop retrying failed patterns"})

    try:
        result = await ACTIONS[action_name](client, args_dict)
        return json.dumps(result, ensure_ascii=False, default=str)[:max_len]
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response'):
            try:
                error_msg = e.response.text[:500]
            except Exception:
                pass
        log.warning(f"Action {action_name} failed: {error_msg[:200]}")
        return json.dumps({"error": error_msg[:500]}, ensure_ascii=False)


@agent.tool
async def discover_sandbox(ctx: RunContext[AgentDeps], args: DiscoverSandboxArgs) -> str:
    """Discover what exists in the sandbox: employees, customers, invoices, departments, payment types."""
    return await _safe_action("discover_sandbox", ctx.deps.client, {})


@agent.tool
async def create_employee(ctx: RunContext[AgentDeps], args: CreateEmployeeArgs) -> str:
    """Create an employee. Resolves department plus employment, salary, and standard work-time details when provided."""
    return await _safe_action("create_employee", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_customer(ctx: RunContext[AgentDeps], args: CreateCustomerArgs) -> str:
    """Create a customer."""
    return await _safe_action("create_customer", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_supplier(ctx: RunContext[AgentDeps], args: CreateSupplierArgs) -> str:
    """Register a supplier."""
    return await _safe_action("create_supplier", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_product(ctx: RunContext[AgentDeps], args: CreateProductArgs) -> str:
    """Create a product. vatTypeId defaults to 3 (25% MVA)."""
    return await _safe_action("create_product", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_department(ctx: RunContext[AgentDeps], args: CreateDepartmentArgs) -> str:
    """Create a department."""
    return await _safe_action("create_department", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_invoice(ctx: RunContext[AgentDeps], args: CreateInvoiceArgs) -> str:
    """Create an invoice. Handles bank account setup, customer lookup/creation, and order line formatting automatically."""
    return await _safe_action("create_invoice", ctx.deps.client, args.model_dump(exclude_none=True), 3000)


@agent.tool
async def create_order(ctx: RunContext[AgentDeps], args: CreateOrderArgs) -> str:
    """Create an order (not yet invoiced)."""
    return await _safe_action("create_order", ctx.deps.client, args.model_dump(exclude_none=True), 3000)


@agent.tool
async def register_payment(ctx: RunContext[AgentDeps], args: RegisterPaymentArgs) -> str:
    """Register payment on an invoice. Use negative amount for reversal. Finds invoice and payment type automatically."""
    return await _safe_action("register_payment", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_credit_note(ctx: RunContext[AgentDeps], args: CreateCreditNoteArgs) -> str:
    """Create a credit note to cancel an invoice. Finds the invoice automatically if invoiceId not provided."""
    return await _safe_action("create_credit_note", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_voucher(ctx: RunContext[AgentDeps], args: CreateVoucherArgs) -> str:
    """Create a journal entry / voucher. Postings must balance (sum to zero). Uses account numbers from chart of accounts."""
    return await _safe_action("create_voucher", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_project(ctx: RunContext[AgentDeps], args: CreateProjectArgs) -> str:
    """Create a project. Activates the PROJECT module automatically."""
    return await _safe_action("create_project", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def activate_module(ctx: RunContext[AgentDeps], args: ActivateModuleArgs) -> str:
    """Activate a Tripletex module (PROJECT, SMART_PROJECT, TIME_TRACKING, etc.)."""
    return await _safe_action("activate_module", ctx.deps.client, args.model_dump(), 1000)


@agent.tool
async def create_travel_expense(ctx: RunContext[AgentDeps], args: CreateTravelExpenseArgs) -> str:
    """Create a travel expense report with per diem. Finds employee automatically."""
    return await _safe_action("create_travel_expense", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def create_accounting_dimension(ctx: RunContext[AgentDeps], args: CreateAccountingDimensionArgs) -> str:
    """Create a custom accounting dimension with values. Can optionally post a voucher linked to the dimension."""
    return await _safe_action("create_accounting_dimension", ctx.deps.client, args.model_dump(exclude_none=True), 3000)


@agent.tool
async def update_employee(ctx: RunContext[AgentDeps], args: UpdateEmployeeArgs) -> str:
    """Update an existing employee's details. Finds by name or ID."""
    return await _safe_action("update_employee", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def register_timesheet(ctx: RunContext[AgentDeps], args: RegisterTimesheetArgs) -> str:
    """Register hours on a timesheet for an employee on a project activity. Activates required modules automatically."""
    return await _safe_action("register_timesheet", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def process_salary(ctx: RunContext[AgentDeps], args: ProcessSalaryArgs) -> str:
    """Process salary/payroll for an employee. Creates a voucher with salary cost, tax withholding, and bank payment postings."""
    return await _safe_action("process_salary", ctx.deps.client, args.model_dump(exclude_none=True), 2000)


@agent.tool
async def register_supplier_invoice(ctx: RunContext[AgentDeps], args: RegisterSupplierInvoiceArgs) -> str:
    """Register an incoming supplier invoice. Creates supplier if needed, posts the invoice as a voucher."""
    return await _safe_action("register_supplier_invoice", ctx.deps.client, args.model_dump(exclude_none=True), 3000)


@agent.tool
async def create_fixed_price_project_invoice(ctx: RunContext[AgentDeps], args: CreateFixedPriceProjectInvoiceArgs) -> str:
    """Create or find a project, set fixed price, then invoice a percentage of that fixed price."""
    return await _safe_action("create_fixed_price_project_invoice", ctx.deps.client, args.model_dump(exclude_none=True), 3500)


@agent.tool
async def register_timesheet_and_invoice(ctx: RunContext[AgentDeps], args: RegisterTimesheetAndInvoiceArgs) -> str:
    """Register project hours and then create an invoice for the logged hours times the hourly rate."""
    return await _safe_action("register_timesheet_and_invoice", ctx.deps.client, args.model_dump(exclude_none=True), 3500)


@agent.tool
async def generic_api_call(ctx: RunContext[AgentDeps], args: GenericApiCallArgs) -> str:
    """Fallback: make any Tripletex API call. Use only when no typed tool fits. Will reject calls that should use typed tools."""
    args_dict = args.model_dump(exclude_none=True)
    method = args_dict.get("method", "GET").upper()
    path = args.path.lower()
    # Redirect to typed tools
    redirects = {
        "/salary": "Use process_salary tool instead",
        "/payroll": "Use process_salary tool instead",
        "/travelexpense": "Use create_travel_expense tool instead",
        "/incominginvoice": "Use register_supplier_invoice tool instead",
        "/supplierinvoice": "Use register_supplier_invoice tool instead",
        "/product": "Use create_product tool instead",
        "/department": "Use create_department tool instead",
        "/company/salesmodules": "Module activation is handled automatically by typed tools (create_project, register_timesheet)",
        "/employee/employment": "Use create_employee tool with startDate, annualSalary, percentageOfFullTimeEquivalent instead. It handles employment creation automatically.",
    }
    if method == "POST" and path == "/activity":
        body = args_dict.get("body") or {}
        activity_type = body.get("activityType", "")
        # Ensure name is non-blank (Tripletex requires it)
        name = body.get("name") or body.get("displayName") or body.get("description") or "General"
        body["name"] = name
        # PROJECT_SPECIFIC activities MUST go through /project/projectActivity
        if activity_type == "PROJECT_SPECIFIC_ACTIVITY" or "project" in str(body).lower():
            body["activityType"] = "PROJECT_SPECIFIC_ACTIVITY"
            project_ref = body.pop("project", None)
            project_id = None
            if project_ref and isinstance(project_ref, dict):
                project_id = project_ref.get("id")
            if project_id:
                link_body = {"project": {"id": project_id}, "activity": body}
                link_args = {"method": "POST", "path": "/project/projectActivity", "body": link_body}
                return await _safe_action("generic_api_call", ctx.deps.client, link_args, 4000)
            else:
                # No project ID - fall back to GENERAL_ACTIVITY via /activity
                body["activityType"] = "GENERAL_ACTIVITY"
                args_dict["body"] = body
                return await _safe_action("generic_api_call", ctx.deps.client, args_dict, 4000)
        else:
            # Not project-specific - use GENERAL_ACTIVITY via /activity
            body.setdefault("activityType", "GENERAL_ACTIVITY")
            args_dict["body"] = body
            return await _safe_action("generic_api_call", ctx.deps.client, args_dict, 4000)
    # Fix wrong activity endpoint: /project/{id}/activity -> create via /project/projectActivity
    import re
    project_activity_match = re.match(r"/project/(\d+)/activity", path)
    if project_activity_match:
        project_id = int(project_activity_match.group(1))
        body = args_dict.get("body") or {}
        body.setdefault("activityType", "PROJECT_SPECIFIC_ACTIVITY")
        name = body.get("name") or body.get("displayName") or body.get("description") or "General"
        body["name"] = name
        link_body = {"project": {"id": project_id}, "activity": body}
        link_args = {"method": "POST", "path": "/project/projectActivity", "body": link_body}
        return await _safe_action("generic_api_call", ctx.deps.client, link_args, 4000)
    for pattern, msg in redirects.items():
        if pattern in path:
            return json.dumps({"error": msg, "hint": "Do NOT use generic_api_call for this. Call the typed tool directly."})
    return await _safe_action("generic_api_call", ctx.deps.client, args_dict, 4000)


# --- Core prompt ---

CORE_PROMPT = """You are an expert Tripletex accounting agent. Complete the task by calling the available tools.

MANDATORY TOOL ROUTING — you MUST use these typed tools, NEVER generic_api_call for these tasks:
- Products → create_product
- Departments → create_department
- Supplier/incoming invoices → register_supplier_invoice
- Salary/payroll → process_salary
- Time tracking/hours only → register_timesheet
- Time tracking/hours + invoice → register_timesheet_and_invoice
- Travel expenses → create_travel_expense
- Plain projects → create_project
- Fixed-price project + invoice percentage → create_fixed_price_project_invoice
- Invoices → create_invoice
- Credit notes → create_credit_note
- Payments/reversals → register_payment
- Dimensions → create_accounting_dimension

generic_api_call is ONLY for tasks with no matching typed tool above. If you use generic_api_call for any of the above, it will fail.

MULTI-STEP TASK PATTERNS:
- "Set fixed price on project + invoice X%": call create_fixed_price_project_invoice
- "Log hours + generate project invoice": call register_timesheet_and_invoice
- "Create invoice + register payment": call create_invoice, note the amount, THEN call register_payment with that amount
- "Payment was returned/reversed": call register_payment with NEGATIVE amount
- "Foreign currency invoice + payment + agio": create_invoice with currencyCode (e.g. "EUR"), then register_payment with BOTH amount (NOK = foreign × payment_rate) AND paidAmountCurrency (the foreign amount, e.g. EUR). CRITICAL: paidAmountCurrency is REQUIRED for foreign currency payments or it will fail. Then create_voucher to book the exchange rate difference (agio): debit account 8060 (agio gain) or credit 8160 (agio loss), balanced against 1500 (customer receivables). Agio = (payment_rate - invoice_rate) × foreign_amount.
- "Bank statement / reconciliation / CSV payment matching": use register_payment with invoiceNumber when the statement includes an invoice reference; do not guess Tripletex invoice IDs from raw numbers
- "Ledger review / find errors in vouchers": use discover_sandbox + generic_api_call GET /ledger/voucher to find vouchers, then create_voucher to post corrections. IMPORTANT: if correcting postings on account 2400 (leverandørgjeld/accounts payable), you MUST include supplierName in create_voucher args so the supplier reference is added to postings. Similarly for 1500 (kundefordringer), include customerName.
- "General ledger analysis / cost increase tasks": use generic_api_call GET /ledger/posting with explicit dateFrom/dateTo and analyze the returned postings by account. DO NOT guess voucher IDs or iterate GET /ledger/voucher/{id} unless you already have a confirmed existing ID from list data. Only create vouchers if the prompt explicitly asks for corrections/bookings.

KEY FACTS:
- Fresh sandbox: 1 employee, 1 department, no customers/invoices. Some tasks have pre-populated data.
- ONLY call discover_sandbox if you need to find existing entities (invoices, customers, etc). Skip it for simple creation tasks.
- For invoices: create_invoice handles bank account setup and customer creation automatically.
- CRITICAL: For admin/administrator/Administratorrolle/tilgang/full access roles: ALWAYS set userType="EXTENDED" in create_employee. For onboarding/prompts from contracts or PDFs, also pass departmentName, startDate, percentageOfFullTimeEquivalent, annualSalary, occupationCode, and hoursPerDay whenever the prompt provides them.
- VAT types: 3=25% standard (default), 31=15% food, 32=12% transport, 5=0%.
- Dates must be YYYY-MM-DD format.
- IMPORTANT: If a tool returns an error, DO NOT retry the same call more than once. Read the error, adjust, or try a different approach.
- If you see "403 Forbidden" or auth errors on multiple calls, STOP — the session may be invalid.

Complete the task efficiently with ONE tool call when possible, then stop."""


def build_system_prompt(playbook: dict | None = None) -> str:
    """Build system prompt: core + optional playbook tips."""
    parts = [CORE_PROMPT]
    if playbook and playbook.get("tips"):
        parts.append(f"\nTask hint: {playbook['tips']}")
    return "\n".join(parts)


# --- Run the agent ---

async def run_agent(api_client: TripletexClient, prompt: str, files: list = None, playbook: dict = None) -> dict:
    """Run the Pydantic AI agent."""
    start_time = time.time()

    system_prompt = build_system_prompt(playbook)

    user_content = prompt
    if files:
        user_content += "\n\n## ATTACHED FILES\n"
        for f in files:
            user_content += f"\n### {f['filename']} ({f['mime_type']})\n"
            # Extract text content from PDFs and CSVs
            try:
                import base64
                raw = base64.b64decode(f.get('content_base64', ''))
                if f['mime_type'] == 'application/pdf' or f['filename'].endswith('.pdf'):
                    import fitz
                    doc = fitz.open(stream=raw, filetype="pdf")
                    text = "\n".join(page.get_text() for page in doc)
                    user_content += f"PDF TEXT CONTENT:\n{text[:3000]}\n"
                elif f['mime_type'] == 'text/csv' or f['filename'].endswith('.csv'):
                    csv_text = raw.decode('utf-8', errors='replace')
                    user_content += f"CSV CONTENT:\n{csv_text[:3000]}\n"
                elif f['mime_type'].startswith('image/'):
                    user_content += f"(Image file - {len(raw)} bytes)\n"
                else:
                    text = raw.decode('utf-8', errors='replace')
                    user_content += f"TEXT CONTENT:\n{text[:2000]}\n"
            except Exception as e:
                user_content += f"(Could not extract content: {e})\n"

    deps = AgentDeps(client=api_client)

    # Override system prompt for this run
    agent._system_prompts = (system_prompt,)

    result = await agent.run(user_content, deps=deps)

    elapsed = time.time() - start_time
    stats = api_client.get_stats()

    # Extract output - try .output first (newer pydantic-ai), then .data
    output = getattr(result, 'output', None) or getattr(result, 'data', None) or str(result)

    return {
        "iterations": stats["total_calls"],
        "elapsed_seconds": round(elapsed, 1),
        "api_calls": stats["total_calls"],
        "api_errors": stats["errors_4xx"],
        "final_message": output if isinstance(output, str) else str(output),
    }

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


class CreateCustomerArgs(BaseModel):
    name: str
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    organizationNumber: Optional[str] = None
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
    orderLines: list[OrderLine]


class CreateOrderArgs(BaseModel):
    customerName: Optional[str] = None
    customerId: Optional[int] = None
    orderDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    deliveryDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    orderLines: list[OrderLine]


class RegisterPaymentArgs(BaseModel):
    invoiceId: Optional[int] = Field(default=None, description="Invoice ID. If not provided, searches for invoices.")
    amount: float = Field(description="Payment amount. Use negative for reversal.")
    paymentDate: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")
    paymentTypeId: Optional[int] = Field(default=None, description="Payment type ID. Auto-detected if not provided.")


class CreateCreditNoteArgs(BaseModel):
    invoiceId: Optional[int] = Field(default=None, description="Invoice ID. Searches if not provided.")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD, defaults to today")
    comment: Optional[str] = None


class VoucherPosting(BaseModel):
    accountNumber: int = Field(description="Account number from chart of accounts (e.g. 5000 for salary, 1920 for bank)")
    amountGross: float = Field(description="Positive=debit, negative=credit. Postings must sum to zero.")
    vatTypeId: Optional[int] = None
    description: Optional[str] = None


class CreateVoucherArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    description: str
    postings: list[VoucherPosting]


class CreateProjectArgs(BaseModel):
    name: str
    number: Optional[str] = None
    customerId: Optional[int] = None
    projectManagerId: Optional[int] = Field(default=None, description="Employee ID. Uses first employee if not provided.")
    isInternal: bool = False
    startDate: Optional[str] = None
    endDate: Optional[str] = None


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


class GenericApiCallArgs(BaseModel):
    """Fallback for any API call not covered by typed tools."""
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE")
    path: str = Field(description="API path, e.g. /employee/123")
    params: Optional[dict] = Field(default=None, description="Query parameters")
    body: Optional[dict] = Field(default=None, description="JSON request body")


# --- Build the agent ---

agent = Agent(
    OpenRouterModel(MODEL),
    deps_type=AgentDeps,
    system_prompt="",  # Set dynamically per run
    retries=2,
)


# --- Register tools ---

@agent.tool
async def discover_sandbox(ctx: RunContext[AgentDeps], args: DiscoverSandboxArgs) -> str:
    """Discover what exists in the sandbox: employees, customers, invoices, departments, payment types."""
    result = await ACTIONS["discover_sandbox"](ctx.deps.client, {})
    return json.dumps(result, ensure_ascii=False, default=str)[:4000]


@agent.tool
async def create_employee(ctx: RunContext[AgentDeps], args: CreateEmployeeArgs) -> str:
    """Create an employee. Handles department lookup automatically."""
    result = await ACTIONS["create_employee"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_customer(ctx: RunContext[AgentDeps], args: CreateCustomerArgs) -> str:
    """Create a customer."""
    result = await ACTIONS["create_customer"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_supplier(ctx: RunContext[AgentDeps], args: CreateSupplierArgs) -> str:
    """Register a supplier."""
    result = await ACTIONS["create_supplier"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_product(ctx: RunContext[AgentDeps], args: CreateProductArgs) -> str:
    """Create a product. vatTypeId defaults to 3 (25% MVA)."""
    result = await ACTIONS["create_product"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_department(ctx: RunContext[AgentDeps], args: CreateDepartmentArgs) -> str:
    """Create a department."""
    result = await ACTIONS["create_department"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_invoice(ctx: RunContext[AgentDeps], args: CreateInvoiceArgs) -> str:
    """Create an invoice. Handles bank account setup, customer lookup/creation, and order line formatting automatically."""
    result = await ACTIONS["create_invoice"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:3000]


@agent.tool
async def create_order(ctx: RunContext[AgentDeps], args: CreateOrderArgs) -> str:
    """Create an order (not yet invoiced)."""
    result = await ACTIONS["create_order"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:3000]


@agent.tool
async def register_payment(ctx: RunContext[AgentDeps], args: RegisterPaymentArgs) -> str:
    """Register payment on an invoice. Use negative amount for reversal. Finds invoice and payment type automatically."""
    result = await ACTIONS["register_payment"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_credit_note(ctx: RunContext[AgentDeps], args: CreateCreditNoteArgs) -> str:
    """Create a credit note to cancel an invoice. Finds the invoice automatically if invoiceId not provided."""
    result = await ACTIONS["create_credit_note"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_voucher(ctx: RunContext[AgentDeps], args: CreateVoucherArgs) -> str:
    """Create a journal entry / voucher. Postings must balance (sum to zero). Uses account numbers from chart of accounts."""
    result = await ACTIONS["create_voucher"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def create_project(ctx: RunContext[AgentDeps], args: CreateProjectArgs) -> str:
    """Create a project. Activates the PROJECT module automatically."""
    result = await ACTIONS["create_project"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def activate_module(ctx: RunContext[AgentDeps], args: ActivateModuleArgs) -> str:
    """Activate a Tripletex module (PROJECT, SMART_PROJECT, TIME_TRACKING, etc.)."""
    result = await ACTIONS["activate_module"](ctx.deps.client, args.model_dump())
    return json.dumps(result, ensure_ascii=False, default=str)[:1000]


@agent.tool
async def create_accounting_dimension(ctx: RunContext[AgentDeps], args: CreateAccountingDimensionArgs) -> str:
    """Create a custom accounting dimension with values. Can optionally post a voucher linked to the dimension."""
    result = await ACTIONS["create_accounting_dimension"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:3000]


@agent.tool
async def update_employee(ctx: RunContext[AgentDeps], args: UpdateEmployeeArgs) -> str:
    """Update an existing employee's details. Finds by name or ID."""
    result = await ACTIONS["update_employee"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def register_timesheet(ctx: RunContext[AgentDeps], args: RegisterTimesheetArgs) -> str:
    """Register hours on a timesheet for an employee on a project activity. Activates required modules automatically."""
    result = await ACTIONS["register_timesheet"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:2000]


@agent.tool
async def generic_api_call(ctx: RunContext[AgentDeps], args: GenericApiCallArgs) -> str:
    """Fallback: make any Tripletex API call. Use only when no typed tool fits."""
    result = await ACTIONS["generic_api_call"](ctx.deps.client, args.model_dump(exclude_none=True))
    return json.dumps(result, ensure_ascii=False, default=str)[:4000]


# --- Core prompt ---

CORE_PROMPT = """You are an expert Tripletex accounting agent. Complete the task by calling the available tools.

KEY FACTS:
- Fresh sandbox: 1 employee, 1 department, no customers/invoices. Some tasks have pre-populated data.
- Start with discover_sandbox to see what exists.
- For invoices: create_invoice handles bank account setup and customer creation automatically.
- For admin/administrator roles: use userType="EXTENDED" in create_employee.
- VAT types: 3=25% standard (default), 31=15% food, 32=12% transport, 5=0%.
- Dates must be YYYY-MM-DD format.
- If a typed tool doesn't exist for what you need, use generic_api_call as fallback.

Complete the task efficiently, then stop."""


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
        file_descriptions = "\n".join(
            f"- Attached file: {f['filename']} ({f['mime_type']})"
            for f in files
        )
        user_content += f"\n\nAttached files:\n{file_descriptions}"

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

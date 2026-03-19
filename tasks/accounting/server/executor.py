"""
Task executor. Takes a parsed task and executes it against the Tripletex API.
Routes to the appropriate handler based on task_type.
"""

import logging
from tripletex_client import TripletexClient

log = logging.getLogger(__name__)


async def execute_task(client: TripletexClient, task: dict) -> dict:
    """Execute a parsed task against the Tripletex API."""
    task_type = task.get("task_type", "other")
    fields = task.get("fields", {})
    steps = task.get("steps", [])

    # Multi-step tasks
    if steps:
        return await execute_multistep(client, steps, task)

    # Single-step dispatch
    handler = HANDLERS.get(task_type)
    if handler:
        return await handler(client, fields, task)

    log.warning(f"Unknown task type: {task_type}, attempting generic execution")
    return await handle_other(client, fields, task)


async def execute_multistep(client: TripletexClient, steps: list, task: dict) -> dict:
    """Execute a multi-step task, passing context between steps."""
    context = {}
    results = []

    for i, step in enumerate(steps):
        action = step.get("action", "other")
        step_fields = step.get("fields", {})

        # Inject context from previous steps (e.g., customer_id for invoice)
        step_fields.update({k: v for k, v in context.items() if k not in step_fields})

        handler = HANDLERS.get(action)
        if handler:
            result = await handler(client, step_fields, task)
            results.append(result)

            # Extract IDs for subsequent steps
            if "id" in result:
                entity = action.replace("create_", "").replace("update_", "")
                context[f"{entity}_id"] = result["id"]
                context[f"{entity}"] = result
        else:
            log.warning(f"Step {i}: Unknown action {action}")

    return {"steps": results}


# --- Individual task handlers ---

async def handle_create_employee(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {
        "firstName": fields.get("firstName", ""),
        "lastName": fields.get("lastName", ""),
    }
    if fields.get("email"):
        data["email"] = fields["email"]
    if fields.get("phoneNumberMobile"):
        data["phoneNumberMobile"] = fields["phoneNumberMobile"]
    if fields.get("dateOfBirth"):
        data["dateOfBirth"] = fields["dateOfBirth"]
    if fields.get("startDate"):
        data["employments"] = [{"startDate": fields["startDate"]}]
    if fields.get("employments"):
        data["employments"] = fields["employments"]

    result = await client.create_employee(data)
    log.info(f"Created employee: {result}")
    return result.get("value", result)


async def handle_update_employee(client: TripletexClient, fields: dict, task: dict) -> dict:
    # Find the employee first
    employees = await client.get_employees()
    employee = None

    search_name = fields.get("firstName", "") or fields.get("name", "")
    search_last = fields.get("lastName", "")

    for emp in employees:
        if search_name and search_name.lower() in emp.get("firstName", "").lower():
            if not search_last or search_last.lower() in emp.get("lastName", "").lower():
                employee = emp
                break

    if not employee:
        log.error(f"Employee not found: {fields}")
        return {"error": "Employee not found"}

    # Merge updates
    for key, value in fields.items():
        if key not in ("firstName_search", "lastName_search", "name"):
            employee[key] = value

    result = await client.update_employee(employee["id"], employee)
    return result.get("value", result)


async def handle_create_customer(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {"name": fields.get("name", "")}

    if fields.get("email"):
        data["email"] = fields["email"]
    if fields.get("phoneNumber"):
        data["phoneNumber"] = fields["phoneNumber"]

    # Address
    address = {}
    if fields.get("addressLine1"):
        address["addressLine1"] = fields["addressLine1"]
    if fields.get("postalCode"):
        address["postalCode"] = fields["postalCode"]
    if fields.get("city"):
        address["city"] = fields["city"]
    if fields.get("postalAddress"):
        address = fields["postalAddress"]
    if address:
        data["postalAddress"] = address

    if fields.get("organizationNumber"):
        data["organizationNumber"] = fields["organizationNumber"]

    result = await client.create_customer(data)
    return result.get("value", result)


async def handle_create_product(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {"name": fields.get("name", "")}

    if fields.get("number"):
        data["number"] = fields["number"]
    if fields.get("priceExcludingVat") is not None:
        data["priceExcludingVat"] = fields["priceExcludingVat"]
    if fields.get("priceIncludingVat") is not None:
        data["priceIncludingVat"] = fields["priceIncludingVat"]
    if fields.get("vatType"):
        data["vatType"] = fields["vatType"]

    result = await client.create_product(data)
    return result.get("value", result)


async def handle_create_invoice(client: TripletexClient, fields: dict, task: dict) -> dict:
    # May need to find/create customer first
    customer_id = fields.get("customer_id") or fields.get("customerId")
    if not customer_id and fields.get("customerName"):
        customers = await client.get_customers()
        for c in customers:
            if fields["customerName"].lower() in c.get("name", "").lower():
                customer_id = c["id"]
                break

    # Build order with order lines
    order_lines = []
    for line in fields.get("orderLines", fields.get("lines", [])):
        ol = {"count": line.get("count", 1)}
        if line.get("product_id") or line.get("productId"):
            ol["product"] = {"id": line.get("product_id") or line.get("productId")}
        if line.get("description"):
            ol["description"] = line["description"]
        if line.get("unitPriceExcludingVat") is not None:
            ol["unitPriceExcludingVatCurrency"] = line["unitPriceExcludingVat"]
        order_lines.append(ol)

    order_data = {
        "customer": {"id": customer_id},
        "orderDate": fields.get("invoiceDate", fields.get("orderDate", "")),
        "deliveryDate": fields.get("dueDate", fields.get("deliveryDate", "")),
        "orderLines": order_lines,
    }

    order_result = await client.create_order(order_data)
    order = order_result.get("value", order_result)
    order_id = order.get("id")

    # Invoice the order
    if order_id:
        invoice_data = {
            "invoiceDate": fields.get("invoiceDate", ""),
            "orderId": order_id,
        }
        if fields.get("dueDate"):
            invoice_data["invoiceDueDate"] = fields["dueDate"]

        result = await client.post(f"/order/{order_id}/:invoice", json=invoice_data)
        return result.get("value", result)

    return order


async def handle_register_payment(client: TripletexClient, fields: dict, task: dict) -> dict:
    invoice_id = fields.get("invoice_id") or fields.get("invoiceId")

    if not invoice_id:
        # Try to find the invoice
        invoices = await client.get_invoices()
        if invoices:
            invoice_id = invoices[-1]["id"]  # Most recent

    payment_data = {
        "amount": fields.get("amount", 0),
        "paymentDate": fields.get("paymentDate", ""),
    }
    if fields.get("paymentTypeId"):
        payment_data["paymentType"] = {"id": fields["paymentTypeId"]}

    result = await client.post(f"/invoice/{invoice_id}/:payment", json=payment_data)
    return result.get("value", result)


async def handle_create_project(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {
        "name": fields.get("name", ""),
    }
    if fields.get("number"):
        data["number"] = fields["number"]
    if fields.get("projectManager") or fields.get("projectManagerId"):
        data["projectManager"] = {"id": fields.get("projectManagerId") or fields.get("projectManager", {}).get("id")}
    if fields.get("customer_id") or fields.get("customerId"):
        data["customer"] = {"id": fields.get("customer_id") or fields.get("customerId")}
    if fields.get("startDate"):
        data["startDate"] = fields["startDate"]
    if fields.get("endDate"):
        data["endDate"] = fields["endDate"]

    result = await client.create_project(data)
    return result.get("value", result)


async def handle_create_department(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {"name": fields.get("name", "")}
    if fields.get("departmentNumber"):
        data["departmentNumber"] = fields["departmentNumber"]

    result = await client.create_department(data)
    return result.get("value", result)


async def handle_create_travel_expense(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {}
    if fields.get("employee_id") or fields.get("employeeId"):
        data["employee"] = {"id": fields.get("employee_id") or fields.get("employeeId")}
    if fields.get("project_id") or fields.get("projectId"):
        data["project"] = {"id": fields.get("project_id") or fields.get("projectId")}
    if fields.get("department_id") or fields.get("departmentId"):
        data["department"] = {"id": fields.get("department_id") or fields.get("departmentId")}
    if fields.get("title"):
        data["title"] = fields["title"]
    if fields.get("travelDetails"):
        data["travelDetails"] = fields["travelDetails"]

    # Copy remaining fields
    for key in ("departureDate", "returnDate", "description"):
        if fields.get(key):
            data[key] = fields[key]

    result = await client.create_travel_expense(data)
    return result.get("value", result)


async def handle_delete_travel_expense(client: TripletexClient, fields: dict, task: dict) -> dict:
    expense_id = fields.get("expense_id") or fields.get("expenseId") or fields.get("id")

    if not expense_id:
        # Find and delete
        expenses = await client.get_travel_expenses()
        if expenses:
            expense_id = expenses[-1]["id"]

    if expense_id:
        await client.delete_travel_expense(expense_id)
        return {"deleted": expense_id}
    return {"error": "No travel expense found"}


async def handle_create_credit_note(client: TripletexClient, fields: dict, task: dict) -> dict:
    invoice_id = fields.get("invoice_id") or fields.get("invoiceId")

    if not invoice_id:
        invoices = await client.get_invoices()
        if invoices:
            invoice_id = invoices[-1]["id"]

    if invoice_id:
        result = await client.post(f"/invoice/{invoice_id}/:createCreditNote", json={})
        return result.get("value", result)
    return {"error": "No invoice found"}


async def handle_delete_invoice(client: TripletexClient, fields: dict, task: dict) -> dict:
    invoice_id = fields.get("invoice_id") or fields.get("invoiceId") or fields.get("id")
    if invoice_id:
        await client.delete(f"/invoice/{invoice_id}")
        return {"deleted": invoice_id}
    return {"error": "No invoice ID provided"}


async def handle_create_order(client: TripletexClient, fields: dict, task: dict) -> dict:
    data = {}
    if fields.get("customer_id") or fields.get("customerId"):
        data["customer"] = {"id": fields.get("customer_id") or fields.get("customerId")}
    if fields.get("orderDate"):
        data["orderDate"] = fields["orderDate"]
    if fields.get("deliveryDate"):
        data["deliveryDate"] = fields["deliveryDate"]
    if fields.get("orderLines"):
        data["orderLines"] = fields["orderLines"]

    result = await client.create_order(data)
    return result.get("value", result)


async def handle_enable_module(client: TripletexClient, fields: dict, task: dict) -> dict:
    # Modules are enabled via company settings
    module_name = fields.get("module", fields.get("name", ""))
    log.info(f"Attempting to enable module: {module_name}")

    # Try to enable via the modules endpoint
    try:
        result = await client.put("/company/modules", json={module_name: True})
        return result.get("value", result)
    except Exception as e:
        log.error(f"Failed to enable module: {e}")
        return {"error": str(e)}


async def handle_other(client: TripletexClient, fields: dict, task: dict) -> dict:
    """Fallback handler for unrecognized task types."""
    log.warning(f"Unhandled task type, fields: {fields}")
    return {"warning": "Unrecognized task, attempted best effort"}


# Handler dispatch table
HANDLERS = {
    "create_employee": handle_create_employee,
    "update_employee": handle_update_employee,
    "create_customer": handle_create_customer,
    "update_customer": handle_create_customer,  # Same as create for now
    "create_product": handle_create_product,
    "create_invoice": handle_create_invoice,
    "register_payment": handle_register_payment,
    "create_credit_note": handle_create_credit_note,
    "create_project": handle_create_project,
    "create_department": handle_create_department,
    "create_travel_expense": handle_create_travel_expense,
    "delete_travel_expense": handle_delete_travel_expense,
    "create_order": handle_create_order,
    "enable_module": handle_enable_module,
    "delete_invoice": handle_delete_invoice,
    "other": handle_other,
}

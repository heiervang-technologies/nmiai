"""
Typed action layer. Each action compiles to exact HTTP calls.
The LLM provides structured args; the code handles the contract.
"""

import logging
from datetime import date
from tripletex_client import TripletexClient

log = logging.getLogger(__name__)
TODAY = date.today().isoformat()


async def action_discover_sandbox(client: TripletexClient, args: dict) -> dict:
    """Discover sandbox state: departments, employees, customers, invoices, payment types."""
    result = {}
    try:
        deps = await client.get("/department", params={"count": 10})
        result["departments"] = deps.get("values", [])
    except Exception as e:
        result["departments_error"] = str(e)[:200]

    try:
        emps = await client.get("/employee", params={"count": 10})
        result["employees"] = emps.get("values", [])
    except Exception as e:
        result["employees_error"] = str(e)[:200]

    try:
        custs = await client.get("/customer", params={"count": 10})
        result["customers"] = custs.get("values", [])
    except Exception as e:
        result["customers_error"] = str(e)[:200]

    try:
        invs = await client.get("/invoice", params={"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "count": 10})
        result["invoices"] = invs.get("values", [])
    except Exception as e:
        result["invoices_error"] = str(e)[:200]

    try:
        pts = await client.get("/invoice/paymentType")
        result["payment_types"] = pts.get("values", [])
    except Exception as e:
        result["payment_types_error"] = str(e)[:200]

    try:
        orders = await client.get("/order", params={"orderDateFrom": "2020-01-01", "orderDateTo": "2030-12-31", "count": 10})
        result["orders"] = orders.get("values", [])
    except Exception as e:
        result["orders_error"] = str(e)[:200]

    return result


async def action_create_employee(client: TripletexClient, args: dict) -> dict:
    """Create an employee. Checks for existing by email first. Handles department lookup."""
    # Check if employee already exists (avoid duplicates)
    if args.get("email"):
        try:
            existing = await client.get("/employee", params={"email": args["email"], "count": 1})
            if existing.get("values"):
                log.info(f"Employee with email {args['email']} already exists, returning existing")
                return {"value": existing["values"][0]}
        except Exception:
            pass

    # Get department ID
    deps = await client.get("/department", params={"count": 1})
    dep_id = deps["values"][0]["id"] if deps.get("values") else None

    body = {
        "firstName": args["firstName"],
        "lastName": args["lastName"],
        "userType": args.get("userType", "STANDARD"),
    }
    if dep_id:
        body["department"] = {"id": dep_id}
    for field in ["email", "dateOfBirth", "phoneNumberMobile"]:
        if args.get(field):
            body[field] = args[field]
    if args.get("address"):
        body["address"] = args["address"]

    return await client.post("/employee", json=body)


async def action_create_customer(client: TripletexClient, args: dict) -> dict:
    """Create a customer."""
    body = {"name": args["name"]}
    for field in ["email", "phoneNumber", "organizationNumber", "isPrivateIndividual"]:
        if args.get(field) is not None:
            body[field] = args[field]
    if args.get("postalAddress"):
        body["postalAddress"] = args["postalAddress"]
    return await client.post("/customer", json=body)


async def action_create_supplier(client: TripletexClient, args: dict) -> dict:
    """Create a supplier."""
    body = {"name": args["name"]}
    for field in ["email", "phoneNumber", "organizationNumber"]:
        if args.get(field):
            body[field] = args[field]
    return await client.post("/supplier", json=body)


async def action_create_product(client: TripletexClient, args: dict) -> dict:
    """Create a product. Auto-sets vatType if not provided."""
    body = {"name": args["name"]}
    if args.get("number"):
        body["number"] = str(args["number"])
    if args.get("priceExcludingVat") is not None:
        body["priceExcludingVatCurrency"] = float(args["priceExcludingVat"])
    if args.get("priceIncludingVat") is not None:
        body["priceIncludingVatCurrency"] = float(args["priceIncludingVat"])
    # Default vatType to 25% standard
    vat_id = args.get("vatTypeId", 3)
    body["vatType"] = {"id": int(vat_id)}
    return await client.post("/product", json=body)


async def action_create_department(client: TripletexClient, args: dict) -> dict:
    """Create a department."""
    body = {
        "name": args["name"],
        "departmentNumber": str(args.get("departmentNumber", args.get("number", ""))),
    }
    return await client.post("/department", json=body)


async def action_setup_bank_account(client: TripletexClient, args: dict) -> dict:
    """Register a bank account on the company. Required before invoice creation."""
    bank_num = args.get("bankAccountNumber", "12345678903")

    # Approach 1: Find company via withLoginAccess, GET full object, PUT back with bank number
    try:
        companies = await client.get("/company/%3EwithLoginAccess")
        if companies.get("values"):
            company_id = companies["values"][0]["id"]
            company_data = await client.get(f"/company/{company_id}")
            company_obj = company_data.get("value", company_data)
            company_obj["bankAccountNumber"] = bank_num
            result = await client.put(f"/company/{company_id}", json=company_obj)
            log.info(f"Bank account set via withLoginAccess company {company_id}")
            return result
    except Exception as e1:
        log.warning(f"Bank account via withLoginAccess failed: {e1}")

    # Approach 2: Try PUT /company with just the bank field
    for company_id in [0, 1]:
        try:
            # GET first to get version
            company_data = await client.get(f"/company/{company_id}")
            company_obj = company_data.get("value", company_data)
            if company_obj and company_obj.get("id"):
                company_obj["bankAccountNumber"] = bank_num
                result = await client.put(f"/company/{company_id}", json=company_obj)
                log.info(f"Bank account set via company/{company_id}")
                return result
        except Exception as e:
            log.warning(f"Bank account via company/{company_id} failed: {e}")

    # Approach 3: Update the bank ledger account
    try:
        accounts = await client.get("/ledger/account", params={"number": "1920", "count": 1})
        if accounts.get("values"):
            acc = accounts["values"][0]
            acc["isBankAccount"] = True
            acc["bankAccountNumber"] = bank_num
            return await client.put(f"/ledger/account/{acc['id']}", json=acc)
    except Exception as e3:
        log.warning(f"Bank account via ledger account failed: {e3}")

    return {"warning": "Could not set bank account, invoice creation may fail"}


async def action_create_invoice(client: TripletexClient, args: dict) -> dict:
    """Create an invoice. Handles bank account setup and order line formatting."""
    # Ensure bank account is set up
    try:
        await action_setup_bank_account(client, {})
    except Exception as e:
        log.warning(f"Bank account setup failed (may already exist): {e}")

    # Find or create customer
    customer_id = args.get("customerId")
    if not customer_id and args.get("customerName"):
        custs = await client.get("/customer", params={"count": 100})
        for c in custs.get("values", []):
            if args["customerName"].lower() in c.get("name", "").lower():
                customer_id = c["id"]
                break
        if not customer_id:
            # Create customer
            cust_body = {"name": args["customerName"]}
            if args.get("customerOrgNumber"):
                cust_body["organizationNumber"] = args["customerOrgNumber"]
            cust_result = await client.post("/customer", json=cust_body)
            customer_id = cust_result.get("value", {}).get("id")

    if not customer_id:
        return {"error": "Could not find or create customer"}

    # Build order lines
    order_lines = []
    for line in args.get("orderLines", []):
        ol = {
            "description": line.get("description", ""),
            "count": float(line.get("count", 1)),
            "unitPriceExcludingVatCurrency": float(line.get("unitPrice", line.get("unitPriceExcludingVat", 0))),
            "vatType": {"id": int(line.get("vatTypeId", 3))},
        }
        if line.get("productNumber"):
            ol["product"] = {"number": str(line["productNumber"])}
        order_lines.append(ol)

    invoice_date = args.get("invoiceDate", TODAY)
    due_date = args.get("invoiceDueDate", args.get("dueDate", invoice_date))

    body = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "orders": [{
            "customer": {"id": customer_id},
            "orderDate": invoice_date,
            "deliveryDate": invoice_date,
            "orderLines": order_lines,
        }],
    }

    return await client.post("/invoice", json=body)


async def action_create_order(client: TripletexClient, args: dict) -> dict:
    """Create an order."""
    customer_id = args.get("customerId")
    if not customer_id and args.get("customerName"):
        custs = await client.get("/customer", params={"count": 100})
        for c in custs.get("values", []):
            if args["customerName"].lower() in c.get("name", "").lower():
                customer_id = c["id"]
                break

    order_lines = []
    for line in args.get("orderLines", []):
        ol = {
            "description": line.get("description", ""),
            "count": float(line.get("count", 1)),
            "unitPriceExcludingVatCurrency": float(line.get("unitPrice", line.get("unitPriceExcludingVat", 0))),
            "vatType": {"id": int(line.get("vatTypeId", 3))},
        }
        order_lines.append(ol)

    order_date = args.get("orderDate", TODAY)
    body = {
        "customer": {"id": customer_id},
        "orderDate": order_date,
        "deliveryDate": args.get("deliveryDate", order_date),
        "orderLines": order_lines,
    }
    return await client.post("/order", json=body)


async def action_register_payment(client: TripletexClient, args: dict) -> dict:
    """Register payment on an invoice. Finds invoice and payment type automatically."""
    invoice_id = args.get("invoiceId")

    if not invoice_id:
        # Search for the invoice
        invoices = await client.get("/invoice", params={
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 50,
        })
        for inv in invoices.get("values", []):
            invoice_id = inv["id"]
            break  # Take the first/most recent

    if not invoice_id:
        return {"error": "No invoice found"}

    # Get payment types
    pt_resp = await client.get("/invoice/paymentType")
    payment_types = pt_resp.get("values", [])
    payment_type_id = args.get("paymentTypeId")
    if not payment_type_id and payment_types:
        # Default to first available (usually "Kontant" or "Betalt til bank")
        payment_type_id = payment_types[0]["id"]

    amount = float(args.get("amount", args.get("paidAmount", 0)))
    payment_date = args.get("paymentDate", TODAY)

    return await client.put(
        f"/invoice/{invoice_id}/:payment",
        params={
            "paymentDate": payment_date,
            "paymentTypeId": int(payment_type_id),
            "paidAmount": amount,
        }
    )


async def action_create_credit_note(client: TripletexClient, args: dict) -> dict:
    """Create a credit note for an invoice."""
    invoice_id = args.get("invoiceId")

    if not invoice_id:
        invoices = await client.get("/invoice", params={
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 50,
        })
        for inv in invoices.get("values", []):
            if not inv.get("isCredited"):
                invoice_id = inv["id"]
                break

    if not invoice_id:
        return {"error": "No invoice found for credit note"}

    credit_date = args.get("date", TODAY)
    params = {"date": credit_date}
    if args.get("comment"):
        params["comment"] = args["comment"]

    return await client.put(f"/invoice/{invoice_id}/:createCreditNote", params=params)


async def action_create_voucher(client: TripletexClient, args: dict) -> dict:
    """Create a journal entry / voucher. Handles account lookup and posting format."""
    # Look up accounts if needed
    account_cache = {}
    accounts_resp = await client.get("/ledger/account", params={"count": 1000})
    for acc in accounts_resp.get("values", []):
        account_cache[acc["number"]] = acc["id"]

    postings = []
    for i, p in enumerate(args.get("postings", []), start=1):
        account_number = p.get("accountNumber")
        account_id = p.get("accountId")

        if account_number and not account_id:
            account_id = account_cache.get(int(account_number))

        if not account_id:
            return {"error": f"Account not found: {account_number}"}

        posting = {
            "row": i,
            "account": {"id": account_id},
            "amountGross": float(p.get("amountGross", p.get("amount", 0))),
        }
        if p.get("vatTypeId"):
            posting["vatType"] = {"id": int(p["vatTypeId"])}
        if p.get("description"):
            posting["description"] = p["description"]
        postings.append(posting)

    body = {
        "date": args.get("date", TODAY),
        "description": args.get("description", ""),
        "postings": postings,
    }

    return await client.post("/ledger/voucher", json=body)


async def action_activate_module(client: TripletexClient, args: dict) -> dict:
    """Activate a Tripletex module. Returns success even if already active or forbidden."""
    module_name = args.get("moduleName", args.get("name", ""))
    try:
        return await client.post("/company/salesmodules", json={"name": module_name})
    except Exception as e:
        # 403 = forbidden (already active or not available), not a real failure
        log.warning(f"Module activation for {module_name} failed (non-fatal): {e}")
        return {"warning": f"Module {module_name} activation failed, may already be active"}


async def action_create_project(client: TripletexClient, args: dict) -> dict:
    """Create a project. Activates module if needed."""
    # Try to activate project module (non-fatal if fails)
    await action_activate_module(client, {"name": "PROJECT"})

    # Find project manager (first employee)
    manager_id = args.get("projectManagerId")
    if not manager_id:
        emps = await client.get("/employee", params={"count": 10})
        for emp in emps.get("values", []):
            # Prefer matching by email if provided
            if args.get("projectManagerEmail") and emp.get("email") == args["projectManagerEmail"]:
                manager_id = emp["id"]
                break
            elif args.get("projectManagerName") and args["projectManagerName"].lower() in emp.get("displayName", "").lower():
                manager_id = emp["id"]
                break
        if not manager_id and emps.get("values"):
            manager_id = emps["values"][0]["id"]

    body = {
        "name": args["name"],
        "projectManager": {"id": manager_id},
        "isInternal": args.get("isInternal", False),
        "startDate": args.get("startDate", TODAY),
    }
    if args.get("number"):
        body["number"] = str(args["number"])
    if args.get("customerId"):
        body["customer"] = {"id": args["customerId"]}
    if args.get("endDate"):
        body["endDate"] = args["endDate"]
    if args.get("fixedPrice") is not None:
        body["isFixedPrice"] = True
        body["fixedprice"] = float(args["fixedPrice"])

    return await client.post("/project", json=body)


async def action_create_accounting_dimension(client: TripletexClient, args: dict) -> dict:
    """Create a custom accounting dimension with values, optionally post a voucher linked to it."""
    # Step 1: Create the dimension name
    dim_body = {
        "dimensionName": args["dimensionName"],
        "active": True,
    }
    if args.get("description"):
        dim_body["description"] = args["description"]

    dim_result = await client.post("/ledger/accountingDimensionName", json=dim_body)
    dim = dim_result.get("value", dim_result)
    dim_index = dim.get("dimensionIndex")

    # Step 2: Create dimension values
    created_values = []
    for val in args.get("values", []):
        val_body = {
            "displayName": val if isinstance(val, str) else val.get("name", val.get("displayName", "")),
            "dimensionIndex": dim_index,
            "active": True,
        }
        val_result = await client.post("/ledger/accountingDimensionValue", json=val_body)
        created_values.append(val_result.get("value", val_result))

    result = {
        "dimension": dim,
        "values": created_values,
    }

    # Step 3: If voucher posting requested, create it with the dimension
    if args.get("voucherPostings"):
        account_cache = {}
        accounts_resp = await client.get("/ledger/account", params={"count": 1000})
        for acc in accounts_resp.get("values", []):
            account_cache[acc["number"]] = acc["id"]

        postings = []
        for i, p in enumerate(args["voucherPostings"], start=1):
            account_number = p.get("accountNumber")
            account_id = account_cache.get(int(account_number)) if account_number else p.get("accountId")

            posting = {
                "row": i,
                "account": {"id": account_id},
                "amountGross": float(p.get("amountGross", p.get("amount", 0))),
            }
            if p.get("vatTypeId"):
                posting["vatType"] = {"id": int(p["vatTypeId"])}
            # Link to dimension value
            if p.get("dimensionValueId"):
                posting[f"customDimension{dim_index}"] = {"id": p["dimensionValueId"]}
            elif created_values and p.get("dimensionValueName"):
                for cv in created_values:
                    if cv.get("displayName") == p["dimensionValueName"]:
                        posting[f"customDimension{dim_index}"] = {"id": cv["id"]}
                        break
            postings.append(posting)

        voucher_body = {
            "date": args.get("voucherDate", TODAY),
            "description": args.get("voucherDescription", f"Voucher with dimension {args['dimensionName']}"),
            "postings": postings,
        }
        voucher_result = await client.post("/ledger/voucher", json=voucher_body)
        result["voucher"] = voucher_result.get("value", voucher_result)

    return result


async def action_update_employee(client: TripletexClient, args: dict) -> dict:
    """Update an existing employee. Finds by name or ID, then updates fields."""
    employee_id = args.get("employeeId")

    if not employee_id:
        # Search by name
        params = {"count": 100}
        if args.get("firstName"):
            params["firstName"] = args["firstName"]
        if args.get("lastName"):
            params["lastName"] = args["lastName"]
        emps = await client.get("/employee", params=params)
        for emp in emps.get("values", []):
            employee_id = emp["id"]
            break

    if not employee_id:
        return {"error": "Employee not found"}

    # Get full employee object
    emp_data = await client.get(f"/employee/{employee_id}")
    emp = emp_data.get("value", emp_data)

    # Apply updates
    for field in ["email", "phoneNumberMobile", "dateOfBirth", "userType"]:
        if args.get(field) is not None:
            emp[field] = args[field]
    if args.get("address"):
        emp["address"] = args["address"]

    return await client.put(f"/employee/{employee_id}", json=emp)


async def action_register_timesheet(client: TripletexClient, args: dict) -> dict:
    """Register hours on a timesheet for an employee on a project activity."""
    # Activate project + timesheet modules
    for module in ["PROJECT", "TIME_TRACKING"]:
        try:
            await client.post("/company/salesmodules", json={"name": module})
        except Exception:
            pass

    # Find or use provided IDs
    employee_id = args.get("employeeId")
    if not employee_id:
        emps = await client.get("/employee", params={"count": 10})
        for emp in emps.get("values", []):
            if args.get("employeeEmail") and emp.get("email") == args["employeeEmail"]:
                employee_id = emp["id"]
                break
            elif args.get("employeeName") and args["employeeName"].lower() in emp.get("displayName", "").lower():
                employee_id = emp["id"]
                break

    project_id = args.get("projectId")
    if not project_id and args.get("projectName"):
        projects = await client.get("/project", params={"count": 100})
        for proj in projects.get("values", []):
            if args["projectName"].lower() in proj.get("name", "").lower():
                project_id = proj["id"]
                break

    # Find or create activity (use /activity, NOT /project/{id}/activity)
    activity_id = args.get("activityId")
    if not activity_id and args.get("activityName"):
        activities = await client.get("/activity", params={"count": 100})
        for act in activities.get("values", []):
            if args["activityName"].lower() in act.get("name", "").lower():
                activity_id = act["id"]
                break
        # If not found, create it
        if not activity_id:
            act_body = {"name": args["activityName"]}
            if project_id:
                act_body["project"] = {"id": project_id}
            act_result = await client.post("/activity", json=act_body)
            activity_id = act_result.get("value", {}).get("id")

    # Register hours via /timesheet/entry
    entry_body = {
        "employee": {"id": employee_id},
        "date": args.get("date", TODAY),
        "hours": float(args.get("hours", 0)),
    }
    if activity_id:
        entry_body["activity"] = {"id": activity_id}
    if project_id:
        entry_body["project"] = {"id": project_id}
    if args.get("comment"):
        entry_body["comment"] = args["comment"]

    return await client.post("/timesheet/entry", json=entry_body)


# Generic fallback for unknown actions
async def action_generic_api_call(client: TripletexClient, args: dict) -> dict:
    """Make a generic API call. Only used as fallback."""
    method = args.get("method", "GET").upper()
    path = args.get("path", "")
    params = args.get("params")
    body = args.get("body")

    if method == "GET":
        return await client.get(path, params=params)
    elif method == "POST":
        return await client.post(path, json=body)
    elif method == "PUT":
        return await client.put(path, json=body, params=params)
    elif method == "DELETE":
        return await client.delete(path)
    return {"error": f"Unknown method: {method}"}


# Action registry
ACTIONS = {
    "discover_sandbox": action_discover_sandbox,
    "create_employee": action_create_employee,
    "create_customer": action_create_customer,
    "create_supplier": action_create_supplier,
    "create_product": action_create_product,
    "create_department": action_create_department,
    "create_accounting_dimension": action_create_accounting_dimension,
    "update_employee": action_update_employee,
    "register_timesheet": action_register_timesheet,
    "create_invoice": action_create_invoice,
    "create_order": action_create_order,
    "register_payment": action_register_payment,
    "create_credit_note": action_create_credit_note,
    "create_voucher": action_create_voucher,
    "create_project": action_create_project,
    "activate_module": action_activate_module,
    "setup_bank_account": action_setup_bank_account,
    "generic_api_call": action_generic_api_call,
}

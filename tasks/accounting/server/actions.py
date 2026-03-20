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

    # Approach 1: Find company via >withLoginAccess (use unencoded >)
    try:
        companies = await client.get("/company", params={"isMyCompany": "true", "count": 1})
        if not companies.get("values"):
            companies = await client.get("/company/>withLoginAccess")
        if companies.get("values"):
            company_id = companies["values"][0]["id"]
            company_data = await client.get(f"/company/{company_id}")
            company_obj = company_data.get("value", company_data)
            company_obj["bankAccountNumber"] = bank_num
            result = await client.put(f"/company/{company_id}", json=company_obj)
            log.info(f"Bank account set via company {company_id}")
            return result
    except Exception as e1:
        log.warning(f"Bank account via company lookup failed: {e1}")

    # Approach 2: Find existing 1920 bank account in ledger and ensure it has bank number
    try:
        accounts = await client.get("/ledger/account", params={"count": 1000})
        for acc in accounts.get("values", []):
            if acc.get("number") == 1920:
                if not acc.get("bankAccountNumber"):
                    acc["isBankAccount"] = True
                    acc["bankAccountNumber"] = bank_num
                    result = await client.put(f"/ledger/account/{acc['id']}", json=acc)
                    log.info(f"Bank account set via ledger account 1920")
                    return result
                else:
                    log.info("Bank account 1920 already has bank number set")
                    return {"status": "already_set"}
    except Exception as e2:
        log.warning(f"Bank account via ledger account failed: {e2}")

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

        amount = float(p.get("amountGross", p.get("amount", 0)))
        posting = {
            "row": i,
            "account": {"id": account_id},
            "amountGross": amount,
            "amountGrossCurrency": amount,  # Must match amountGross
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
    """Create a project. Activates module if needed. Handles customer and employee lookup."""
    # Try to activate project module (non-fatal if fails)
    await action_activate_module(client, {"name": "PROJECT"})

    # Find or create customer if needed
    customer_id = args.get("customerId")
    if not customer_id and args.get("customerName"):
        custs = await client.get("/customer", params={"count": 100})
        for c in custs.get("values", []):
            if args["customerName"].lower() in c.get("name", "").lower():
                customer_id = c["id"]
                break
        if not customer_id:
            cust_body = {"name": args["customerName"]}
            if args.get("customerOrgNumber"):
                cust_body["organizationNumber"] = args["customerOrgNumber"]
            cust_result = await client.post("/customer", json=cust_body)
            customer_id = cust_result.get("value", {}).get("id")

    # Find project manager — search by email first, then name, then fallback to first employee
    manager_id = args.get("projectManagerId")
    if not manager_id:
        emps = await client.get("/employee", params={"count": 10})
        for emp in emps.get("values", []):
            if args.get("projectManagerEmail") and emp.get("email") == args["projectManagerEmail"]:
                manager_id = emp["id"]
                break
            elif args.get("projectManagerName") and args["projectManagerName"].lower() in emp.get("displayName", "").lower():
                manager_id = emp["id"]
                break
        # If manager not found by email/name, create them
        if not manager_id and args.get("projectManagerEmail"):
            name_parts = args.get("projectManagerName", "Project Manager").split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else "Manager"
            try:
                emp_result = await action_create_employee(client, {
                    "firstName": first_name, "lastName": last_name,
                    "email": args["projectManagerEmail"],
                })
                manager_id = emp_result.get("value", {}).get("id")
            except Exception:
                pass
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
    if customer_id:
        body["customer"] = {"id": customer_id}
    if args.get("endDate"):
        body["endDate"] = args["endDate"]

    result = await client.post("/project", json=body)

    # Set fixed price via PUT after creation (fixedPrice is not valid on POST)
    if args.get("fixedPrice") is not None:
        project_id = result.get("value", {}).get("id")
        if project_id:
            try:
                proj_data = await client.get(f"/project/{project_id}")
                proj_obj = proj_data.get("value", proj_data)
                proj_obj["isFixedPrice"] = True
                proj_obj["fixedprice"] = float(args["fixedPrice"])
                result = await client.put(f"/project/{project_id}", json=proj_obj)
            except Exception as e:
                log.warning(f"Failed to set fixed price on project {project_id}: {e}")

    return result


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

        # Map dimensionIndex to the correct field name
        # Tripletex uses: freeAccountingDimension1, freeAccountingDimension2, freeAccountingDimension3
        dim_field = f"freeAccountingDimension{dim_index}" if dim_index else "freeAccountingDimension1"

        postings = []
        for i, p in enumerate(args["voucherPostings"], start=1):
            account_number = p.get("accountNumber")
            account_id = account_cache.get(int(account_number)) if account_number else p.get("accountId")

            amount = float(p.get("amountGross", p.get("amount", 0)))
            posting = {
                "row": i,
                "account": {"id": account_id},
                "amountGross": amount,
                "amountGrossCurrency": amount,  # Must match amountGross
            }
            if p.get("vatTypeId"):
                posting["vatType"] = {"id": int(p["vatTypeId"])}
            # Link to dimension value using correct field name
            if p.get("dimensionValueId"):
                posting[dim_field] = {"id": p["dimensionValueId"]}
            elif created_values and p.get("dimensionValueName"):
                for cv in created_values:
                    if cv.get("displayName") == p["dimensionValueName"]:
                        posting[dim_field] = {"id": cv["id"]}
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
    # Activate project + timesheet modules (non-fatal)
    for module in ["PROJECT", "TIME_TRACKING"]:
        await action_activate_module(client, {"name": module})

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

    # Find or create activity
    # Use /activity for listing, /project/projectActivity for creating
    activity_id = args.get("activityId")
    if not activity_id and args.get("activityName"):
        # Try listing existing activities
        try:
            activities = await client.get("/activity", params={"count": 100})
            for act in activities.get("values", []):
                if args["activityName"].lower() in act.get("name", "").lower():
                    activity_id = act["id"]
                    break
        except Exception:
            pass

        # If not found, create via /project/projectActivity
        if not activity_id and project_id:
            try:
                act_body = {
                    "project": {"id": project_id},
                    "activity": {"name": args["activityName"]},
                    "startDate": args.get("date", TODAY),
                }
                act_result = await client.post("/project/projectActivity", json=act_body)
                act_val = act_result.get("value", {})
                activity_id = act_val.get("activity", {}).get("id") or act_val.get("id")
            except Exception as e:
                log.warning(f"Failed to create project activity: {e}")
                # Fallback: try /activity directly
                try:
                    act_result = await client.post("/activity", json={"name": args["activityName"]})
                    activity_id = act_result.get("value", {}).get("id")
                except Exception:
                    pass

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


async def action_register_supplier_invoice(client: TripletexClient, args: dict) -> dict:
    """Register a supplier invoice (incoming invoice). Creates supplier if needed, then posts the invoice."""
    # Find or create supplier
    supplier_id = args.get("supplierId")
    if not supplier_id and args.get("supplierName"):
        suppliers = await client.get("/supplier", params={"count": 100})
        for s in suppliers.get("values", []):
            if args["supplierName"].lower() in s.get("name", "").lower():
                supplier_id = s["id"]
                break
        if not supplier_id:
            sup_body = {"name": args["supplierName"]}
            if args.get("supplierOrgNumber"):
                sup_body["organizationNumber"] = args["supplierOrgNumber"]
            sup_result = await client.post("/supplier", json=sup_body)
            supplier_id = sup_result.get("value", {}).get("id")

    # Look up account
    account_id = args.get("accountId")
    if not account_id and args.get("accountNumber"):
        accounts = await client.get("/ledger/account", params={"count": 1000})
        for acc in accounts.get("values", []):
            if acc.get("number") == int(args["accountNumber"]):
                account_id = acc["id"]
                break

    # Create the incoming invoice via voucher
    amount = float(args.get("amountIncludingVat", args.get("amount", 0)))
    invoice_date = args.get("invoiceDate", TODAY)

    # Use POST /incomingInvoice (BETA) with correct schema
    body = {
        "invoiceHeader": {
            "invoiceNumber": args.get("invoiceNumber", ""),
            "invoiceDate": invoice_date,
            "vendorId": supplier_id,
            "invoiceAmount": amount,
        },
        "orderLines": [],
    }
    if args.get("dueDate"):
        body["invoiceHeader"]["dueDate"] = args["dueDate"]
    if args.get("description"):
        body["invoiceHeader"]["description"] = args["description"]

    # Add order line if account is specified
    if account_id:
        body["orderLines"].append({
            "description": args.get("description", "Supplier invoice"),
            "account": {"id": account_id},
            "amountExcludingVat": amount * 0.8,  # Estimate ex-VAT from inc-VAT at 25%
            "amountExcludingVatCurrency": amount * 0.8,
            "vatType": {"id": int(args.get("vatTypeId", 1))},  # id=1 = 25% inngående MVA
        })

    try:
        result = await client.post("/incomingInvoice", json=body)
        return result
    except Exception as e:
        log.warning(f"incomingInvoice failed, trying voucher approach: {e}")

    # Fallback: create a voucher with supplier reference
    # Look up all needed accounts
    account_cache = {}
    accounts = await client.get("/ledger/account", params={"count": 1000})
    for acc in accounts.get("values", []):
        account_cache[acc["number"]] = acc["id"]

    if not account_id and args.get("accountNumber"):
        account_id = account_cache.get(int(args["accountNumber"]))

    credit_account_id = account_cache.get(2400)  # Leverandørgjeld
    vat_account_id = account_cache.get(2710)  # Inngående MVA

    if not account_id:
        # Default to 6300 (office services) if no account specified
        account_id = account_cache.get(6300) or account_cache.get(6590)
    if not credit_account_id:
        return {"error": f"Missing leverandørgjeld account 2400"}

    # Calculate VAT split: 25% MVA means amount_ex_vat = amount / 1.25
    amount_ex_vat = round(amount / 1.25, 2)
    vat_amount = round(amount - amount_ex_vat, 2)

    # Balanced postings: expense debit + VAT debit = supplier credit
    postings = [
        {"row": 1, "account": {"id": account_id}, "amountGross": amount_ex_vat, "amountGrossCurrency": amount_ex_vat},
        {"row": 2, "account": {"id": credit_account_id}, "amountGross": -amount, "amountGrossCurrency": -amount},
    ]
    # Add VAT line if we have the account
    if vat_account_id:
        postings.append(
            {"row": 3, "account": {"id": vat_account_id}, "amountGross": vat_amount, "amountGrossCurrency": vat_amount}
        )

    voucher_body = {
        "date": invoice_date,
        "description": f"Supplier invoice {args.get('invoiceNumber', '')} from {args.get('supplierName', 'supplier')}",
        "postings": postings,
    }
    return await client.post("/ledger/voucher", json=voucher_body)


async def action_create_travel_expense(client: TripletexClient, args: dict) -> dict:
    """Create a travel expense report with per diem."""
    # Find employee
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
        if not employee_id and emps.get("values"):
            employee_id = emps["values"][0]["id"]

    body = {
        "employee": {"id": employee_id},
        "title": args.get("title", "Travel expense"),
        "isChargeable": args.get("isChargeable", False),
        "isFixedInvoicedAmount": args.get("isFixedInvoicedAmount", False),
        "isIncludeAttachedReceiptsWhenReinvoicing": False,
    }

    if args.get("departureDate") or args.get("returnDate"):
        body["travelDetails"] = {
            "departureDate": args.get("departureDate"),
            "returnDate": args.get("returnDate"),
            "purpose": args.get("title", ""),
        }
        if args.get("departure"):
            body["travelDetails"]["departureFrom"] = args["departure"]
        if args.get("destination"):
            body["travelDetails"]["destination"] = args["destination"]

    result = await client.post("/travelExpense", json=body)
    expense = result.get("value", result)
    expense_id = expense.get("id")

    # Add per diem if specified
    if expense_id and args.get("perDiemDays") and args.get("perDiemRate"):
        try:
            per_diem_body = {
                "travelExpense": {"id": expense_id},
                "count": int(args["perDiemDays"]),
                "rate": float(args["perDiemRate"]),
                "overnightAccommodation": args.get("accommodation", "HOTEL"),
                "location": args.get("destination", args.get("title", "Norway")),
                "address": args.get("destination", ""),
            }
            await client.post("/travelExpense/perDiemCompensation", json=per_diem_body)
        except Exception as e:
            log.warning(f"Per diem creation failed: {e}")

    # Deliver the expense
    if expense_id:
        try:
            await client.put(f"/travelExpense/:deliver", params={"id": expense_id})
        except Exception as e:
            log.warning(f"Travel expense delivery failed (may need approval): {e}")

    return result


async def action_process_salary(client: TripletexClient, args: dict) -> dict:
    """Process salary by creating a manual voucher. Salary API requires special setup,
    so we use voucher postings on salary accounts (5000-series)."""
    # Find employee
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

    # Look up accounts
    account_cache = {}
    accounts = await client.get("/ledger/account", params={"count": 1000})
    for acc in accounts.get("values", []):
        account_cache[acc["number"]] = acc["id"]

    base_salary = float(args.get("baseSalary", 0))
    bonus = float(args.get("bonus", 0))
    total_gross = base_salary + bonus

    # Standard Norwegian salary posting:
    # Debit 5000 (salary cost) for gross amount
    # Debit 5000 (bonus) if applicable
    # Credit 1920 (bank) for net pay
    # Credit 2780 (withholding tax) for estimated tax (~30%)
    estimated_tax = round(total_gross * 0.30, 2)
    net_pay = round(total_gross - estimated_tax, 2)

    salary_account = account_cache.get(5000) or account_cache.get(5001)
    bank_account = account_cache.get(1920)
    tax_account = account_cache.get(2780) or account_cache.get(2600)

    if not salary_account or not bank_account:
        return {"error": f"Missing accounts: salary={salary_account}, bank={bank_account}"}

    postings = [
        {"row": 1, "account": {"id": salary_account}, "amountGross": total_gross, "amountGrossCurrency": total_gross,
         "description": f"Lønn {args.get('employeeName', '')}"},
    ]
    row = 2
    if tax_account:
        postings.append(
            {"row": row, "account": {"id": tax_account}, "amountGross": -estimated_tax, "amountGrossCurrency": -estimated_tax,
             "description": "Skattetrekk"}
        )
        row += 1
    postings.append(
        {"row": row, "account": {"id": bank_account}, "amountGross": -net_pay, "amountGrossCurrency": -net_pay,
         "description": "Utbetaling"}
    )

    voucher_body = {
        "date": args.get("date", TODAY),
        "description": f"Lønn {args.get('employeeName', '')} - grunnlønn {base_salary}" + (f" + bonus {bonus}" if bonus else ""),
        "postings": postings,
    }

    return await client.post("/ledger/voucher", json=voucher_body)


# Generic fallback for unknown actions
async def action_generic_api_call(client: TripletexClient, args: dict) -> dict:
    """Make a generic API call. Auto-adds required params for known endpoints."""
    method = args.get("method", "GET").upper()
    path = args.get("path", "")
    params = args.get("params") or {}
    body = args.get("body")

    # Auto-add required date params for endpoints that need them
    if method == "GET":
        if "/invoice" in path and "invoiceDateFrom" not in params:
            params["invoiceDateFrom"] = "2020-01-01"
            params["invoiceDateTo"] = "2030-12-31"
        if "/order" in path and "orderDateFrom" not in params:
            params["orderDateFrom"] = "2020-01-01"
            params["orderDateTo"] = "2030-12-31"
        if "/ledger/voucher" in path and "dateFrom" not in params:
            params["dateFrom"] = "2020-01-01"
            params["dateTo"] = "2030-12-31"
        return await client.get(path, params=params or None)
    elif method == "POST":
        return await client.post(path, json=body)
    elif method == "PUT":
        return await client.put(path, json=body, params=params or None)
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
    "register_supplier_invoice": action_register_supplier_invoice,
    "create_travel_expense": action_create_travel_expense,
    "process_salary": action_process_salary,
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

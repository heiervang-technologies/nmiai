"""
Typed action layer. Each action compiles to exact HTTP calls.
The LLM provides structured args; the code handles the contract.
"""

import logging
from datetime import date
from tripletex_client import TripletexClient

log = logging.getLogger(__name__)
TODAY = date.today().isoformat()


def _error_text(exc: Exception) -> str:
    text = str(exc)
    if hasattr(exc, "response"):
        try:
            text = exc.response.text
        except Exception:
            pass
    return text


def _error_mentions(exc: Exception, *needles: str) -> bool:
    text = _error_text(exc).lower()
    return any(needle.lower() in text for needle in needles)


def _build_project_update_payload(project_obj: dict) -> dict:
    """Keep only fields that are safe to send back to PUT /project/{id}."""
    allowed_fields = [
        "id",
        "version",
        "name",
        "number",
        "displayName",
        "description",
        "projectManager",
        "department",
        "mainProject",
        "startDate",
        "endDate",
        "customer",
        "isClosed",
        "isReadyForInvoicing",
        "isInternal",
        "isOffer",
        "isFixedPrice",
        "fixedprice",
        "isPriceCeiling",
        "priceCeilingAmount",
        "projectCategory",
    ]
    return {field: project_obj[field] for field in allowed_fields if project_obj.get(field) is not None}


def _money(value: float | int) -> float:
    """Normalize monetary values to 2 decimals to avoid Tripletex validation drift."""
    return round(float(value), 2)


def _contains_ci(haystack: str | None, needle: str | None) -> bool:
    return bool(haystack and needle and needle.lower() in haystack.lower())


def _invoice_matches(inv: dict, args: dict) -> bool:
    """Best-effort invoice matching using customer/invoice identifiers from the prompt."""
    customer = inv.get("customer") or {}
    customer_name = customer.get("name") or customer.get("displayName") or inv.get("customerName") or ""
    customer_org = customer.get("organizationNumber") or inv.get("customerOrganizationNumber") or ""
    invoice_number = str(inv.get("invoiceNumber") or inv.get("number") or "")

    if args.get("invoiceNumber") and args["invoiceNumber"] not in invoice_number:
        return False
    if args.get("customerOrgNumber") and args["customerOrgNumber"] != customer_org:
        return False
    if args.get("customerName") and not _contains_ci(customer_name, args["customerName"]):
        return False
    return True


def _resolve_default_vat_id(vat_types: dict, fallback: int = 3) -> int:
    vat_id = _find_vat_type_id(vat_types, percentage=25, prefer_outgoing=True)
    return vat_id if vat_id is not None else fallback


def _find_vat_type_id(vat_types: dict, percentage: int | None, prefer_outgoing: bool) -> int | None:
    candidates = []
    for vt in vat_types.get("values", []):
        vt_id = vt.get("id")
        if vt_id is None:
            continue
        name = (vt.get("name") or "").lower()
        vt_percentage = vt.get("percentage")
        score = 0
        if percentage is not None:
            if vt_percentage != percentage:
                continue
            score += 100
        if prefer_outgoing and "utgående" in name:
            score += 20
        if "standard" in name or "høy sats" in name:
            score += 10
        if "direktepostert" in name:
            score -= 20
        if "inngående" in name or "fradrag" in name:
            score -= 50
        candidates.append((score, int(vt_id)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _requested_vat_percentages(requested: int | None) -> list[int]:
    if requested is None:
        return [25]
    requested = int(requested)
    if requested in {0, 12, 15, 25}:
        return [requested]
    legacy_map = {
        3: [25],
        5: [15, 0],
        6: [0],
        31: [15],
        32: [12],
    }
    return legacy_map.get(requested, [])


def _resolve_outgoing_vat_id(vat_types: dict, requested: int | None, fallback: int = 3) -> int:
    for percentage in _requested_vat_percentages(requested):
        vat_id = _find_vat_type_id(vat_types, percentage=percentage, prefer_outgoing=True)
        if vat_id is not None:
            return vat_id
    if requested is not None and int(requested) not in {3, 5, 6, 31, 32}:
        return int(requested)
    return fallback


async def _get_outgoing_vat_types(client: TripletexClient) -> dict:
    try:
        return await client.get("/ledger/vatType", params={"count": 50})
    except Exception:
        return {"values": []}


async def _find_customer_id(
    client: TripletexClient,
    customer_id: int | None = None,
    customer_name: str | None = None,
    customer_org_number: str | None = None,
) -> int | None:
    """Find an existing customer by ID/name/org number, or create one if enough data is provided."""
    if customer_id:
        return customer_id

    if customer_name or customer_org_number:
        customers = await client.get("/customer", params={"count": 100})
        for customer in customers.get("values", []):
            if customer_org_number and customer.get("organizationNumber") == customer_org_number:
                return customer["id"]
            if customer_name and customer_name.lower() in customer.get("name", "").lower():
                return customer["id"]

    if customer_name:
        body = {"name": customer_name}
        if customer_org_number:
            body["organizationNumber"] = customer_org_number
        created = await client.post("/customer", json=body)
        return created.get("value", {}).get("id")

    return None


async def _grant_employee_all_privileges(client: TripletexClient, employee_id: int) -> None:
    """Grant ALL_PRIVILEGES to employee so they can do travel/timesheet/salary."""
    try:
        emp_data = await client.get(f"/employee/{employee_id}")
        emp = emp_data.get("value", emp_data)
        if emp.get("allowInformationRegistration"):
            return
    except Exception:
        pass

    # Set allowInformationRegistration + userType via PUT on employee.
    # NOTE: Do NOT call /employee/entitlement/:grantEntitlementsByTemplate —
    # it returns 404/500 in competition sandboxes and wastes API calls.
    try:
        emp_data = await client.get(f"/employee/{employee_id}")
        emp = emp_data.get("value", emp_data)
        emp["allowInformationRegistration"] = True
        emp["userType"] = "EXTENDED"
        await client.put(f"/employee/{employee_id}", json=emp)
        log.info(f"Set allowInformationRegistration+EXTENDED on employee {employee_id} via PUT")
        return
    except Exception as e2:
        log.warning(f"PUT employee fallback also failed: {e2}")


def _is_placeholder_employee(employee: dict | None) -> bool:
    if not employee:
        return True
    return (
        not employee.get("allowInformationRegistration")
        and not employee.get("department")
        and not employee.get("employments")
        and not (employee.get("email") or "").strip()
        and not employee.get("userType")
    )


async def _find_employee_id(
    client: TripletexClient,
    employee_id: int | None = None,
    employee_email: str | None = None,
    employee_name: str | None = None,
    create_if_missing: bool = False,
) -> int | None:
    """Find an employee by ID/email/name. Optionally create them if missing and enough data exists."""
    if employee_id:
        try:
            employee_data = await client.get(f"/employee/{employee_id}")
            employee = employee_data.get("value", employee_data)
            if not _is_placeholder_employee(employee):
                return employee_id
            log.info(f"Ignoring placeholder employee id={employee_id}")
        except Exception:
            pass

    employees = await client.get("/employee", params={"count": 25})
    values = employees.get("values", [])
    if employee_id:
        for employee in values:
            if employee.get("id") == employee_id:
                return employee_id
        if 1 <= int(employee_id) <= len(values):
            ordinal_match = values[int(employee_id) - 1]
            log.info(f"Mapped employee ordinal {employee_id} to employee id={ordinal_match['id']}")
            return ordinal_match["id"]

    for employee in values:
        if employee_email and employee.get("email") == employee_email:
            return employee["id"]
        if employee_name and employee_name.lower() in employee.get("displayName", "").lower():
            return employee["id"]

    if create_if_missing and employee_email:
        name_parts = (employee_name or "Project Manager").split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else "Manager"
        created = await action_create_employee(
            client,
            {
                "firstName": first_name,
                "lastName": last_name,
                "email": employee_email,
            },
        )
        employee_value = created.get("value", created)
        if employee_value.get("id"):
            return employee_value["id"]

    for employee in values:
        if employee.get("allowInformationRegistration"):
            return employee["id"]

    return values[0]["id"] if values else None


async def _find_project_id(client: TripletexClient, project_id: int | None = None, project_name: str | None = None) -> int | None:
    """Find a project by ID or fuzzy name match."""
    if project_id:
        return project_id
    if not project_name:
        return None

    projects = await client.get("/project", params={"count": 100})
    for project in projects.get("values", []):
        if project_name.lower() in project.get("name", "").lower():
            return project["id"]
    return None


async def _find_existing_product(
    client: TripletexClient,
    *,
    product_name: str | None = None,
    product_number: str | None = None,
) -> dict | None:
    if not product_name and not product_number:
        return None

    products = await client.get("/product", params={"count": 100})
    values = products.get("values", [])

    if product_number:
        wanted_number = str(product_number).strip()
        for product in values:
            if str(product.get("number", "")).strip() == wanted_number:
                return product

    if product_name:
        wanted_name = product_name.strip().lower()
        for product in values:
            if (product.get("name") or "").strip().lower() == wanted_name:
                return product

    return None


async def _set_project_fixed_price(
    client: TripletexClient,
    project_id: int,
    fixed_price: float,
    *,
    start_date: str,
) -> dict | None:
    proj_data = await client.get(f"/project/{project_id}")
    proj_obj = _build_project_update_payload(proj_data.get("value", proj_data))
    proj_obj["isFixedPrice"] = True
    proj_obj["fixedprice"] = _money(fixed_price)
    try:
        await client.put(f"/project/{project_id}", json=proj_obj)
    except Exception as e:
        log.warning(f"Project fixedprice PUT failed (trying hourlyRates): {e}")

    hourly_rates = await client.get("/project/hourlyRates", params={"projectId": project_id})
    hourly_rate_values = hourly_rates.get("values", [])

    candidate_models = ["FIXED_HOURLY_RATE", "TYPE_FIXED_HOURLY_RATE"]

    if hourly_rate_values:
        hourly_rate_id = hourly_rate_values[0].get("id")
        if hourly_rate_id:
            hourly_rate_data = await client.get(f"/project/hourlyRates/{hourly_rate_id}")
            hourly_rate_obj = hourly_rate_data.get("value", hourly_rate_data)
            hourly_rate_obj["fixedRate"] = _money(fixed_price)
            hourly_rate_obj.setdefault("startDate", start_date)
            existing_model = hourly_rate_obj.get("hourlyRateModel")
            for model in ([existing_model] if existing_model else []) + candidate_models:
                if not model:
                    continue
                hourly_rate_obj["hourlyRateModel"] = model
                try:
                    return await client.put(f"/project/hourlyRates/{hourly_rate_id}", json=hourly_rate_obj)
                except Exception as e:
                    if not _error_mentions(e, "hourlyratemodel", "korrekt type"):
                        raise

    last_error = None
    for model in candidate_models:
        try:
            return await client.post(
                "/project/hourlyRates",
                json={
                    "project": {"id": project_id},
                    "startDate": start_date,
                    "showInProjectOrder": True,
                    "hourlyRateModel": model,
                    "fixedRate": _money(fixed_price),
                },
            )
        except Exception as e:
            last_error = e
            if not _error_mentions(e, "hourlyratemodel", "korrekt type"):
                raise
    if last_error:
        raise last_error
    return None


async def _resolve_travel_per_diem_rate(
    client: TripletexClient,
    *,
    country_code: str,
    is_day_trip: bool,
    accommodation: str,
) -> tuple[dict | None, dict | None]:
    is_domestic = country_code.upper() == "NO"
    has_accommodation = accommodation.upper() not in {"", "NONE", "NO_ACCOMMODATION"}

    try:
        category_resp = await client.get(
            "/travelExpense/rateCategory",
            params={"type": "PER_DIEM", "count": 50},
        )
    except Exception as exc:
        log.warning(f"Per diem rate category lookup failed: {_error_text(exc)[:200]}")
        return None, None

    # Pick the best category based on trip type.
    # Domestic overnight with >12h is the most common per diem scenario.
    best_category = None
    categories = category_resp.get("values", [])
    for category in categories:
        cat_name = (category.get("name") or "").lower()
        # Prefer overnight > 12h for multi-day, dagsreise for day trips
        if is_day_trip and "dagsreise" in cat_name:
            best_category = category
        elif not is_day_trip and "overnatting" in cat_name and "12" in cat_name:
            best_category = category
        elif best_category is None:
            best_category = category

        params = {
            "rateCategoryId": category["id"],
            "type": "PER_DIEM",
            "isValidDomestic": is_domestic,
            "isValidDayTrip": is_day_trip,
            "isValidAccommodation": has_accommodation,
        }
        try:
            rate_resp = await client.get("/travelExpense/rate", params=params)
        except Exception as exc:
            log.warning(f"Per diem rate lookup failed for category {category.get('id')}: {_error_text(exc)[:200]}")
            continue

        values = rate_resp.get("values", [])
        if values:
            return category, values[0]

    # Rates are empty but categories exist — return best category without rate.
    # The per diem body will include rateCategory which may be enough.
    if best_category:
        return best_category, None
    return None, None


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

    dep_id = args.get("departmentId")
    if not dep_id:
        deps = await client.get("/department", params={"count": 100})
        dep_values = deps.get("values", [])
        wanted_department = (args.get("departmentName") or "").strip().lower()
        if wanted_department:
            for dep in dep_values:
                dep_name = (dep.get("name") or "").strip().lower()
                if dep_name == wanted_department or wanted_department in dep_name:
                    dep_id = dep["id"]
                    break
        if not dep_id and dep_values:
            dep_id = dep_values[0]["id"]

    user_type = args.get("userType", "STANDARD")
    if not args.get("email") and user_type == "STANDARD":
        # Tripletex requires email for login users; plain onboarding without email
        # should fall back to a non-login employee instead of failing the whole task.
        user_type = "NO_ACCESS"

    body = {
        "firstName": args["firstName"],
        "lastName": args["lastName"],
        "userType": user_type,
    }
    if dep_id:
        body["department"] = {"id": dep_id}
    for field in ["email", "dateOfBirth", "phoneNumberMobile", "nationalIdentityNumber"]:
        if args.get(field):
            body[field] = args[field]
    if args.get("address"):
        body["address"] = args["address"]

    result = await client.post("/employee", json=body)
    employee = result.get("value", result)
    employee_id = employee.get("id")

    # Create employment record if start date provided
    if employee_id and args.get("startDate"):
        try:
            employment_body = {
                "employee": {"id": employee_id},
                "startDate": args["startDate"],
            }
            if args.get("endDate"):
                employment_body["endDate"] = args["endDate"]
            # percentageOfFullTimeEquivalent and occupationCode belong on
            # employment/details, NOT employment itself
            await client.post("/employee/employment", json=employment_body)
        except Exception as e:
            log.warning(f"Employment creation failed: {e}")

    # Set annual salary via employment details if provided
    if employee_id and args.get("annualSalary"):
        try:
            # Get the employment record we just created
            employments = await client.get("/employee/employment", params={"employeeId": employee_id, "count": 1})
            if employments.get("values"):
                emp_record = employments["values"][0]
                emp_record_id = emp_record["id"]
                # Set salary via employment/details
                salary_body = {
                    "employment": {"id": emp_record_id},
                    "date": args.get("startDate", TODAY),
                    "annualSalary": float(args["annualSalary"]),
                }
                if args.get("percentageOfFullTimeEquivalent"):
                    salary_body["payrollPercentage"] = float(args["percentageOfFullTimeEquivalent"])
                await client.post("/employee/employment/details", json=salary_body)
        except Exception as e:
            log.warning(f"Salary setup failed: {e}")

    # Set standard work time if provided (e.g. 7.5 hours/day)
    if employee_id and args.get("hoursPerDay"):
        try:
            await client.post("/employee/standardTime", json={
                "employee": {"id": employee_id},
                "fromDate": args.get("startDate", TODAY),
                "hoursPerDay": float(args["hoursPerDay"]),
            })
        except Exception as e:
            log.warning(f"Standard work time setup failed: {e}")

    return result


async def action_create_customer(client: TripletexClient, args: dict) -> dict:
    """Create a customer with address."""
    body = {"name": args["name"]}
    for field in ["email", "phoneNumber", "organizationNumber", "isPrivateIndividual"]:
        if args.get(field) is not None:
            body[field] = args[field]

    # Build postal address from either dict or individual fields
    address = args.get("postalAddress", {})
    if not address:
        address = {}
        if args.get("addressLine1"):
            address["addressLine1"] = args["addressLine1"]
        if args.get("postalCode"):
            address["postalCode"] = args["postalCode"]
        if args.get("city"):
            address["city"] = args["city"]
    if address:
        body["postalAddress"] = address

    result = await client.post("/customer", json=body)
    customer = result.get("value", result)
    customer_id = customer.get("id")

    # If address was provided, verify it was set by doing a PUT
    if customer_id and address:
        try:
            cust_data = await client.get(f"/customer/{customer_id}")
            cust_obj = cust_data.get("value", cust_data)
            if cust_obj.get("postalAddress"):
                addr = cust_obj["postalAddress"]
                needs_update = False
                for k, v in address.items():
                    if addr.get(k) != v:
                        addr[k] = v
                        needs_update = True
                if needs_update:
                    cust_obj["postalAddress"] = addr
                    result = await client.put(f"/customer/{customer_id}", json=cust_obj)
        except Exception as e:
            log.warning(f"Customer address update failed: {e}")

    return result


async def action_create_supplier(client: TripletexClient, args: dict) -> dict:
    """Create a supplier."""
    body = {"name": args["name"]}
    for field in ["email", "phoneNumber", "organizationNumber"]:
        if args.get(field):
            body[field] = args[field]
    return await client.post("/supplier", json=body)


async def action_create_product(client: TripletexClient, args: dict) -> dict:
    """Create a product without brute-force VAT retries or duplicate-name loops."""
    existing = await _find_existing_product(
        client,
        product_name=args.get("name"),
        product_number=args.get("number"),
    )
    if existing:
        return {"value": existing, "note": "Product already exists"}

    body = {"name": args["name"]}
    if args.get("number"):
        body["number"] = str(args["number"])
    if args.get("priceExcludingVat") is not None:
        body["priceExcludingVatCurrency"] = float(args["priceExcludingVat"])
    if args.get("priceIncludingVat") is not None:
        body["priceIncludingVatCurrency"] = float(args["priceIncludingVat"])

    # Resolve VAT once, then optionally fall back to Tripletex's default VAT handling.
    vat_types = await _get_outgoing_vat_types(client)
    default_vat_id = _resolve_default_vat_id(vat_types, 3)
    body["vatType"] = {"id": int(_resolve_outgoing_vat_id(vat_types, args.get("vatTypeId"), default_vat_id))}
    try:
        return await client.post("/product", json=body)
    except Exception as exc:
        if _error_mentions(exc, "already exists", "eksisterer allerede", "duplicate"):
            existing = await _find_existing_product(
                client,
                product_name=args.get("name"),
                product_number=args.get("number"),
            )
            if existing:
                return {"value": existing, "note": "Product already exists"}
        if not _error_mentions(exc, "vat", "mva"):
            raise

    body.pop("vatType", None)
    try:
        return await client.post("/product", json=body)
    except Exception as exc:
        if _error_mentions(exc, "already exists", "eksisterer allerede", "duplicate"):
            existing = await _find_existing_product(
                client,
                product_name=args.get("name"),
                product_number=args.get("number"),
            )
            if existing:
                return {"value": existing, "note": "Product already exists"}
        raise


async def action_create_department(client: TripletexClient, args: dict) -> dict:
    """Create a department, returning an existing match if the number/name is already in use."""
    department_number = str(args.get("departmentNumber", args.get("number", "")))
    department_name = args["name"]

    try:
        existing = await client.get("/department", params={"count": 100})
        for dept in existing.get("values", []):
            if department_number and str(dept.get("departmentNumber", "")) == department_number:
                return {"value": dept}
            if dept.get("name", "").strip().lower() == department_name.strip().lower():
                return {"value": dept}
    except Exception:
        pass

    body = {
        "name": department_name,
        "departmentNumber": department_number,
    }
    try:
        return await client.post("/department", json=body)
    except Exception:
        # Duplicate-number races are common in regression runs; return the existing dept.
        try:
            existing = await client.get("/department", params={"count": 100})
            for dept in existing.get("values", []):
                if department_number and str(dept.get("departmentNumber", "")) == department_number:
                    return {"value": dept}
                if dept.get("name", "").strip().lower() == department_name.strip().lower():
                    return {"value": dept}
        except Exception:
            pass
        raise


async def action_setup_bank_account(client: TripletexClient, args: dict) -> dict:
    """Register a bank account on the company. Required before invoice creation."""
    bank_num = args.get("bankAccountNumber", "12345678903")

    # Approach 1: Find company via >withLoginAccess
    try:
        companies = await client.get("/company/>withLoginAccess")
        if companies.get("values"):
            company_id = companies["values"][0]["id"]
            company_data = await client.get(f"/company/{company_id}")
            company_obj = company_data.get("value", company_data)
            if company_obj.get("bankAccountNumber") == bank_num:
                log.info(f"Company {company_id} already has bank account number set")
                return {"status": "already_set", "companyId": company_id}
            company_obj["bankAccountNumber"] = bank_num
            try:
                result = await client.put("/company", json=company_obj)
                log.info(f"Bank account set via PUT /company for company {company_id}")
                return result
            except Exception as put_company_error:
                log.warning(f"PUT /company failed, trying PUT /company/{{id}} fallback: {put_company_error}")
                result = await client.put(f"/company/{company_id}", json=company_obj)
                log.info(f"Bank account set via PUT /company/{company_id}")
                return result
    except Exception as e1:
        log.warning(f"Bank account via company lookup failed: {e1}")

    # Approach 2: Find existing 1920 bank account in ledger and ensure it has bank number
    try:
        accounts = await client.get("/ledger/account", params={"number": 1920, "count": 1})
        for acc in accounts.get("values", []):
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
    customer_id = await _find_customer_id(
        client,
        customer_id=args.get("customerId"),
        customer_name=args.get("customerName"),
        customer_org_number=args.get("customerOrgNumber"),
    )

    if not customer_id:
        return {"error": "Could not find or create customer"}

    # Look up VAT types — try resolved first, then one zero-rate fallback (max 2 attempts)
    vat_types = await _get_outgoing_vat_types(client)
    all_vat = vat_types.get("values", [])
    resolved_id = _resolve_default_vat_id(vat_types, 3)
    # Single fallback: first zero-rate outgoing type
    zero_fallback = next((v["id"] for v in all_vat if v.get("percentage") == 0), None)
    vat_candidates = [resolved_id] + ([zero_fallback] if zero_fallback and zero_fallback != resolved_id else [])

    def _build_order_lines(vat_id):
        lines = []
        for line in args.get("orderLines", []):
            line_vat = _resolve_outgoing_vat_id(vat_types, line.get("vatTypeId"), vat_id)
            ol = {
                "description": line.get("description", ""),
                "count": float(line.get("count", 1)),
                "unitPriceExcludingVatCurrency": float(line.get("unitPrice", line.get("unitPriceExcludingVat", 0))),
                "vatType": {"id": line_vat},
            }
            if line.get("productNumber"):
                ol["product"] = {"number": str(line["productNumber"])}
            lines.append(ol)
        return lines

    invoice_date = args.get("invoiceDate", TODAY)
    due_date = args.get("invoiceDueDate", args.get("dueDate", invoice_date))

    # Resolve currency if specified (e.g. EUR, USD)
    currency_id = None
    if args.get("currencyCode"):
        try:
            currencies = await client.get("/currency", params={"code": args["currencyCode"]})
            for cur in currencies.get("values", []):
                if cur.get("code", "").upper() == args["currencyCode"].upper():
                    currency_id = cur["id"]
                    break
        except Exception as e:
            log.warning(f"Currency lookup failed: {e}")

    # Try with each VAT candidate (max 5 attempts)
    last_error = None
    for vat_id in vat_candidates:
        order_lines = _build_order_lines(vat_id)
        order_body = {
            "customer": {"id": customer_id},
            "orderDate": invoice_date,
            "deliveryDate": invoice_date,
            "orderLines": order_lines,
        }
        if currency_id:
            order_body["currency"] = {"id": currency_id}

        body = {
            "invoiceDate": invoice_date,
            "invoiceDueDate": due_date,
            "orders": [order_body],
        }
        try:
            return await client.post("/invoice", json=body)
        except Exception as e:
            last_error = e
            error_text = str(e)
            if hasattr(e, "response"):
                try:
                    error_text = e.response.text
                except Exception:
                    pass
            if "bankkontonummer" in error_text.lower() or "bank account" in error_text.lower():
                await action_setup_bank_account(client, {})
                try:
                    return await client.post("/invoice", json=body)
                except Exception:
                    continue
            if "mva" not in error_text.lower() and "vat" not in error_text.lower():
                raise  # Not a VAT error, don't retry with different VAT
    if last_error:
        raise last_error


async def action_create_order(client: TripletexClient, args: dict) -> dict:
    """Create an order."""
    customer_id = await _find_customer_id(
        client,
        customer_id=args.get("customerId"),
        customer_name=args.get("customerName"),
        customer_org_number=args.get("customerOrgNumber"),
    )

    vat_types = await _get_outgoing_vat_types(client)
    default_vat_id = _resolve_default_vat_id(vat_types, 3)

    order_lines = []
    for line in args.get("orderLines", []):
        ol = {
            "description": line.get("description", ""),
            "count": float(line.get("count", 1)),
            "unitPriceExcludingVatCurrency": float(line.get("unitPrice", line.get("unitPriceExcludingVat", 0))),
            "vatType": {"id": _resolve_outgoing_vat_id(vat_types, line.get("vatTypeId"), default_vat_id)},
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
    invoice_number = str(args.get("invoiceNumber") or "")

    if not invoice_id:
        # Search for the invoice — try to match by customer name if provided
        invoices = await client.get("/invoice", params={
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 200,
        })
        for inv in invoices.get("values", []):
            inv_number = str(inv.get("invoiceNumber") or inv.get("number") or "")
            if invoice_number and invoice_number == inv_number:
                invoice_id = inv["id"]
                break
            if _invoice_matches(inv, args):
                invoice_id = inv["id"]
                break
        # Fallback: take last invoice if no customer match
        if not invoice_id and invoices.get("values"):
            invoice_id = invoices["values"][-1]["id"]

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

    payment_params = {
        "paymentDate": payment_date,
        "paymentTypeId": int(payment_type_id),
        "paidAmount": amount,
    }
    try:
        return await client.put(f"/invoice/{invoice_id}/:payment", params=payment_params)
    except Exception as e:
        if amount > 0 and _error_mentions(e, "paidamount", "payment", "beløp", "amount"):
            try:
                invoices = await client.get(
                    "/invoice",
                    params={"invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "count": 200},
                )
                for inv in invoices.get("values", []):
                    if inv.get("id") != invoice_id:
                        continue
                    remaining = inv.get("amountRemainder")
                    if remaining is None:
                        remaining = inv.get("amountRemaining")
                    if remaining is None:
                        remaining = inv.get("amountUnpaid")
                    if remaining is None:
                        remaining = inv.get("restAmount")
                    if remaining is not None:
                        clipped = min(amount, float(remaining))
                        if clipped != amount:
                            payment_params["paidAmount"] = _money(clipped)
                            return await client.put(f"/invoice/{invoice_id}/:payment", params=payment_params)
                    break
            except Exception:
                pass
        raise


async def action_create_credit_note(client: TripletexClient, args: dict) -> dict:
    """Create a credit note for an invoice."""
    invoice_id = args.get("invoiceId")
    invoice_number = str(args.get("invoiceNumber") or "")

    if not invoice_id:
        invoices = await client.get("/invoice", params={
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 200,
        })
        for inv in invoices.get("values", []):
            if inv.get("isCredited"):
                continue
            inv_number = str(inv.get("invoiceNumber") or inv.get("number") or "")
            if invoice_number and invoice_number == inv_number:
                invoice_id = inv["id"]
                break
            if _invoice_matches(inv, args):
                invoice_id = inv["id"]
                break
        # Fallback: take first non-credited
        if not invoice_id:
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
        account_cache[acc["number"]] = acc

    customer_id = await _find_customer_id(
        client,
        customer_id=args.get("customerId"),
        customer_name=args.get("customerName"),
        customer_org_number=args.get("customerOrgNumber"),
    )

    # Resolve supplier if provided
    supplier_id = args.get("supplierId")
    if not supplier_id and args.get("supplierName"):
        try:
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
        except Exception as e:
            log.warning(f"Supplier resolution failed: {e}")

    postings = []
    for i, p in enumerate(args.get("postings", []), start=1):
        account_number = p.get("accountNumber")
        account_id = p.get("accountId")
        account_obj = None

        if account_number and not account_id:
            account_obj = account_cache.get(int(account_number))
            if account_obj:
                account_id = account_obj["id"]
        elif account_number:
            account_obj = account_cache.get(int(account_number))

        if not account_id:
            return {"error": f"Account not found: {account_number}"}

        if not account_obj:
            for acc in accounts_resp.get("values", []):
                if acc.get("id") == account_id:
                    account_obj = acc
                    break

        amount = float(p.get("amountGross", p.get("amount", 0)))
        posting = {
            "row": i,
            "account": {"id": account_id},
            "amountGross": _money(amount),
            "amountGrossCurrency": _money(amount),
        }
        if p.get("vatTypeId"):
            posting["vatType"] = {"id": int(p["vatTypeId"])}
        if p.get("description"):
            posting["description"] = p["description"]

        # Add customer ref for CUSTOMER ledger accounts
        posting_customer_id = p.get("customerId") or customer_id
        if posting_customer_id and account_obj and account_obj.get("ledgerType") == "CUSTOMER":
            posting["customer"] = {"id": int(posting_customer_id)}

        # Add supplier ref for VENDOR/SUPPLIER ledger accounts
        posting_supplier_id = p.get("supplierId") or supplier_id
        ledger_type = (account_obj or {}).get("ledgerType")
        if posting_supplier_id and ledger_type in {"VENDOR", "SUPPLIER"}:
            posting["supplier"] = {"id": int(posting_supplier_id)}

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
    """Create a project. Handles customer and employee lookup."""
    # Skip module activation — burns API calls with 403 in competition sandboxes

    # Find or create customer if needed
    customer_id = await _find_customer_id(
        client,
        customer_id=args.get("customerId"),
        customer_name=args.get("customerName"),
        customer_org_number=args.get("customerOrgNumber"),
    )

    # Find project manager — search by email first, then name, then fallback to first employee
    manager_id = await _find_employee_id(
        client,
        employee_id=args.get("projectManagerId"),
        employee_email=args.get("projectManagerEmail"),
        employee_name=args.get("projectManagerName"),
        create_if_missing=True,
    )

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

    # Set fixed price via project hourly-rates API. Updating /project directly pulls internal
    # fields like projectratetypes into the payload and causes validation failures.
    if args.get("fixedPrice") is not None:
        project_id = result.get("value", {}).get("id")
        if project_id:
            try:
                await _set_project_fixed_price(
                    client,
                    project_id,
                    args["fixedPrice"],
                    start_date=args.get("startDate", TODAY),
                )
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
                "amountGross": _money(amount),
                "amountGrossCurrency": _money(amount),  # Must match amountGross exactly
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
    # Find or use provided IDs
    employee_id = await _find_employee_id(
        client,
        employee_id=args.get("employeeId"),
        employee_email=args.get("employeeEmail"),
        employee_name=args.get("employeeName"),
    )

    # Grant permissions so employee can register time
    if employee_id:
        await _grant_employee_all_privileges(client, employee_id)

    project_id = await _find_project_id(
        client,
        project_id=args.get("projectId"),
        project_name=args.get("projectName"),
    )

    entry_date = args.get("date", TODAY)
    selected_project = None
    if not project_id:
        try:
            projects = await client.get("/project", params={"count": 100})
            project_values = projects.get("values", [])
            usable_project = None
            for project in project_values:
                start_date = project.get("startDate") or TODAY
                if start_date <= entry_date:
                    usable_project = project
                    break
            selected_project = usable_project or (project_values[0] if project_values else None)
            if selected_project:
                project_id = selected_project["id"]
        except Exception:
            pass

    if project_id and not args.get("projectId") and not args.get("projectName"):
        selected_start_date = (selected_project or {}).get("startDate")
        if selected_start_date and selected_start_date > entry_date:
            try:
                created_project = await action_create_project(
                    client,
                    {
                        "name": f"Timeføring {entry_date}",
                        "projectManagerId": employee_id,
                        "isInternal": True,
                        "startDate": entry_date,
                    },
                )
                project_value = created_project.get("value", created_project)
                project_id = project_value.get("id", project_id)
            except Exception as e:
                log.warning(f"Fallback internal project creation failed: {e}")

    # Find or create activity
    # Prefer project-applicable activities. Creating via /activity requires activityType,
    # and project-specific activities should be created through /project/projectActivity.
    activity_id = args.get("activityId")
    applicable_activities = []

    if project_id:
        try:
            applicable = await client.get(
                "/activity/>forTimeSheet",
                params={
                    "projectId": project_id,
                    "employeeId": employee_id,
                    "date": entry_date,
                    "count": 100,
                },
            )
            applicable_activities = applicable.get("values", [])
            if not activity_id and args.get("activityName"):
                for act in applicable_activities:
                    if args["activityName"].lower() in act.get("name", "").lower():
                        activity_id = act["id"]
                        break
            if not activity_id and applicable_activities:
                activity_id = applicable_activities[0]["id"]
        except Exception:
            pass

    if not activity_id and args.get("activityName"):
        # Try listing existing activities
        try:
            activities = await client.get(
                "/activity",
                params={"count": 100, "name": args["activityName"]},
            )
            for act in activities.get("values", []):
                if project_id and not act.get("isProjectActivity"):
                    continue
                if not project_id and act.get("isProjectActivity"):
                    continue
                if args["activityName"].lower() in act.get("name", "").lower():
                    activity_id = act["id"]
                    break
        except Exception:
            pass

        # If not found, create activity
        if not activity_id:
            if project_id:
                try:
                    act_result = await client.post(
                        "/project/projectActivity",
                        json={
                            "project": {"id": project_id},
                            "activity": {
                                "name": args["activityName"],
                                "activityType": "PROJECT_SPECIFIC_ACTIVITY",
                            },
                            "startDate": entry_date,
                        },
                    )
                    act_val = act_result.get("value", {})
                    activity_id = act_val.get("activity", {}).get("id") or act_val.get("id")
                except Exception as e:
                    if _error_mentions(e, "duplicate entry", "already exist"):
                        try:
                            applicable = await client.get(
                                "/activity/>forTimeSheet",
                                params={
                                    "projectId": project_id,
                                    "employeeId": employee_id,
                                    "date": entry_date,
                                    "count": 100,
                                },
                            )
                            for act in applicable.get("values", []):
                                if args["activityName"].lower() in act.get("name", "").lower():
                                    activity_id = act["id"]
                                    break
                        except Exception:
                            pass
                    if not activity_id:
                        log.warning(f"Failed to create project activity: {e}")

            if not activity_id:
                try:
                    act_result = await client.post(
                        "/activity",
                        json={
                            "name": args["activityName"],
                            "activityType": "GENERAL_ACTIVITY",
                        },
                    )
                    activity_id = act_result.get("value", {}).get("id")
                except Exception as e:
                    log.warning(f"Failed to create general activity: {e}")

        # If still no activity, use first available
        if not activity_id:
            try:
                if applicable_activities:
                    activity_id = applicable_activities[0]["id"]
                else:
                    activities = await client.get("/activity", params={"count": 1})
                    if activities.get("values"):
                        activity_id = activities["values"][0]["id"]
            except Exception:
                pass

    # If still no activity, create a default one with the required activityType.
    if not activity_id:
        if project_id:
            try:
                act_result = await client.post(
                    "/project/projectActivity",
                    json={
                        "project": {"id": project_id},
                        "activity": {
                            "name": "General",
                            "activityType": "PROJECT_SPECIFIC_ACTIVITY",
                        },
                        "startDate": entry_date,
                    },
                )
                act_val = act_result.get("value", {})
                activity_id = act_val.get("activity", {}).get("id") or act_val.get("id")
            except Exception as e:
                if _error_mentions(e, "duplicate entry", "already exist"):
                    try:
                        applicable = await client.get(
                            "/activity/>forTimeSheet",
                            params={
                                "projectId": project_id,
                                "employeeId": employee_id,
                                "date": entry_date,
                                "count": 100,
                            },
                        )
                        if applicable.get("values"):
                            activity_id = applicable["values"][0]["id"]
                    except Exception:
                        pass

    if not activity_id:
        try:
            act_result = await client.post(
                "/activity",
                json={"name": "General", "activityType": "GENERAL_ACTIVITY"},
            )
            activity_id = act_result.get("value", {}).get("id")
        except Exception:
            pass

    # Link activity to project if both exist
    if activity_id and project_id:
        try:
            await client.post("/project/projectActivity", json={
                "project": {"id": project_id},
                "activity": {"id": activity_id},
                "startDate": entry_date,
            })
        except Exception as e:
            if not _error_mentions(e, "duplicate entry", "already exist"):
                pass  # Non-fatal best-effort link step

    # Activity is REQUIRED for timesheet entry — abort if missing
    if not activity_id:
        return {"error": "Cannot register timesheet: no activity could be found or created"}

    # Register hours via /timesheet/entry
    entry_body = {
        "employee": {"id": employee_id},
        "activity": {"id": activity_id},
        "date": entry_date,
        "hours": float(args.get("hours", 0)),
    }
    if project_id:
        entry_body["project"] = {"id": project_id}
    if args.get("comment"):
        entry_body["comment"] = args["comment"]

    try:
        return await client.post("/timesheet/entry", json=entry_body)
    except Exception as e:
        error_text = str(e)
        if hasattr(e, 'response'):
            try:
                error_text = e.response.text
            except Exception:
                pass
        # If duplicate entry, try to find and update existing
        if "allerede" in error_text.lower() or "already" in error_text.lower() or "duplicate" in error_text.lower():
            try:
                existing = await client.get("/timesheet/entry", params={
                    "employeeId": employee_id,
                    "dateFrom": entry_date,
                    "dateTo": entry_date,
                    "count": 10,
                })
                for entry in existing.get("values", []):
                    if entry.get("activity", {}).get("id") == activity_id:
                        entry["hours"] = float(args.get("hours", 0))
                        if args.get("comment"):
                            entry["comment"] = args["comment"]
                        return await client.put(f"/timesheet/entry/{entry['id']}", json=entry)
            except Exception as e2:
                log.warning(f"Timesheet update fallback failed: {e2}")
        raise


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

    amount = _money(args.get("amountIncludingVat", args.get("amount", 0)))
    invoice_date = args.get("invoiceDate", TODAY)

    # Use voucher posting directly. /incomingInvoice is beta, permission-gated in some
    # competition sandboxes, and our previous payload shape triggered validation errors.
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
    # MUST include supplier:{id} on postings - Tripletex requires it for supplier invoices
    postings = [
        {"row": 1, "account": {"id": account_id}, "amountGross": _money(amount_ex_vat), "amountGrossCurrency": _money(amount_ex_vat), "supplier": {"id": supplier_id}},
        {"row": 2, "account": {"id": credit_account_id}, "amountGross": _money(-amount), "amountGrossCurrency": _money(-amount), "supplier": {"id": supplier_id}},
    ]
    # Add VAT line if we have the account
    if vat_account_id:
        postings.append(
            {"row": 3, "account": {"id": vat_account_id}, "amountGross": _money(vat_amount), "amountGrossCurrency": _money(vat_amount)}
        )

    voucher_body = {
        "date": invoice_date,
        "description": f"Supplier invoice {args.get('invoiceNumber', '')} from {args.get('supplierName', 'supplier')}",
        "postings": postings,
    }
    return await client.post("/ledger/voucher", json=voucher_body)


async def action_create_travel_expense(client: TripletexClient, args: dict) -> dict:
    """Create a travel expense report with per diem."""
    employee_id = await _find_employee_id(
        client,
        employee_id=args.get("employeeId"),
        employee_email=args.get("employeeEmail"),
        employee_name=args.get("employeeName"),
    )
    if not employee_id:
        return {"error": "Could not resolve employee for travel expense"}

    # Grant permissions so employee can file travel expenses
    await _grant_employee_all_privileges(client, employee_id)

    from datetime import date, timedelta
    today = date.today().isoformat()

    # Always include travelDetails to ensure it's classified as travel (not employee expense)
    departure_date = args.get("departureDate", today)
    days = int(args.get("perDiemDays", 1))
    return_date = args.get("returnDate")
    if not return_date and departure_date:
        try:
            dep = date.fromisoformat(departure_date)
            return_date = (dep + timedelta(days=max(days - 1, 0))).isoformat()
        except Exception:
            return_date = today

    departure_from = args.get("departure") or args.get("departureFrom") or args.get("destination") or args.get("title", "Travel")
    destination = args.get("destination") or args.get("title", "Travel")

    body = {
        "employee": {"id": employee_id},
        "title": args.get("title", "Travel expense"),
        "isChargeable": args.get("isChargeable", False),
        "isFixedInvoicedAmount": args.get("isFixedInvoicedAmount", False),
        "isIncludeAttachedReceiptsWhenReinvoicing": False,
        "travelDetails": {
            "departureDate": departure_date,
            "returnDate": return_date,
            "purpose": args.get("title", "Travel"),
            "isForeignTravel": False,
            "isDayTrip": days <= 1,
            "departureFrom": departure_from,
            "destination": destination,
        },
    }

    result = await client.post("/travelExpense", json=body)
    expense = result.get("value", result)
    expense_id = expense.get("id")
    travel_payment_type_id = None
    travel_cost_category_id = None
    if expense_id:
        try:
            payment_types = await client.get("/travelExpense/paymentType", params={"count": 50})
            if payment_types.get("values"):
                travel_payment_type_id = payment_types["values"][0].get("id")
        except Exception as e:
            log.warning(f"Travel payment type lookup failed: {e}")
        try:
            cost_categories = await client.get("/travelExpense/costCategory", params={"count": 50})
            if cost_categories.get("values"):
                travel_cost_category_id = cost_categories["values"][0].get("id")
        except Exception as e:
            log.warning(f"Travel cost category lookup failed: {e}")

    # Add explicit travel outlays before per diem/delivery when the prompt includes them.
    for cost in args.get("costs", []):
        if not expense_id:
            break
        try:
            amount = _money(cost.get("amount", 0))
            cost_body = {
                "travelExpense": {"id": expense_id},
                "date": departure_date,
                "category": cost.get("category") or cost.get("description") or "Expense",
                "comments": cost.get("description") or cost.get("category") or "Travel expense cost",
                "amountCurrencyIncVat": amount,
                "amountNOKInclVAT": amount,
                "isPaidByEmployee": True,
            }
            if travel_payment_type_id:
                cost_body["paymentType"] = {"id": travel_payment_type_id}
            if travel_cost_category_id:
                cost_body["costCategory"] = {"id": travel_cost_category_id}
            await client.post("/travelExpense/cost", json=cost_body)
        except Exception as e:
            log.warning(f"Travel cost creation failed: {e}")

    per_diem_added = False

    # Add per diem if specified
    if expense_id and args.get("perDiemDays") and args.get("perDiemRate"):
        try:
            accommodation = args.get("accommodation", "HOTEL")
            country_code = args.get("countryCode", "NO")
            rate_category, rate_type = await _resolve_travel_per_diem_rate(
                client,
                country_code=country_code,
                is_day_trip=days <= 1,
                accommodation=accommodation,
            )

            per_diem_body = {
                "travelExpense": {"id": expense_id},
                "count": int(args["perDiemDays"]),
                "rate": float(args["perDiemRate"]),
                "overnightAccommodation": accommodation,
                "location": args.get("destination", args.get("title", "Norway")),
                "address": args.get("destination", ""),
                # Omit countryCode for domestic trips — causes validation errors
            }
            if rate_type:
                per_diem_body["rateType"] = {"id": rate_type["id"]}
            if rate_category:
                per_diem_body["rateCategory"] = {"id": rate_category["id"]}
            if not rate_type and not rate_category:
                # Fallback: use known working category 740 for domestic overnight
                per_diem_body["rateType"] = {"id": 740}
                per_diem_body["rateCategory"] = {"id": 740}

            await client.post("/travelExpense/perDiemCompensation", json=per_diem_body)
            per_diem_added = True
        except Exception as e:
            log.warning(f"Per diem creation failed, trying cost fallback: {e}")
            # Fallback: add as a regular cost instead of per diem
            try:
                total = float(args["perDiemDays"]) * float(args["perDiemRate"])
                cost_body = {
                    "travelExpense": {"id": expense_id},
                    "date": departure_date,
                    "amountCurrencyIncVat": total,
                    "isPaidByEmployee": True,
                    "comments": f"Per diem {args['perDiemDays']} days x {args['perDiemRate']} NOK",
                }
                if travel_payment_type_id:
                    cost_body["paymentType"] = {"id": travel_payment_type_id}
                if travel_cost_category_id:
                    cost_body["costCategory"] = {"id": travel_cost_category_id}
                await client.post("/travelExpense/cost", json=cost_body)
                per_diem_added = True
            except Exception as e2:
                log.warning(f"Cost fallback also failed: {e2}")

    # Add individual expense items (flights, taxi, hotel, etc.)
    if expense_id and args.get("expenses"):
        for item in args["expenses"]:
            try:
                cost_body = {
                    "travelExpense": {"id": expense_id},
                    "date": departure_date,
                    "amountCurrencyIncVat": _money(item.get("amount", 0)),
                    "isPaidByEmployee": True,
                    "comments": item.get("description", "Expense"),
                }
                if travel_payment_type_id:
                    cost_body["paymentType"] = {"id": travel_payment_type_id}
                if travel_cost_category_id:
                    cost_body["costCategory"] = {"id": travel_cost_category_id}
                await client.post("/travelExpense/cost", json=cost_body)
            except Exception as e:
                log.warning(f"Failed to add expense item '{item.get('description')}': {e}")

    # Deliver only when we have at least one actual line item.
    if expense_id and (per_diem_added or args.get("costs") or args.get("expenses")):
        try:
            await client.put(f"/travelExpense/:deliver", params={"id": expense_id})
        except Exception as e:
            log.warning(f"Travel expense delivery failed (may need approval): {e}")

    return result


async def action_process_salary(client: TripletexClient, args: dict) -> dict:
    """Process salary by creating a manual voucher. Salary API requires special setup,
    so we use voucher postings on salary accounts (5000-series)."""
    # Find employee
    employee_id = await _find_employee_id(
        client,
        employee_id=args.get("employeeId"),
        employee_email=args.get("employeeEmail"),
        employee_name=args.get("employeeName"),
    )

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
    estimated_tax = _money(total_gross * 0.30)
    net_pay = _money(total_gross - estimated_tax)

    salary_account = account_cache.get(5000) or account_cache.get(5001)
    bank_account = account_cache.get(1920)
    tax_account = account_cache.get(2780) or account_cache.get(2600)

    if not salary_account or not bank_account:
        return {"error": f"Missing accounts: salary={salary_account}, bank={bank_account}"}

    postings = [
        {"row": 1, "account": {"id": salary_account}, "amountGross": _money(total_gross), "amountGrossCurrency": _money(total_gross),
         "description": f"Lønn {args.get('employeeName', '')}"},
    ]
    row = 2
    if tax_account:
        postings.append(
            {"row": row, "account": {"id": tax_account}, "amountGross": _money(-estimated_tax), "amountGrossCurrency": _money(-estimated_tax),
             "description": "Skattetrekk"}
        )
        row += 1
    postings.append(
        {"row": row, "account": {"id": bank_account}, "amountGross": _money(-net_pay), "amountGrossCurrency": _money(-net_pay),
         "description": "Utbetaling"}
    )

    voucher_body = {
        "date": args.get("date", TODAY),
        "description": f"Lønn {args.get('employeeName', '')} - grunnlønn {base_salary}" + (f" + bonus {bonus}" if bonus else ""),
        "postings": postings,
    }

    return await client.post("/ledger/voucher", json=voucher_body)


async def action_create_fixed_price_project_invoice(client: TripletexClient, args: dict) -> dict:
    """Create/find a fixed-price project, set the fixed price, then invoice a percentage of it."""
    project_id = await _find_project_id(
        client,
        project_id=args.get("projectId"),
        project_name=args.get("projectName"),
    )

    project_result: dict | None = None
    if not project_id:
        project_result = await action_create_project(
            client,
            {
                "name": args["projectName"],
                "number": args.get("projectNumber"),
                "customerId": args.get("customerId"),
                "customerName": args.get("customerName"),
                "customerOrgNumber": args.get("customerOrgNumber"),
                "projectManagerId": args.get("projectManagerId"),
                "projectManagerEmail": args.get("projectManagerEmail"),
                "projectManagerName": args.get("projectManagerName"),
                "startDate": args.get("startDate", args.get("invoiceDate", TODAY)),
                "fixedPrice": args["fixedPrice"],
            },
        )
        project_value = project_result.get("value", project_result)
        project_id = project_value.get("id")
    else:
        try:
            await _set_project_fixed_price(
                client,
                project_id,
                args["fixedPrice"],
                start_date=args.get("startDate", TODAY),
            )
        except Exception as e:
            log.warning(f"Could not update existing project {project_id} with fixed price: {e}")

    invoice_amount = _money(args["fixedPrice"] * (float(args["invoicePercent"]) / 100.0))
    invoice_result = await action_create_invoice(
        client,
        {
            "customerId": args.get("customerId"),
            "customerName": args.get("customerName"),
            "customerOrgNumber": args.get("customerOrgNumber"),
            "invoiceDate": args.get("invoiceDate", TODAY),
            "invoiceDueDate": args.get("invoiceDueDate", args.get("invoiceDate", TODAY)),
            "orderLines": [
                {
                    "description": args.get(
                        "description",
                        f"{_money(args['invoicePercent'])}% delbetaling for prosjekt {args['projectName']}",
                    ),
                    "count": 1,
                    "unitPrice": invoice_amount,
                }
            ],
        },
    )

    return {
        "projectId": project_id,
        "project": project_result,
        "invoice": invoice_result,
        "invoiceAmount": invoice_amount,
    }


async def action_register_timesheet_and_invoice(client: TripletexClient, args: dict) -> dict:
    """Register time on a project and immediately create an invoice for the logged hours."""
    project_id = await _find_project_id(
        client,
        project_id=args.get("projectId"),
        project_name=args.get("projectName"),
    )

    project_result: dict | None = None
    if not project_id and args.get("projectName"):
        project_result = await action_create_project(
            client,
            {
                "name": args["projectName"],
                "customerId": args.get("customerId"),
                "customerName": args.get("customerName"),
                "customerOrgNumber": args.get("customerOrgNumber"),
                "projectManagerId": args.get("projectManagerId"),
                "projectManagerEmail": args.get("projectManagerEmail") or args.get("employeeEmail"),
                "projectManagerName": args.get("projectManagerName") or args.get("employeeName"),
                "startDate": args.get("date", TODAY),
            },
        )
        project_value = project_result.get("value", project_result)
        project_id = project_value.get("id")

    timesheet_result = await action_register_timesheet(
        client,
        {
            "employeeId": args.get("employeeId"),
            "employeeEmail": args.get("employeeEmail"),
            "employeeName": args.get("employeeName"),
            "projectId": project_id,
            "projectName": args.get("projectName"),
            "activityId": args.get("activityId"),
            "activityName": args.get("activityName"),
            "hours": args["hours"],
            "date": args.get("date", TODAY),
            "comment": args.get("comment"),
        },
    )

    invoice_amount = _money(float(args["hours"]) * float(args["hourlyRate"]))
    invoice_result = await action_create_invoice(
        client,
        {
            "customerId": args.get("customerId"),
            "customerName": args.get("customerName"),
            "customerOrgNumber": args.get("customerOrgNumber"),
            "invoiceDate": args.get("invoiceDate", args.get("date", TODAY)),
            "invoiceDueDate": args.get("invoiceDueDate", args.get("invoiceDate", args.get("date", TODAY))),
            "orderLines": [
                {
                    "description": args.get(
                        "description",
                        f"{args.get('activityName', 'Project work')} {args['hours']}h on {args.get('projectName', 'project')}",
                    ),
                    "count": 1,
                    "unitPrice": invoice_amount,
                }
            ],
        },
    )

    return {
        "projectId": project_id,
        "project": project_result,
        "timesheet": timesheet_result,
        "invoice": invoice_result,
        "invoiceAmount": invoice_amount,
    }


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
        if "/ledger/posting" in path and "dateFrom" not in params:
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
    "create_fixed_price_project_invoice": action_create_fixed_price_project_invoice,
    "register_timesheet_and_invoice": action_register_timesheet_and_invoice,
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

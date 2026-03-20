"""
Parse the Tripletex OpenAPI spec into per-family endpoint cards.
These cards are small enough to inject into the agent context.
"""

import json
from pathlib import Path

SPEC_PATH = "/tmp/tripletex_openapi.json"
OUTPUT_DIR = Path(__file__).parent / "endpoint_cards"
OUTPUT_DIR.mkdir(exist_ok=True)

# Map path prefixes to task families
FAMILY_PATHS = {
    "employee": ["/employee", "/employee/employment", "/employee/employment/details"],
    "customer": ["/customer", "/customer/category", "/contact", "/deliveryAddress"],
    "supplier": ["/supplier"],
    "product": ["/product", "/product/unit", "/product/group"],
    "invoice": ["/invoice", "/order", "/order/orderline"],
    "project": ["/project", "/project/category", "/project/hourlyRates", "/project/orderline"],
    "department": ["/department"],
    "travel_expense": ["/travelExpense", "/travelExpense/cost", "/travelExpense/mileageAllowance",
                       "/travelExpense/perDiemCompensation", "/travelExpense/accommodationAllowance",
                       "/travelExpense/costCategory", "/travelExpense/paymentType",
                       "/travelExpense/rate", "/travelExpense/rateCategory"],
    "voucher": ["/ledger/voucher", "/ledger/account", "/ledger/vatType",
                "/ledger/posting", "/ledger/paymentTypeOut", "/ledger/voucherType"],
    "salary": ["/salary/transaction", "/salary/settings", "/salary/paySlip",
               "/salary/type", "/salary/taxDeduction"],
    "timesheet": ["/timesheet/entry", "/timesheet/timeClock", "/timesheet/week",
                  "/timesheet/month", "/timesheet/allocated", "/activity"],
    "company": ["/company", "/company/salesmodules"],
    "bank": ["/bank/reconciliation", "/bank/statement"],
}


def extract_endpoint_card(spec, path_prefix, path_data):
    """Extract a concise endpoint card from an OpenAPI path."""
    cards = []
    for method, op in path_data.items():
        if method in ("get", "post", "put", "delete"):
            # Get parameters
            params = []
            for p in op.get("parameters", []):
                if p.get("required"):
                    params.append(f"{p['name']} ({p.get('in','query')}, REQUIRED)")
                elif p.get("in") == "query":
                    params.append(f"{p['name']} ({p.get('in','query')})")

            # Get request body schema
            body_schema = ""
            rb = op.get("requestBody", {})
            if rb:
                content = rb.get("content", {})
                for ct, ct_data in content.items():
                    schema = ct_data.get("schema", {})
                    if "$ref" in schema:
                        ref_name = schema["$ref"].split("/")[-1]
                        # Try to resolve
                        resolved = spec.get("components", {}).get("schemas", {}).get(ref_name, {})
                        props = resolved.get("properties", {})
                        if props:
                            fields = []
                            for fname, fdata in list(props.items())[:20]:
                                ftype = fdata.get("type", fdata.get("$ref", "").split("/")[-1] or "?")
                                fields.append(f"{fname}: {ftype}")
                            body_schema = ", ".join(fields)
                    elif schema.get("properties"):
                        fields = [f"{k}: {v.get('type','?')}" for k, v in list(schema["properties"].items())[:15]]
                        body_schema = ", ".join(fields)

            card = {
                "method": method.upper(),
                "path": path_prefix,
                "summary": op.get("summary", ""),
                "required_params": [p for p in params if "REQUIRED" in p],
                "optional_params": [p for p in params if "REQUIRED" not in p][:5],
                "body_fields": body_schema[:500] if body_schema else None,
            }
            cards.append(card)
    return cards


def main():
    spec = json.load(open(SPEC_PATH))
    paths = spec.get("paths", {})

    for family, prefixes in FAMILY_PATHS.items():
        family_cards = []
        for path_key, path_data in paths.items():
            for prefix in prefixes:
                # Match exact prefix or prefix with ID
                if path_key == prefix or path_key.startswith(prefix + "/") or path_key.startswith(prefix + "/{"):
                    cards = extract_endpoint_card(spec, path_key, path_data)
                    family_cards.extend(cards)

        # Write card file
        output = {
            "family": family,
            "endpoint_count": len(family_cards),
            "endpoints": family_cards,
        }

        out_file = OUTPUT_DIR / f"{family}.json"
        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"{family}: {len(family_cards)} endpoints -> {out_file}")

    # Also create a compact text version for each family
    for family in FAMILY_PATHS:
        card_file = OUTPUT_DIR / f"{family}.json"
        if card_file.exists():
            data = json.loads(card_file.read_text())
            lines = [f"# {family.upper()} API Endpoints\n"]
            for ep in data["endpoints"]:
                line = f"{ep['method']} {ep['path']}"
                if ep.get("summary"):
                    line += f" — {ep['summary']}"
                lines.append(line)
                if ep.get("required_params"):
                    lines.append(f"  Required params: {', '.join(ep['required_params'])}")
                if ep.get("body_fields"):
                    lines.append(f"  Body: {{{ep['body_fields'][:300]}}}")
                lines.append("")

            txt_file = OUTPUT_DIR / f"{family}.txt"
            txt_file.write_text("\n".join(lines))
            print(f"  -> {txt_file} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

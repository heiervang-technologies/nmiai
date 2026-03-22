"""Generate realistic test PDFs for adversarial prompts.
Creates employee contracts and supplier invoices with extractable text.
"""
import base64
import fitz  # PyMuPDF


def _create_pdf(lines: list[str], title: str = "") -> str:
    """Create a PDF with text content, return base64-encoded string."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    if title:
        page.insert_text((72, y), title, fontsize=16, fontname="helv")
        y += 30
        page.insert_text((72, y), "—" * 50, fontsize=10)
        y += 25
    for line in lines:
        if y > 750:
            page = doc.new_page()
            y = 60
        page.insert_text((72, y), line, fontsize=10, fontname="helv")
        y += 16
    raw = doc.tobytes()
    doc.close()
    return base64.b64encode(raw).decode()


def employee_contract_pt() -> dict:
    """Portuguese employee contract PDF (matches adversarial test #29)."""
    lines = [
        "CONTRATO DE TRABALHO",
        "",
        "Empregador: NM i AI Solutions AS",
        "Org.nr: 999888777",
        "",
        "Funcionário: Mariana Costa",
        "E-mail: mariana.costa@example.org",
        "Data de nascimento: 22.10.1999",
        "Número de identidade nacional: 22109922602",
        "",
        "Departamento: Kundeservice",
        "Código de ocupação (STYRK): 2511",
        "",
        "Data de início: 08.06.2026",
        "Tipo de emprego: Ordinário",
        "Percentagem de emprego: 100%",
        "Salário anual: 510 000 NOK",
        "Horário de trabalho: 7,5 horas por dia",
        "",
        "Local: Oslo, Noruega",
        "",
        "Assinatura do empregador: _______________",
        "Assinatura do funcionário: _______________",
    ]
    return {
        "filename": "files/arbeidskontrakt_pt_05.pdf",
        "mime_type": "application/pdf",
        "content_base64": _create_pdf(lines, "Contrato de Trabalho"),
    }


def employee_contract_es() -> dict:
    """Spanish employee contract PDF (matches adversarial test #32)."""
    lines = [
        "CONTRATO DE TRABAJO",
        "",
        "Empleador: NM i AI Solutions AS",
        "Org.nr: 999888777",
        "",
        "Empleado: Lucía Martínez",
        "Correo electrónico: lucia.martinez@example.org",
        "Fecha de nacimiento: 17.04.1998",
        "Número de identidad nacional: 17049848676",
        "",
        "Departamento: Markedsføring",
        "Código de ocupación (STYRK): 1211",
        "",
        "Fecha de inicio: 15.06.2026",
        "Tipo de empleo: Ordinario",
        "Porcentaje de empleo: 100%",
        "Salario anual: 550 000 NOK",
        "Horario de trabajo: 7,5 horas por día",
        "",
        "Lugar: Oslo, Noruega",
        "",
        "Firma del empleador: _______________",
        "Firma del empleado: _______________",
    ]
    return {
        "filename": "files/arbeidskontrakt_es_06.pdf",
        "mime_type": "application/pdf",
        "content_base64": _create_pdf(lines, "Contrato de Trabajo"),
    }


def employee_contract_nb() -> dict:
    """Norwegian employee contract PDF (for onboarding tasks)."""
    lines = [
        "ARBEIDSKONTRAKT",
        "",
        "Arbeidsgiver: NM i AI Solutions AS",
        "Org.nr: 999888777",
        "",
        "Ansatt: Erik Johansen",
        "E-post: erik.johansen@example.org",
        "Fødselsdato: 15.03.1992",
        "Personnummer: 15039254821",
        "",
        "Avdeling: Utvikling",
        "Yrkeskode (STYRK): 2512",
        "",
        "Startdato: 01.08.2026",
        "Ansettelsestype: Ordinær",
        "Stillingsprosent: 100%",
        "Årslønn: 620 000 NOK",
        "Arbeidstid: 7,5 timer per dag",
        "",
        "Sted: Oslo, Norge",
        "",
        "Arbeidsgivers underskrift: _______________",
        "Arbeidstakers underskrift: _______________",
    ]
    return {
        "filename": "files/arbeidskontrakt_nb_01.pdf",
        "mime_type": "application/pdf",
        "content_base64": _create_pdf(lines, "Arbeidskontrakt"),
    }


def supplier_invoice_fr() -> dict:
    """French supplier invoice PDF (matches adversarial test #34)."""
    lines = [
        "FACTURE FOURNISSEUR",
        "",
        "Fournisseur: Forêt SARL",
        "Numéro d'organisation: 900969147",
        "E-mail: facture@foretsarl.fr",
        "Adresse: 14 Rue de la Paix",
        "Code postal: 0250",
        "Ville: Oslo",
        "",
        "Numéro de facture: INV-2026-1881",
        "Date de facture: 19.02.2026",
        "Date d'échéance: 21.03.2026",
        "",
        "Description: Datautstyr",
        "Compte de charges: 6540",
        "Montant HT: 24 449,60 NOK",
        "TVA 25%: 6 112,40 NOK",
        "Montant TTC: 30 562,00 NOK",
        "",
        "Conditions de paiement: 30 jours",
    ]
    return {
        "filename": "files/leverandorfaktura_fr_08.pdf",
        "mime_type": "application/pdf",
        "content_base64": _create_pdf(lines, "Facture Fournisseur"),
    }


# Export all test PDFs
TEST_PDFS = {
    "employee_contract_pt": employee_contract_pt,
    "employee_contract_es": employee_contract_es,
    "employee_contract_nb": employee_contract_nb,
    "supplier_invoice_fr": supplier_invoice_fr,
}


if __name__ == "__main__":
    import fitz as _fitz
    for name, gen_fn in TEST_PDFS.items():
        pdf = gen_fn()
        raw = base64.b64decode(pdf["content_base64"])
        doc = _fitz.open(stream=raw, filetype="pdf")
        text = doc[0].get_text()
        doc.close()
        print(f"=== {name} ({pdf['filename']}) ===")
        print(f"  Size: {len(raw)} bytes")
        print(f"  Text preview: {text[:200]}...")
        print()

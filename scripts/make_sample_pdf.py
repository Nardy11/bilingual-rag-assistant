"""Build data/corpus/travel-policy.en.pdf, the sample PDF (two pages, so citations show page numbers).

The text is written for this repo and released under CC BY 4.0. Run once; the PDF is committed.
Requires reportlab (dev-only dependency): pip install reportlab
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "corpus" / "travel-policy.en.pdf"

PAGES = [
    [
        ("h1", "Falak Home Systems - Business Travel Policy"),
        (
            "p",
            "Falak Home Systems is a fictional company invented for this demo. This text is released under CC BY 4.0.",
        ),
        ("h2", "Booking"),
        (
            "p",
            "All business trips must be approved by the line manager before any booking is made. "
            "Flights and hotels are booked through the company travel desk at least 14 days before "
            "departure, except for urgent customer visits approved by a director.",
        ),
        ("h2", "Flights"),
        (
            "p",
            "Employees fly economy class on all flights. Directors may fly business class on flights "
            "longer than six hours. Upgrades paid with personal loyalty points are allowed. "
            "Checked baggage for equipment and samples is paid by the company.",
        ),
        ("h2", "Hotels"),
        (
            "p",
            "The hotel limit is 45 KWD per night inside the GCC and 70 KWD per night elsewhere. "
            "Employees who stay with friends or family instead of a hotel receive a flat allowance "
            "of 10 KWD per night.",
        ),
    ],
    [
        ("h2", "Daily allowance"),
        (
            "p",
            "A daily allowance (per diem) of 25 KWD covers meals and local transport on travel days. "
            "No receipts are needed for the daily allowance. Meals that are already provided by a "
            "conference or a customer reduce the allowance by 8 KWD each.",
        ),
        ("h2", "Visas, insurance and vaccinations"),
        (
            "p",
            "Visa fees, travel insurance and vaccinations required for the destination are paid by "
            "the company. Every traveller is covered by the company travel insurance, which includes "
            "emergency medical treatment up to 50,000 KWD.",
        ),
        ("h2", "After the trip"),
        (
            "p",
            "Travel expenses must be submitted through the expense app within 30 days of returning, "
            "following the same rules as other business expenses. Unused cash advances are returned "
            "to finance within five working days of the employee's return.",
        ),
    ],
]


def main() -> None:
    styles = getSampleStyleSheet()
    style = {"h1": styles["Title"], "h2": styles["Heading2"], "p": styles["BodyText"]}
    story = []
    for i, page in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        for kind, text in page:
            story.append(Paragraph(text, style[kind]))
            story.append(Spacer(1, 6))
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        title="Falak Business Travel Policy",
        author="Falak Home Systems (fictional)",
        invariant=1,
    )
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

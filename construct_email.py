# ----------------------------------------------------------------------
# Build a nicely formatted HTML e-mail for the daily preprint digest and
# send it via SMTP.  Favourite-author papers are highlighted and placed
# at the top of the list (sorting happens in main.py).
# ----------------------------------------------------------------------
from __future__ import annotations

import datetime
import math
import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import Sequence

from loguru import logger
from tqdm import tqdm

from paper import PreprintPaper            # unified preprint model
from fav_utils import is_fav_author        # <-- unified check

# ------------------------------ HTML -------------------------------- #
framework = """
<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {{
      font-family: Arial, sans-serif;
    }}
    .star-wrapper {{
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }}
    .half-star {{
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }}
    .full-star {{
      vertical-align: middle;
    }}
    .fav-author {{
      color: #d9534f;      /* orange-red */
      font-weight: 700;
    }}
  </style>
</head>
<body>
<div>
__CONTENT__
</div>

<br><br>
<div style="font-size:12px;color:#666;">
To unsubscribe, disable this workflow in your GitHub repository.
</div>
</body>
</html>
"""


def _empty_html() -> str:
    """HTML block when there are no new papers."""
    return """
    <table width="100%" style="border:1px solid #ddd;border-radius:8px;
                               padding:16px;background-color:#f9f9f9;">
      <tr>
        <td style="font-size:20px;font-weight:bold;color:#333;">
          No Preprints Today. Enjoy your day!
        </td>
      </tr>
    </table>
    """


# -------------------------- star rating ----------------------------- #
def _stars(score: float) -> str:
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low, high = 6, 8
    if score <= low:
        return ""
    if score >= high:
        return full_star * 5
    interval = (high - low) / 10
    star_count = math.ceil((score - low) / interval)
    full_cnt, half_cnt = divmod(star_count, 2)
    return (
        '<div class="star-wrapper">'
        + full_star * full_cnt
        + half_star * half_cnt
        + "</div>"
    )


# ----------------------- single-paper template ---------------------- #
def _paper_block(p: PreprintPaper) -> str:
    """Return the HTML block for one paper."""
    # Title prefix was already set via mark_paper() in main.py
    title_prefix = "🧑🏻‍🔬 " if getattr(p, "is_favorite", False) else ""
    title_html = f"{title_prefix}{p.title}"

    # Authors (highlight favourites)
    author_cells = []
    for a in p.authors:
        if is_fav_author(a):
            author_cells.append(f'🧑🏻‍🔬 <span class="fav-author">{a}</span>')
        else:
            author_cells.append(a)
    authors_html = ", ".join(author_cells)

    # Affiliations
    affil_html = ", ".join(p.affiliations) if getattr(p, "affiliations", None) else None

    # Open-Web button
    web_btn = (
        f'<a href="{p.url}" style="display:inline-block;text-decoration:none;'
        'font-size:14px;font-weight:bold;color:#fff;background-color:#0275d8;'
        'padding:8px 16px;border-radius:4px;">Open Web</a>'
        if p.url
        else ""
    )

    return f"""
    <table width="100%" style="border:1px solid #ddd;border-radius:8px;
                               padding:16px;background-color:#f9f9f9;">
      <tr>
        <td style="font-size:20px;font-weight:bold;color:#333;">{title_html}</td>
      </tr>
      <tr>
        <td style="font-size:14px;color:#666;padding:8px 0;">
          {authors_html}<br><i>{affil_html or 'Unknown Affiliation'}</i>
        </td>
      </tr>
      <tr>
        <td style="font-size:14px;color:#333;padding:8px 0;">
          <strong>Relevance:</strong> {_stars(p.score)}
        </td>
      </tr>
      <tr>
        <td style="font-size:14px;color:#333;padding:8px 0;">
          <strong>DOI:</strong> {p.doi}
        </td>
      </tr>
      <tr>
        <td style="font-size:14px;color:#333;padding:8px 0;">
          <strong>TLDR:</strong> {p.tldr_content}
        </td>
      </tr>
      <tr>
        <td style="padding:8px 0;">{web_btn}</td>
      </tr>
    </table>
    """


# --------------------------- renderer ------------------------------- #
def render_email(papers: Sequence[PreprintPaper]) -> str:
    """Assemble full HTML e-mail."""
    if not papers:
        return framework.replace("__CONTENT__", _empty_html())

    blocks = [_paper_block(p) for p in tqdm(papers, desc="Rendering e-mail")]
    content = "<br>" + "</br><br>".join(blocks) + "</br>"
    return framework.replace("__CONTENT__", content)


# ----------------------------- e-mail ------------------------------- #
def send_email(
    sender: str,
    receiver: str,
    password: str,
    smtp_server: str,
    smtp_port: int,
    html: str,
) -> None:
    """Send the HTML via SMTP."""

    def _fmt(addr: str) -> str:
        name, mail = parseaddr(addr)
        return formataddr((Header(name, "utf-8").encode(), mail))

    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = _fmt(f"Daily Preprint Bot <{sender}>")
    msg["To"] = _fmt(f"Reader <{receiver}>")
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    msg["Subject"] = Header(f"Daily Preprint Digest — {today}", "utf-8").encode()

    # Prefer STARTTLS; fall back to SSL
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"STARTTLS failed: {e} — falling back to SSL")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()

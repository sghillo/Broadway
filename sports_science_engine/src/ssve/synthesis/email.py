"""Email delivery for digests/memos. Dry-run by default so nothing is sent by accident."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def markdown_to_html(md: str) -> str:
    try:
        import markdown  # optional

        return markdown.markdown(md, extensions=["tables"])
    except ImportError:
        from html import escape

        return f"<pre style='font-family:inherit;white-space:pre-wrap'>{escape(md)}</pre>"


class EmailSender:
    def __init__(self, host: str | None = None, port: int | None = None,
                 username: str | None = None, password: str | None = None,
                 sender: str | None = None, dry_run: bool = True,
                 outbox: str | Path = "outbox") -> None:
        self.host = host or os.getenv("SMTP_HOST", "")
        self.port = port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.sender = sender or os.getenv("SMTP_FROM", self.username or "digest@localhost")
        self.dry_run = dry_run
        self.outbox = Path(outbox)

    def build(self, to: list[str], subject: str, markdown_body: str) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.set_content(markdown_body)
        msg.add_alternative(markdown_to_html(markdown_body), subtype="html")
        return msg

    def send(self, to: list[str], subject: str, markdown_body: str) -> Path | None:
        msg = self.build(to, subject, markdown_body)
        if self.dry_run or not self.host:
            self.outbox.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() else "_" for c in subject)[:80]
            path = self.outbox / f"{safe}.eml"
            path.write_bytes(bytes(msg))
            return path
        with smtplib.SMTP(self.host, self.port) as smtp:
            smtp.starttls()
            if self.username and self.password:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)
        return None

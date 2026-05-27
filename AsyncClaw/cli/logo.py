"""Static terminal wordmark for the AsyncClaw CLI."""

from __future__ import annotations

from rich.text import Text


def render_pixel_logo() -> Text:
    """Return a compact Claude/Codex-style AsyncClaw wordmark."""

    logo = Text()
    logo.append("╭", style="dim")
    logo.append("─" * 32, style="dim")
    logo.append("╮\n", style="dim")
    logo.append("│", style="dim")
    logo.append("  Async", style="bold #22d8cf")
    logo.append("Claw", style="bold #a56eff")
    logo.append("                     ")
    logo.append("│\n", style="dim")
    logo.append("│", style="dim")
    logo.append("  local agent runtime           ", style="#94a3b8")
    logo.append("│\n", style="dim")
    logo.append("╰", style="dim")
    logo.append("─" * 32, style="dim")
    logo.append("╯", style="dim")
    return logo

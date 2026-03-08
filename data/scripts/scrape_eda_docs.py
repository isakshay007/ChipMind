#!/usr/bin/env python3
"""
Scrape documentation from open-source EDA tools and Verilog tutorials.
Saves as text files for ChipMind's RAG pipeline.
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Project root: data/scripts/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "eda_docs"

USER_AGENT = "ChipMind/1.0 (EDA docs scraper; https://github.com/chipmind)"
RATE_LIMIT_SEC = 1.0

console = Console()


def _slug_from_url(url: str) -> str:
    """Extract a filesystem-safe slug from URL."""
    path = urlparse(url).path.rstrip("/")
    # Get last meaningful part
    parts = [p for p in path.split("/") if p and p not in ("wiki", "verilog", "en", "latest")]
    slug = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "index")
    # Sanitize
    slug = re.sub(r"[^\w\-]+", "_", slug).strip("_") or "page"
    return slug[:80]


def _extract_text(soup: BeautifulSoup, selectors: list[str] | None = None) -> str:
    """Extract main content from BeautifulSoup, stripping nav/ads/footer."""
    # Remove script, style, nav, footer, header
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    if selectors:
        content = soup.select_one(selectors[0])
        if content:
            return content.get_text(separator="\n", strip=True)

    # Fallback: try common content containers
    for sel in ["main", "article", ".content", "#content", ".main-content", "body"]:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def fetch_page(client: httpx.Client, url: str) -> str | None:
    """Fetch page HTML. Returns None on failure."""
    try:
        r = client.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        console.print(f"[dim]  Failed {url}: {e}[/dim]")
        return None


def scrape_yosys(client: httpx.Client, output_dir: Path) -> list[tuple[str, str]]:
    """Scrape Yosys ReadTheDocs. Returns list of (url, text)."""
    base = "https://yosyshq.readthedocs.io/projects/yosys/en/latest/"
    index_url = base
    html = fetch_page(client, index_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or "javascript:" in href:
            continue
        full = urljoin(base, href)
        if full.startswith(base) and full != base and ".html" in full:
            full = full.split("#")[0]
            if full.endswith(".html"):
                links.add(full)
    links.add(base.rstrip("/") + "/index.html")
    urls = sorted(links)
    results = []
    for i, url in enumerate(urls):
        slug = _slug_from_url(url)
        out_path = output_dir / f"yosys_{slug}.txt"
        if out_path.exists():
            continue
        console.print(f"  [dim]Yosys: page {i+1}/{len(urls)}[/dim]")
        time.sleep(RATE_LIMIT_SEC)
        html = fetch_page(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup, ["div.document", "article", ".wy-nav-content", "body"])
        if text and len(text) > 100:
            results.append((url, text))
    return results


def scrape_asic_world(client: httpx.Client, output_dir: Path) -> list[tuple[str, str]]:
    """Scrape ASIC World Verilog tutorial."""
    base = "http://www.asic-world.com/verilog/"
    index_url = base + "veritut.html"
    html = fetch_page(client, index_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = {index_url}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#") or "mailto:" in href:
            continue
        full = urljoin(base, href)
        if full.startswith(base) and "verilog" in full:
            links.add(full)
    urls = sorted(links)
    results = []
    for i, url in enumerate(urls):
        slug = _slug_from_url(url)
        out_path = output_dir / f"asic_world_{slug}.txt"
        if out_path.exists():
            continue
        console.print(f"  [dim]ASIC World: page {i+1}/{len(urls)}[/dim]")
        time.sleep(RATE_LIMIT_SEC)
        html = fetch_page(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup, ["table", "body"])
        if text and len(text) > 50:
            results.append((url, text))
    return results


def scrape_hdlbits(client: httpx.Client, output_dir: Path) -> list[tuple[str, str]]:
    """Scrape HDLBits tutorial/problem pages."""
    base = "https://hdlbits.01xz.net/wiki/"
    index_url = base + "Main_Page"
    html = fetch_page(client, index_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = {index_url, base + "Problem_sets"}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#") or "Special:" in href or "Category:" in href:
            continue
        full = urljoin(base, href)
        if full.startswith(base) and "wiki" in full:
            full = full.split("?")[0].split("#")[0]
            if "/wiki/" in full:
                links.add(full)
    urls = sorted(links)
    results = []
    for i, url in enumerate(urls):
        slug = _slug_from_url(url)
        out_path = output_dir / f"hdlbits_{slug}.txt"
        if out_path.exists():
            continue
        console.print(f"  [dim]HDLBits: page {i+1}/{len(urls)}[/dim]")
        time.sleep(RATE_LIMIT_SEC)
        html = fetch_page(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup, ["#mw-content-text", "#content", "main", "body"])
        if text and len(text) > 50:
            results.append((url, text))
    return results


def scrape_chipverify(client: httpx.Client, output_dir: Path) -> list[tuple[str, str]]:
    """Scrape ChipVerify Verilog tutorials."""
    base = "https://www.chipverify.com/verilog/"
    index_url = base + "verilog-tutorial"
    html = fetch_page(client, index_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = {index_url}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#") or "mailto:" in href:
            continue
        full = urljoin(base, href)
        if full.startswith("https://www.chipverify.com/verilog/") and "verilog" in full:
            links.add(full)
    urls = sorted(links)
    results = []
    for i, url in enumerate(urls):
        slug = _slug_from_url(url)
        out_path = output_dir / f"chipverify_{slug}.txt"
        if out_path.exists():
            continue
        console.print(f"  [dim]ChipVerify: page {i+1}/{len(urls)}[/dim]")
        time.sleep(RATE_LIMIT_SEC)
        html = fetch_page(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup, ["article", "main", ".content", "#content", "body"])
        if text and len(text) > 100:
            results.append((url, text))
    return results


def write_verilog_concepts() -> None:
    """Write the verilog_concepts.txt file."""
    content = '''Verilog Fundamental Concepts for Chip Design

Module: The basic building block in Verilog. Defines inputs, outputs, and behavior.
Syntax: module name(ports); ... endmodule

Wire: A net type that represents a physical connection. Cannot store values.
Reg: A variable type that can store values. Used in always blocks.

Combinational Logic: Output depends only on current inputs.
Uses: assign statements or always @(*) blocks.
Example: assign sum = a ^ b;

Sequential Logic: Output depends on inputs AND clock/state.
Uses: always @(posedge clk) blocks with non-blocking assignments (<=).
Example: always @(posedge clk) q <= d;

Finite State Machine (FSM): Sequential circuit with defined states and transitions.
Common pattern: Two always blocks - one for state register, one for next-state logic.

Testbench: A non-synthesizable module that drives inputs and checks outputs.
Uses: initial blocks, $display, $finish, clock generation.

Synthesis: Converting Verilog RTL code into a gate-level netlist.
Tool: Yosys (open-source), Synopsys Design Compiler (commercial).

Simulation: Running the design with test inputs to verify behavior.
Tool: Icarus Verilog (open-source), ModelSim (commercial).

Common EDA Flow: RTL Design → Simulation → Synthesis → Place & Route → Fabrication
'''
    path = OUTPUT_DIR / "verilog_concepts.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    console.print(f"[green]Created {path}[/green]")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_verilog_concepts()

    console.print(Panel("[bold]ChipMind — EDA Documentation Scraper[/bold]", style="blue"))

    headers = {"User-Agent": USER_AGENT}
    stats: dict[str, int] = {}
    total_size = 0

    sources = [
        ("yosys", "Yosys", scrape_yosys),
        ("asic_world", "ASIC World", scrape_asic_world),
        ("hdlbits", "HDLBits", scrape_hdlbits),
        ("chipverify", "ChipVerify", scrape_chipverify),
    ]

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for source_name, display_name, scrape_fn in sources:
            console.print(f"\n[bold]Scraping {display_name}...[/bold]")
            try:
                results = scrape_fn(client, OUTPUT_DIR)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                results = []

            count = 0
            for url, text in results:
                slug = _slug_from_url(url)
                out_path = OUTPUT_DIR / f"{source_name}_{slug}.txt"
                try:
                    out_path.write_text(text, encoding="utf-8")
                    count += 1
                    total_size += len(text.encode("utf-8"))
                except OSError as e:
                    console.print(f"[dim]  Write failed {out_path}: {e}[/dim]")

            stats[display_name] = count
            console.print(f"  [green]Saved {count} new pages[/green]")

    # Compute total files and size (all files in output dir)
    all_files = list(OUTPUT_DIR.glob("*.txt"))
    total_files = len(all_files)
    total_size = sum(f.stat().st_size for f in all_files)

    # Summary
    table = Table(title="Scrape Summary")
    table.add_column("Source", style="cyan")
    table.add_column("Pages scraped (this run)", style="green")
    for name, n in stats.items():
        table.add_row(name, str(n))
    table.add_row("verilog_concepts.txt", "(created)")
    table.add_row("Total text files", str(total_files))
    table.add_row("Total size", f"{total_size / 1024:.1f} KB")
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

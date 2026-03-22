"""
compare_results.py

Loads all eval_k*.json files from results/metrics/ and prints a
comparison table plus generates an HTML report with charts.

Usage:
    python compare_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import METRICS_DIR, PLOTS_DIR


# -- Load ---------------------------------------------------------------------

def load_all_results() -> list[dict]:
    """
    Loads every eval_k*.json file found in results/metrics/.

    Returns:
        List of dicts, each containing 'k' and the aggregated metrics,
        sorted by k ascending.
    """
    files = sorted(METRICS_DIR.glob("eval_k*.json"))
    if not files:
        print(f"No result files found in {METRICS_DIR}")
        print("Run: python main.py --k <value>")
        sys.exit(1)

    results = []
    for f in files:
        data = json.loads(f.read_text())
        agg  = data["aggregated"]
        k    = int(f.stem.replace("eval_k", ""))
        results.append({"k": k, **agg})

    return sorted(results, key=lambda r: r["k"])


# -- Table --------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Prints a formatted comparison table to the console."""
    metrics = ["MAP", "P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "NDCG@5", "NDCG@10", "NDCG@20"]

    # Header
    col_w = 10
    k_col = 6
    header = f"{'k':<{k_col}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print("  LSI-Rank — results comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best = {m: max(r[m] for r in results) for m in metrics}

    for r in results:
        row = f"{r['k']:<{k_col}}"
        for m in metrics:
            val  = r[m]
            mark = "*" if val == best[m] else " "
            row += f"{val:>{col_w - 1}.4f}{mark}"
        print(row)

    print("-" * len(header))
    print("  * = best value for that metric")
    print("=" * len(header) + "\n")


# -- HTML report --------------------------------------------------------------

def generate_html_report(results: list[dict]) -> Path:
    """
    Generates a standalone HTML file with interactive Chart.js charts
    comparing all k values across all metrics.

    Returns:
        Path to the generated HTML file.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "comparison.html"

    k_values  = [r["k"] for r in results]
    map_vals  = [r["MAP"] for r in results]
    p5_vals   = [r["P@5"] for r in results]
    p10_vals  = [r["P@10"] for r in results]
    p20_vals  = [r["P@20"] for r in results]
    r5_vals   = [r["R@5"] for r in results]
    r10_vals  = [r["R@10"] for r in results]
    r20_vals  = [r["R@20"] for r in results]
    n5_vals   = [r["NDCG@5"] for r in results]
    n10_vals  = [r["NDCG@10"] for r in results]
    n20_vals  = [r["NDCG@20"] for r in results]

    labels_js = json.dumps([str(k) for k in k_values])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LSI-Rank — Results Comparison</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8f7f4;
      color: #1a1a1a;
      margin: 0;
      padding: 2rem;
    }}
    h1 {{ font-size: 1.4rem; font-weight: 500; margin: 0 0 0.25rem; }}
    p.sub {{ font-size: 0.875rem; color: #73726c; margin: 0 0 2rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: #fff;
      border: 0.5px solid #d3d1c7;
      border-radius: 12px;
      padding: 1.25rem;
    }}
    .card h2 {{
      font-size: 0.8rem;
      font-weight: 500;
      color: #73726c;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 0 0 1rem;
    }}
    .chart-wrap {{ position: relative; height: 220px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    th {{
      font-weight: 500;
      text-align: right;
      padding: 6px 10px;
      border-bottom: 1px solid #d3d1c7;
      color: #73726c;
      font-size: 0.78rem;
    }}
    th:first-child {{ text-align: left; }}
    td {{
      text-align: right;
      padding: 6px 10px;
      border-bottom: 0.5px solid #ebe9e0;
      font-variant-numeric: tabular-nums;
    }}
    td:first-child {{ text-align: left; font-weight: 500; }}
    td.best {{ color: #0f6e56; font-weight: 500; }}
    tr:last-child td {{ border-bottom: none; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 0.75rem;
      font-size: 0.78rem;
      color: #73726c;
    }}
    .legend span {{ display: flex; align-items: center; gap: 5px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  </style>
</head>
<body>
  <h1>LSI-Rank — Results comparison</h1>
  <p class="sub">k = latent dimensions used in SVD. Evaluated on CRAN (225 queries).</p>

  <div class="grid">

    <div class="card">
      <h2>MAP — Mean Average Precision</h2>
      <div class="chart-wrap"><canvas id="mapChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Precision@K</h2>
      <div class="legend">
        <span><span class="dot" style="background:#378ADD"></span>P@5</span>
        <span><span class="dot" style="background:#85B7EB"></span>P@10</span>
        <span><span class="dot" style="background:#B5D4F4"></span>P@20</span>
      </div>
      <div class="chart-wrap"><canvas id="precChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Recall@K</h2>
      <div class="legend">
        <span><span class="dot" style="background:#1D9E75"></span>R@5</span>
        <span><span class="dot" style="background:#5DCAA5"></span>R@10</span>
        <span><span class="dot" style="background:#9FE1CB"></span>R@20</span>
      </div>
      <div class="chart-wrap"><canvas id="recChart"></canvas></div>
    </div>

    <div class="card">
      <h2>NDCG@K</h2>
      <div class="legend">
        <span><span class="dot" style="background:#D85A30"></span>NDCG@5</span>
        <span><span class="dot" style="background:#F0997B"></span>NDCG@10</span>
        <span><span class="dot" style="background:#F5C4B3"></span>NDCG@20</span>
      </div>
      <div class="chart-wrap"><canvas id="ndcgChart"></canvas></div>
    </div>

  </div>

  <div class="card">
    <h2>Full comparison table</h2>
    <table id="compTable"></table>
  </div>

<script>
const labels  = {labels_js};
const results = {json.dumps(results)};

const METRICS = ["MAP","P@5","P@10","P@20","R@5","R@10","R@20","NDCG@5","NDCG@10","NDCG@20"];
const best = {{}};
METRICS.forEach(m => {{
  best[m] = Math.max(...results.map(r => r[m]));
}});

const shared = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ grid: {{ color: "rgba(0,0,0,0.05)" }}, ticks: {{ color: "#888780" }} }},
    y: {{ min: 0, grid: {{ color: "rgba(0,0,0,0.05)" }},
         ticks: {{ color: "#888780", callback: v => v.toFixed(2) }} }}
  }}
}};

function lineDs(data, color, label) {{
  return {{
    label, data,
    borderColor: color, backgroundColor: color + "22",
    borderWidth: 2, pointRadius: 4, pointHoverRadius: 6,
    tension: 0.35, fill: false
  }};
}}

new Chart(document.getElementById("mapChart"), {{
  type: "line",
  data: {{ labels, datasets: [lineDs({json.dumps(map_vals)}, "#378ADD", "MAP")] }},
  options: {{ ...shared, scales: {{ ...shared.scales, y: {{ ...shared.scales.y, suggestedMax: 0.5 }} }} }}
}});

new Chart(document.getElementById("precChart"), {{
  type: "line",
  data: {{ labels, datasets: [
    lineDs({json.dumps(p5_vals)},  "#378ADD", "P@5"),
    lineDs({json.dumps(p10_vals)}, "#85B7EB", "P@10"),
    lineDs({json.dumps(p20_vals)}, "#B5D4F4", "P@20"),
  ]}},
  options: shared
}});

new Chart(document.getElementById("recChart"), {{
  type: "line",
  data: {{ labels, datasets: [
    lineDs({json.dumps(r5_vals)},  "#1D9E75", "R@5"),
    lineDs({json.dumps(r10_vals)}, "#5DCAA5", "R@10"),
    lineDs({json.dumps(r20_vals)}, "#9FE1CB", "R@20"),
  ]}},
  options: shared
}});

new Chart(document.getElementById("ndcgChart"), {{
  type: "line",
  data: {{ labels, datasets: [
    lineDs({json.dumps(n5_vals)},  "#D85A30", "NDCG@5"),
    lineDs({json.dumps(n10_vals)}, "#F0997B", "NDCG@10"),
    lineDs({json.dumps(n20_vals)}, "#F5C4B3", "NDCG@20"),
  ]}},
  options: shared
}});

// Table
const tbl  = document.getElementById("compTable");
const cols = ["k", ...METRICS];
const head = "<tr>" + cols.map(c => `<th>${{c}}</th>`).join("") + "</tr>";
tbl.innerHTML = "<thead>" + head + "</thead><tbody>" +
  results.map(r => {{
    const cells = cols.map(c => {{
      if (c === "k") return `<td>${{r.k}}</td>`;
      const v   = r[c];
      const cls = v === best[c] ? ' class="best"' : '';
      return `<td${{cls}}>${{v.toFixed(4)}}</td>`;
    }});
    return "<tr>" + cells.join("") + "</tr>";
  }}).join("") + "</tbody>";
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


# -- Entry point --------------------------------------------------------------

def main() -> None:
    results = load_all_results()

    print_table(results)

    path = generate_html_report(results)
    print(f"  HTML report saved -> {path}")
    print(f"  Open it in your browser to see the interactive charts.\n")


if __name__ == "__main__":
    main()
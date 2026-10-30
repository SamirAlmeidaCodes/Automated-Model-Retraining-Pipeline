"""Server-rendered dashboard for the current model and retraining history."""

from __future__ import annotations

import html
from datetime import datetime, timezone


def render_dashboard(context: dict) -> str:
    """Return the full HTML page for ``GET /``."""
    current = context.get("current")
    versions = context.get("versions") or []
    runs = context.get("runs") or []
    primary = context.get("primary_metric") or "metric"
    higher_is_better = bool(context.get("higher_is_better", True))
    title = html.escape(context.get("app_name") or "Retraining pipeline")
    dataset = html.escape(context.get("dataset_name") or "No dataset loaded")
    schedule = html.escape(context.get("schedule") or "not scheduled")
    next_run = html.escape(context.get("next_run") or "not scheduled")
    metric_label = html.escape(primary)
    direction = "higher is better" if higher_is_better else "lower is better"

    if current is None:
        metric_value = "—"
        version_label = "none"
        rows_label = "—"
        observation_label = str(context.get("observation_count") or 0)
        status_class = "wait"
        status_text = "waiting for a model"
    else:
        metric_value = f"{current['primary_metric_value']:.4f}"
        version_label = current["version"]
        rows_label = f"{current['training_rows']} train / {current['test_rows']} holdout"
        observation_label = str(context.get("observation_count") or 0)
        status_class = "ok" if current.get("artifact_available") else "bad"
        status_text = "serving" if current.get("artifact_available") else "artifact missing"

    balance = _balance_text(context.get("class_counts") or [])
    chart, chart_note = _chart(versions, primary)
    version_rows = _version_rows(versions, higher_is_better)
    run_rows = _run_rows(runs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --ink: #1c1915;
      --muted: #5f584e;
      --line: #ddd4c4;
      --card: #fffdf8;
      --teal: #0f5c56;
      --teal-soft: #e4f2ef;
      --rust: #8d3f2f;
      --rust-soft: #f8ebe6;
      --amber: #8a5a12;
      --amber-soft: #f8efd8;
      --shadow: 0 1px 0 rgba(28, 25, 21, 0.05);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(900px 280px at 0% -10%, #efe2cc 0%, transparent 55%),
        var(--bg);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 28px 64px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--teal);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", Palatino, serif;
      font-size: 36px;
      font-weight: 500;
      letter-spacing: -0.03em;
    }}
    .dataset {{ margin: 6px 0 0; color: var(--muted); }}
    .side {{ text-align: right; color: var(--muted); font-size: 14px; }}
    .side strong {{ color: var(--ink); font-weight: 600; }}
    .pill {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .pill.ok {{ background: var(--teal-soft); color: var(--teal); }}
    .pill.wait {{ background: var(--amber-soft); color: var(--amber); }}
    .pill.bad {{ background: var(--rust-soft); color: var(--rust); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 22px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 14px 16px 16px;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }}
    .card strong {{
      display: block;
      margin-top: 8px;
      font-family: Georgia, "Iowan Old Style", Palatino, serif;
      font-size: 28px;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.03em;
    }}
    section {{ margin-top: 22px; }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 16px 18px 18px;
    }}
    h2 {{
      margin: 0 0 4px;
      font-family: Georgia, "Iowan Old Style", Palatino, serif;
      font-size: 22px;
      font-weight: 500;
    }}
    .sub {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
    .legend {{ display: flex; gap: 16px; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; }}
    .chart-scroll {{ overflow-x: auto; }}
    svg text {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
    .split {{ display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 12px; }}
    .table-scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; }}
    th, td {{ padding: 8px 8px 8px 0; border-bottom: 1px solid var(--line); vertical-align: top; }}
    .status {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 999px;
      font-size: 11px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .status.production, .status.promoted {{ background: var(--teal-soft); color: var(--teal); }}
    .status.rejected, .status.failed {{ background: var(--rust-soft); color: var(--rust); }}
    .status.archived, .status.skipped {{ background: #eeeae2; color: var(--muted); }}
    .status.busy {{ background: var(--amber-soft); color: var(--amber); }}
    .up {{ color: var(--teal); }}
    .down {{ color: var(--rust); }}
    .flat {{ color: var(--muted); }}
    .empty {{ color: var(--muted); margin: 8px 0; }}
    footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{
      .cards, .split {{ grid-template-columns: 1fr 1fr; }}
      .split {{ grid-template-columns: 1fr; }}
      header {{ flex-direction: column; align-items: flex-start; }}
      .side {{ text-align: left; }}
    }}
    @media (max-width: 560px) {{
      .wrap {{ padding: 16px 16px 48px; }}
      .cards {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <p class="eyebrow">Model foundry</p>
        <h1>{title}</h1>
        <p class="dataset">{dataset}{balance}</p>
      </div>
      <div class="side">
        <div class="pill {status_class}">{html.escape(status_text)}</div>
        <div>Schedule <strong>{schedule}</strong></div>
        <div>Next check <strong>{next_run}</strong></div>
      </div>
    </header>
    <div class="cards">
      <div class="card"><span>Production</span><strong>{html.escape(version_label)}</strong></div>
      <div class="card"><span>{metric_label}</span><strong>{metric_value}</strong></div>
      <div class="card"><span>Last split</span><strong style="font-size:22px">{html.escape(rows_label)}</strong></div>
      <div class="card"><span>Observations</span><strong>{html.escape(observation_label)}</strong></div>
    </div>
    <section class="panel">
      <h2>Comparison</h2>
      <p class="sub">Candidate {metric_label} against the production model on the same holdout. {html.escape(direction)}.</p>
      <div class="legend">
        <span><i class="swatch" style="background:#c8bfb0"></i>Production baseline</span>
        <span><i class="swatch" style="background:#0f5c56"></i>Promoted candidate</span>
        <span><i class="swatch" style="background:#8d3f2f"></i>Rejected candidate</span>
      </div>
      <div class="chart-scroll">{chart}</div>
      <p class="sub">{chart_note}</p>
    </section>
    <div class="split">
      <section class="panel">
        <h2>Versions</h2>
        <p class="sub">Every trained model is kept, including the ones that were not promoted.</p>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>Version</th><th>Status</th><th>Trained</th><th>{metric_label}</th><th>Baseline</th><th>Delta</th></tr>
            </thead>
            <tbody>
              {version_rows}
            </tbody>
          </table>
        </div>
      </section>
      <section class="panel">
        <h2>Checks</h2>
        <p class="sub">Scheduled and manual retraining checks.</p>
        <div class="table-scroll">
          <table>
            <thead>
              <tr><th>When</th><th>Trigger</th><th>Status</th><th>New</th></tr>
            </thead>
            <tbody>
              {run_rows}
            </tbody>
          </table>
        </div>
      </section>
    </div>
    <footer>
      A candidate replaces the production model only when {metric_label} is strictly better on the shared holdout.
      An equal score keeps the current model.
    </footer>
  </div>
</body>
</html>
"""


def _balance_text(counts: list[dict]) -> str:
    if not counts:
        return ""
    parts = [f"{item['count']} {item['label']}" for item in counts]
    return " · " + html.escape(", ".join(parts))


def _version_rows(versions: list[dict], higher_is_better: bool) -> str:
    if not versions:
        return '<tr><td colspan="6" class="empty">No models have been trained yet.</td></tr>'
    rows = []
    for item in versions:
        delta_text, delta_class = _format_delta(item.get("delta"), higher_is_better)
        baseline = item.get("baseline_primary_metric_value")
        baseline_text = "—" if baseline is None else f"{baseline:.4f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['version']))}</td>"
            f"<td><span class=\"status {html.escape(item['status'])}\">{html.escape(item['status'])}</span></td>"
            f"<td>{html.escape(_display_time(item.get('created_at')))}</td>"
            f"<td>{item['primary_metric_value']:.4f}</td>"
            f"<td>{baseline_text}</td>"
            f"<td class=\"{delta_class}\">{delta_text}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _run_rows(runs: list[dict]) -> str:
    if not runs:
        return '<tr><td colspan="4" class="empty">No checks have run yet.</td></tr>'
    rows = []
    for item in runs[:12]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(_display_time(item.get('started_at')))}</td>"
            f"<td>{html.escape(str(item.get('trigger') or ''))}</td>"
            f"<td><span class=\"status {html.escape(str(item.get('status') or ''))}\">{html.escape(str(item.get('status') or ''))}</span></td>"
            f"<td>{int(item.get('new_rows') or 0)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _format_delta(delta: float | None, higher_is_better: bool) -> tuple[str, str]:
    if delta is None:
        return "—", "flat"
    if delta == 0:
        return "0.0000", "flat"
    improved = delta > 0 if higher_is_better else delta < 0
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}", "up" if improved else "down"


def _display_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _chart(versions: list[dict], primary_metric: str) -> tuple[str, str]:
    if not versions:
        return '<p class="empty">The chart appears after the first model is trained.</p>', ""
    window = list(reversed(versions[:12]))
    note = ""
    if len(versions) > 12:
        note = f"Showing the latest 12 of {len(versions)} versions."
    values = [item["primary_metric_value"] for item in window]
    values.extend(
        item["baseline_primary_metric_value"]
        for item in window
        if item.get("baseline_primary_metric_value") is not None
    )
    peak = max(values) if values else 1.0
    y_max = 1.0 if peak <= 1 else peak * 1.1
    if y_max <= 0:
        y_max = 1.0

    group_width = 86
    left = 52
    width = left + group_width * len(window) + 16
    height = 248
    plot_bottom = 196
    plot_height = 168
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        f"<title>Comparison of {html.escape(primary_metric)} by model version</title>",
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        value = y_max * tick
        y = plot_bottom - (value / y_max) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 12}" y2="{y:.1f}" stroke="#e4dccb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5f584e">{value:.2f}</text>'
        )
    for index, item in enumerate(window):
        origin = left + 18 + index * group_width
        baseline = item.get("baseline_primary_metric_value")
        if baseline is None:
            parts.append(_bar(origin + 14, plot_bottom, plot_height, y_max, item["primary_metric_value"], _candidate_color(item)))
        else:
            parts.append(_bar(origin, plot_bottom, plot_height, y_max, baseline, "#c8bfb0"))
            parts.append(
                _bar(origin + 28, plot_bottom, plot_height, y_max, item["primary_metric_value"], _candidate_color(item))
            )
        label = html.escape(str(item["version"]))
        parts.append(
            f'<text x="{origin + 22}" y="{plot_bottom + 22}" text-anchor="middle" font-size="12" fill="#1c1915">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts), html.escape(note)


def _candidate_color(item: dict) -> str:
    if item.get("promoted") or item.get("status") == "production":
        return "#0f5c56"
    return "#8d3f2f"


def _bar(x: float, plot_bottom: float, plot_height: float, y_max: float, value: float, color: str) -> str:
    height = max(float(value), 0.0) / y_max * plot_height
    y = plot_bottom - height
    label_y = y - 6 if height > 16 else y - 6
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="22" height="{height:.1f}" fill="{color}"/>'
        f'<text x="{x + 11:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="10" fill="#5f584e">{value:.3f}</text>'
    )

"""Browser page with one tab per warehouse landmark."""

from __future__ import annotations

import json

from checkpoint_tabs.catalog import checkpoint_tabs


def render_page() -> str:
    """Return a self-contained page. Clicking a checkpoint asks the local server to drive there."""

    tabs = json.dumps(checkpoint_tabs())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jackal checkpoint tabs</title>
<style>
  body {{ margin: 0; font-family: sans-serif; background: #10140f; color: #e7efe4; }}
  header {{ padding: 20px 24px 8px; }}
  h1 {{ margin: 0 0 6px; font-size: 28px; }}
  p {{ margin: 0; color: #b7c4b4; line-height: 1.4; }}
  a {{ color: #d6e38a; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 16px 24px; }}
  button.tab, button.stop {{
    border: 1px solid #3d4a38; background: #1b2419; color: #e7efe4;
    border-radius: 8px; padding: 10px 14px; font-size: 16px; cursor: pointer;
  }}
  button.tab.active {{ background: #d6e38a; color: #17200f; border-color: #d6e38a; }}
  main {{ padding: 0 24px 24px; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
  button.checkpoint {{
    text-align: left; border: 1px solid #3d4a38; background: #182117; color: #e7efe4;
    border-radius: 12px; padding: 16px; cursor: pointer; font-size: 18px;
  }}
  button.checkpoint small {{ display: block; margin-top: 6px; color: #b7c4b4; font-size: 14px; }}
  button.checkpoint:disabled {{ opacity: 0.45; cursor: wait; }}
  #status {{ margin: 8px 24px 20px; padding: 14px 16px; background: #182117; border-radius: 12px; min-height: 48px; }}
</style>
</head>
<body>
<header>
  <h1>Choose a checkpoint</h1>
  <p>Each tab is one part of the warehouse. Click a checkpoint and the Jackal drives there from where it is now. The saved four-route record is <a href="/record">kept separately</a>.</p>
</header>
<nav id="tabs"></nav>
<div id="status">Waiting for a checkpoint.</div>
<main id="choices"></main>
<script>
const tabs = {tabs};
const tabRow = document.getElementById("tabs");
const choices = document.getElementById("choices");
const statusBox = document.getElementById("status");
let active = tabs[0].id;
let driving = false;

function draw() {{
  tabRow.replaceChildren();
  tabs.forEach((tab) => {{
    const button = document.createElement("button");
    button.className = "tab" + (tab.id === active ? " active" : "");
    button.textContent = tab.label;
    button.addEventListener("click", () => {{ active = tab.id; draw(); }});
    tabRow.appendChild(button);
  }});
  choices.replaceChildren();
  const group = tabs.find((tab) => tab.id === active);
  group.checkpoints.forEach((point) => {{
    const button = document.createElement("button");
    button.className = "checkpoint";
    button.dataset.checkpoint = point.id;
    button.disabled = driving;
    button.innerHTML = point.label + "<small>" + point.x.toFixed(2) + " m, " + point.y.toFixed(2) + " m</small>";
    button.addEventListener("click", () => go(point.id));
    choices.appendChild(button);
  }});
}}

function showStatus(payload) {{
  driving = payload.state === "driving";
  if (payload.state === "driving") {{
    statusBox.textContent = "Driving to " + payload.checkpoint_id + ".";
  }} else if (payload.state === "finished") {{
    const result = payload.result || {{}};
    const distance = result.goal_distance == null ? "" : " Final distance " + Number(result.goal_distance).toFixed(2) + " m.";
    statusBox.textContent = payload.checkpoint_id + " finished: " + (result.termination_reason || "done") + "." + distance;
  }} else if (payload.state === "error") {{
    statusBox.textContent = payload.error || "Navigation failed.";
  }} else {{
    statusBox.textContent = "Waiting for a checkpoint.";
  }}
  draw();
}}

async function go(id) {{
  const response = await fetch("/api/go", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{id}})
  }});
  showStatus(await response.json());
}}

async function poll() {{
  const response = await fetch("/api/status");
  showStatus(await response.json());
}}

draw();
setInterval(poll, 1000);
</script>
</body>
</html>
"""

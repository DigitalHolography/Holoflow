# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Build a local viewer for the native compiler's DOT artifact."""

import json
import shutil
import subprocess
import webbrowser


def write_graph_report(dot_path):
    dot_source = dot_path.read_bytes().decode("utf-8")
    # Escaping '<' keeps graph text from terminating the script element.
    script_source = json.dumps(dot_source).replace("<", r"\u003c")
    svg = dot_path.with_name("compiled_graph.svg")
    diagram = "<p>Graphviz rendering is unavailable. The original DOT is below.</p>"
    if dot := shutil.which("dot"):
        try:
            subprocess.run([dot, "-Tsvg", str(dot_path), "-o", str(svg)],
                           check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            diagram = '<div class="diagram"><img src="compiled_graph.svg" alt="Compiled execution graph"></div>'

    report = dot_path.with_name("compiled_graph.html")
    report.write_text(f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compiled Holoflow graph</title>
<style>
  body {{ font: 16px system-ui; margin: 2rem; color: #17212e; }}
  .actions {{ display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }}
  .diagram {{ overflow: auto; max-height: 70vh; border: 1px solid #b8c3d1; margin: 1rem 0; }}
  img {{ display: block; }}
  textarea {{ box-sizing: border-box; width: 100%; min-height: 16rem; font-family: monospace; }}
</style>
<h1>Compiled Holoflow graph</h1>
<div class="actions">
  <button id="copy" type="button">Copy DOT</button>
  <a href="compilation_success.dot" download>Download DOT</a>
  <span id="status" role="status"></span>
</div>
{diagram}
<label for="source">DOT source</label>
<textarea id="source" readonly spellcheck="false"></textarea>
<script>
const dotSource = {script_source};
const source = document.getElementById("source");
const status = document.getElementById("status");
source.value = dotSource;
document.getElementById("copy").addEventListener("click", async () => {{
  status.textContent = "";
  try {{
    await navigator.clipboard.writeText(dotSource);
    status.textContent = "DOT copied.";
    return;
  }} catch {{}}
  source.focus();
  source.select();
  let copied = false;
  try {{ copied = document.execCommand("copy"); }} catch {{}}
  status.textContent = copied ? "DOT copied." : "DOT selected. Use your clipboard shortcut to copy.";
}});
</script>
</html>
""", encoding="utf-8")
    return report


def show_graph_report(dot_path, open_browser):
    print(f"Compiled DOT: {dot_path}", flush=True)
    report = write_graph_report(dot_path)
    print(f"Graph report: {report}", flush=True)
    if open_browser:
        try:
            opened = webbrowser.open(report.as_uri())
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            print("Open the graph report path in your browser to view it.", flush=True)

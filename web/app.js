import { MuseumRenderer } from "./renderer.js";
import { drawLearningChart, normalizeRuns } from "./charts.js";
import { MapEditor, DEFAULT_GRID } from "./editor.js";
const $ = (id) => document.getElementById(id);
const renderer = new MuseumRenderer($("museum-canvas"));
const editor = new MapEditor($("editor-board"), $("editor-tools"));
let replays = [],
  runs = [],
  replay = null,
  index = 0,
  playing = false,
  lastAdvance = 0,
  speed = 1,
  compareRenderers = [],
  comparisonReplays = [],
  requestVersion = 0;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
async function request(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) {
    let message = `Request failed (${r.status})`;
    try {
      const data = await r.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail || data);
    } catch {}
    throw new Error(message);
  }
  return r.json();
}
async function loadData(api, file) {
  try {
    return await request(api);
  } catch {
    return request(file);
  }
}
function notify(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => ($("toast").hidden = true), 6000);
}
function kindLabel(kind = "") {
  return /script|teacher/i.test(kind)
    ? "Scripted guide"
    : /random/i.test(kind)
      ? "Random policy"
      : /bc|imitation/i.test(kind)
        ? "Imitation"
        : "Transformer";
}
function frameAt(r, i) {
  return r.frames[Math.min(i, r.frames.length - 1)];
}
function setFrame(next, animate = true) {
  if (!replay?.frames?.length) return;
  index = Math.min(Math.max(0, next), replay.frames.length - 1);
  const state = frameAt(replay, index);
  renderer.setState(state, animate);
  $("timeline").value = index;
  $("step-counter").textContent =
    `${String(index).padStart(3, "0")} / ${String(replay.frames.length - 1).padStart(3, "0")}`;
  $("inventory").textContent = state.inventory?.banana
    ? "Safely pocketed"
    : "Not yet";
  $("outcome").textContent =
    {
      running: "In progress",
      escaped: "Clean getaway",
      caught: "Caught",
      timeout: "Out of time",
    }[state.status] ||
    state.status ||
    "In progress";
  $("scene-caption").textContent =
    state.status === "escaped"
      ? "A fine addition to the collection."
      : state.status === "caught"
        ? "An unexpected meeting with security."
        : state.status === "timeout"
          ? "The gallery is closing."
          : state.inventory?.banana
            ? "Now for the difficult part: leaving."
            : "A small world. Plenty to learn.";
  for (let i = 0; i < compareRenderers.length; i++)
    compareRenderers[i].setState(frameAt(comparisonReplays[i], index), animate);
}
function togglePlay(value = !playing) {
  playing = value;
  $("play-button").textContent = playing ? "Ⅱ" : "▶";
  $("play-button").setAttribute(
    "aria-label",
    playing ? "Pause replay" : "Play replay",
  );
  lastAdvance = performance.now();
}
async function loadReplay(id) {
  const version = ++requestVersion;
  togglePlay(false);
  try {
    const r = await loadData(
      `/api/replays/${encodeURIComponent(id)}`,
      `/artifacts/data/replays/${encodeURIComponent(id)}.json`,
    );
    if (version !== requestVersion) return;
    acceptReplay(r);
  } catch (e) {
    notify(e.message);
  }
}
function acceptReplay(r) {
  if (!r.frames?.length) throw new Error("This replay contains no frames.");
  replay = r;
  index = 0;
  $("stage-loading").hidden = true;
  $("timeline").max = r.frames.length - 1;
  $("policy-badge").textContent = kindLabel(r.kind);
  $("mode-label").textContent = "RECORDED EPISODE";
  $("room-label").textContent = String(r.stage || "THE BANANA COLLECTION")
    .replaceAll("_", " ")
    .toUpperCase();
  $("moves").textContent = r.steps ?? r.frames.length - 1;
  $("episode-note").textContent =
    `${kindLabel(r.kind)} · seed ${r.seed ?? "custom"}. Recorded simulator episode; ${r.success ? "successful extraction" : "an unsuccessful attempt"}.`;
  $("compare-button").disabled = replays.length < 2;
  setFrame(0, false);
  if (!reduced) togglePlay(true);
}
function formatNumber(value) {
  return Number.isFinite(Number(value))
    ? Intl.NumberFormat("en", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(Number(value))
    : "—";
}
function showRun(id) {
  const r = runs.find((x) => x.id === id) || runs[0];
  if (!r) {
    drawLearningChart($("learning-chart"), []);
    return;
  }
  drawLearningChart($("learning-chart"), r.curve);
  const last = r.curve.at(-1);
  const result = r.test || r.evaluation || r.final_eval || r.summary || {};
  const success = result.success_rate ?? r.success_rate ?? last?.success;
  $("metric-success").textContent =
    success == null ? "—" : `${(Number(success) * 100).toFixed(0)}%`;
  $("metric-steps").textContent = formatNumber(
    r.env_steps ?? r.total_steps ?? last?.step,
  );
  $("metric-parameters").textContent = formatNumber(
    r.parameters ?? r.parameter_count ?? r.config?.parameters,
  );
  $("metric-device").textContent = /cuda|3070|gpu/i.test(r.device || "")
    ? "RTX 3070 Ti"
    : r.device || "Local GPU";
  $("chart-source").textContent =
    `${r.label} · ${r.curve.length} recorded evaluation points`;
  $("chart-legend").textContent =
    "● Evaluation success · environment interactions";
  $("research-detail").textContent =
    r.notes ||
    r.description ||
    "These results describe this experiment and its evaluation distribution. They do not establish general intelligence or universal maze-solving ability.";
}
$("play-button").addEventListener("click", () => {
  if (replay && index === replay.frames.length - 1) setFrame(0, false);
  togglePlay();
});
$("restart-button").addEventListener("click", () => {
  setFrame(0, false);
  togglePlay(true);
});
$("timeline").addEventListener("input", (e) => {
  togglePlay(false);
  setFrame(Number(e.target.value), false);
});
$("speed").addEventListener("change", (e) => (speed = Number(e.target.value)));
$("checkpoint").addEventListener("change", (e) => {
  closeComparison();
  loadReplay(e.target.value);
});
$("run-select").addEventListener("change", (e) => showRun(e.target.value));
for (const [id, value] of [
  ["view-full", false],
  ["view-agent", true],
])
  $(id).addEventListener("click", () => {
    renderer.agentView = value;
    for (const key of ["view-full", "view-agent"]) {
      $(key).classList.toggle("active", key === id);
      $(key).setAttribute("aria-pressed", String(key === id));
    }
  });
function closeComparison() {
  $("comparison").hidden = true;
  compareRenderers.forEach((r) => r.destroy());
  compareRenderers = [];
  comparisonReplays = [];
}
$("close-comparison").addEventListener("click", closeComparison);
$("compare-button").addEventListener("click", async () => {
  if (!replay) return;
  const earlier =
    replays.find((r) => r.id !== replay.id && r.seed === replay.seed) ||
    replays.find((r) => r.id !== replay.id);
  if (!earlier) return;
  try {
    const old = await loadData(
      `/api/replays/${encodeURIComponent(earlier.id)}`,
      `/artifacts/data/replays/${encodeURIComponent(earlier.id)}.json`,
    );
    closeComparison();
    $("comparison").hidden = false;
    comparisonReplays = [old, replay];
    compareRenderers = [
      new MuseumRenderer($("before-canvas"), { decorations: false }),
      new MuseumRenderer($("after-canvas"), { decorations: false }),
    ];
    $("before-label").textContent = old.label || old.id;
    $("after-label").textContent = replay.label || replay.id;
    $("comparison-note").textContent =
      old.seed === replay.seed
        ? "Matched map seed. Both recordings advance one action at a time."
        : "Different map seeds. This is a visual comparison, not a controlled performance test.";
    setFrame(0, false);
    togglePlay(true);
    $("comparison").scrollIntoView({
      behavior: reduced ? "instant" : "smooth",
    });
  } catch (e) {
    notify(e.message);
  }
});
function editorMessage(message, error = false) {
  $("editor-status").textContent = message;
  $("editor-status").classList.toggle("error", error);
}
$("reset-map").addEventListener("click", () => {
  editor.reset();
  editorMessage("The original floor plan is back. Make it your own.");
});
async function validate() {
  const data = await request("/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ grid: editor.value }),
  });
  editorMessage(
    data.valid
      ? "Room checked. A geometric route exists; camera timing may still make it difficult."
      : (data.errors || ["This room needs a few changes."]).join(" "),
    !data.valid,
  );
  return data.valid;
}
$("validate-map").addEventListener("click", async () => {
  try {
    await validate();
  } catch (e) {
    editorMessage(
      `Room checks need the local Python server. ${e.message}`,
      true,
    );
  }
});
$("run-map").addEventListener("click", async () => {
  const button = $("run-map");
  button.disabled = true;
  editorMessage("Checking the room and inviting Joyce in…");
  try {
    if (!(await validate())) return;
    const r = await request("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grid: editor.value,
        seed: 42,
        stage: "camera",
        checkpoint: "champion",
      }),
    });
    closeComparison();
    acceptReplay(r);
    $("checkpoint").value = "";
    $("mode-label").textContent = "YOUR ROOM · NEW ATTEMPT";
    editorMessage(
      r.success
        ? "Joyce escaped with the banana. A worthy collection."
        : "Joyce did not escape this time. New rooms can be difficult.",
    );
    $("museum").scrollIntoView({ behavior: reduced ? "instant" : "smooth" });
  } catch (e) {
    editorMessage(`Could not run this room: ${e.message}`, true);
  } finally {
    button.disabled = false;
  }
});
async function init() {
  renderer.setState({
    grid: DEFAULT_GRID,
    agent: { x: 1, y: 1, dir: 0 },
    inventory: {},
    visible: [],
    step: 0,
    status: "running",
  });
  const results = await Promise.allSettled([
    loadData("/api/replays", "/artifacts/data/replays/index.json"),
    loadData("/api/experiments", "/artifacts/data/experiments.json"),
  ]);
  if (results[0].status === "fulfilled") {
    const data = results[0].value;
    replays = Array.isArray(data) ? data : data.replays || [];
    $("checkpoint").replaceChildren();
    for (const r of replays) {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = r.label || `${kindLabel(r.kind)} · ${r.seed}`;
      $("checkpoint").append(o);
    }
    if (replays.length) {
      const preferred =
        replays.find((r) => /champion|ppo/i.test(r.id) && r.success) ||
        replays.find((r) => r.success) ||
        replays[0];
      $("checkpoint").value = preferred.id;
      await loadReplay(preferred.id);
    }
  }
  if (!replays.length) {
    $("stage-loading").firstChild.textContent = "The gallery is ready.";
    $("stage-loading").querySelector("span").textContent =
      "Run an experiment to give Joyce a first attempt.";
    $("policy-badge").textContent = "No model";
    $("checkpoint").innerHTML = "<option>No recorded attempts</option>";
  }
  $("play-button").disabled = !replays.length;
  $("restart-button").disabled = !replays.length;
  $("compare-button").disabled = replays.length < 2;
  if (results[1].status === "fulfilled") {
    runs = normalizeRuns(results[1].value);
    $("run-select").replaceChildren();
    for (const r of runs) {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = r.label;
      $("run-select").append(o);
    }
    showRun(runs.at(-1)?.id);
    if (runs.length) $("run-select").value = runs.at(-1).id;
  } else drawLearningChart($("learning-chart"), []);
}
function animate(t) {
  if (playing && replay && t - lastAdvance > 420 / speed) {
    if (index < replay.frames.length - 1) setFrame(index + 1);
    else togglePlay(false);
    lastAdvance = t;
  }
  renderer.draw(t);
  compareRenderers.forEach((r) => r.draw(t));
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
init();

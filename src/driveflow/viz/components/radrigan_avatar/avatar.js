// Ported from the prototype's script (setState, wave, beat, visibility pause). Python sends
// {state, label, progress, compact, beat, name, tagline, images: {base, hand} | null}.
const CLASS = { idle: "idle", listening: "listening", thinking: "thinking", using_tool: "tool", extracting: "extracting",
  awaiting_confirmation: "awaiting", confirmed: "confirmed", running: "running", results_ready: "results", error: "error", unavailable: "unavailable" };
const LABELS = { idle: "Ready", listening: "Listening…", thinking: "Thinking…", using_tool: "Working…", extracting: "Reading the datasheet…",
  awaiting_confirmation: "Waiting for your confirmation", confirmed: "Launched", running: "Simulation running", results_ready: "Results ready (simulated)",
  error: "Error", unavailable: "Assistant unavailable: local model not running" };
const SILHOUETTE = '<svg class="rad-silhouette" viewBox="0 0 100 120" aria-hidden="true"><circle cx="50" cy="32" r="22" fill="#2a2640"/>' +
  '<path d="M8 120c0-30 18-50 42-50s42 20 42 50z" fill="#2a2640"/></svg>';

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector(".rad-root");
  if (!root || !data) return;
  const d = data;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const figure = root.querySelector(".rad-figure");
  const hand = root.querySelector(".rad-hand");
  // images: set once (the data URIs are resent on every rerun, but the DOM keeps them)
  if (d.images && d.images.base) {
    const base = root.querySelector(".rad-base");
    if (base.dataset.src !== d.images.digest) { base.src = d.images.base; hand.src = d.images.hand; base.dataset.src = d.images.digest; }
    root.classList.remove("no-assets");
  } else if (!root.classList.contains("no-assets")) {
    root.classList.add("no-assets");
    figure.insertAdjacentHTML("afterbegin", SILHOUETTE);
  }
  root.querySelector(".rad-name").textContent = d.name || "SPADE assistant";
  root.querySelector(".rad-tag").textContent = d.tagline || "AI assistant · local model";
  const prev = root.dataset.state;
  root.className = "rad-root rad-dark s-" + (CLASS[d.state] || "idle") + (d.compact ? " compact" : "") + (root.classList.contains("no-assets") ? " no-assets" : "")
    + (d.speaking ? " speaking" : "");
  root.dataset.state = d.state;
  root.querySelector(".rad-status-text").textContent = d.label || LABELS[d.state] || d.state;
  root.querySelector(".rad-bar").style.width = d.state === "running" && d.progress != null ? Math.round(d.progress * 100) + "%" : "0";
  // one wave when a reply starts streaming; one beat per increment of d.beat (streamed words)
  if (!reduced && d.wave && root.dataset.wave !== String(d.wave)) { hand.classList.remove("wave"); void hand.offsetWidth; hand.classList.add("wave"); }
  root.dataset.wave = String(d.wave || 0);
  if (!reduced && d.beat && root.dataset.beat !== String(d.beat)) { figure.classList.remove("beat"); void figure.offsetWidth; figure.classList.add("beat"); }
  root.dataset.beat = String(d.beat || 0);
  // pause animations while the tab is hidden
  if (!root.dataset.vis) {
    root.dataset.vis = "1";
    document.addEventListener("visibilitychange", () => root.classList.toggle("paused", document.hidden));
  }
  if (setStateValue && prev !== d.state) {
    setStateValue("env", { reduced_motion: reduced, state: d.state, has_assets: !root.classList.contains("no-assets") });
  }
}

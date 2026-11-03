// Contribute page: consent, record, upload, validate. Talks only to Sanchaya's own API.
const CONSENT_VERSION = "draft-1";
const LICENCE = "CC0-1.0";
// No ?api= override on purpose: a link must not be able to send recordings to another server.
const API = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://localhost:8000" : "https://dani-api.sanchaya.net";

const $ = (id) => document.getElementById(id);
const REASONS = {
  short: "Too short: speak for at least 1 second.",
  long: "Too long: keep it under 60 seconds.",
  quiet: "Too quiet: move closer to the microphone.",
  clipped: "Too loud or distorted: move back a little.",
};
let token = null, tokenOk = false;
try { token = localStorage.getItem("kn_token"); } catch (e) { /* storage blocked: consent is asked each visit */ }

function say(text, err) { $("msg").textContent = text; $("msg").className = err ? "err" : "mute"; }

async function api(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers, token ? { Authorization: "Bearer " + token } : {});
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    if (r.status === 401) { token = null; try { localStorage.removeItem("kn_token"); } catch (e) {} show(false); }
    throw new Error((await r.text()).replace(/<[^>]*>/g, " ").trim().slice(0, 160) || r.statusText);
  }
  return r;
}

function show(signedIn) { $("consent").hidden = signedIn; $("app").hidden = !signedIn; if (signedIn) { nextPrompt(); stats(); } }

// ── consent ──
const gate = () => { $("join").disabled = !($("adult").checked && $("agree").checked); };
$("adult").onchange = $("agree").onchange = gate;
$("join").onclick = async () => {
  try {
    const r = await fetch(API + "/v1/consent", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adult: true, version: CONSENT_VERSION, licence: LICENCE }) });
    if (!r.ok) throw new Error("Could not register consent");
    token = (await r.json()).token;
    try { localStorage.setItem("kn_token", token); } catch (e) {}
    show(true);
  } catch (e) { say(e.message, true); }
};

// ── tabs ──
function tab(name) {
  $("speak").hidden = name !== "speak"; $("check").hidden = name !== "check";
  $("tab-speak").classList.toggle("on", name === "speak"); $("tab-check").classList.toggle("on", name === "check");
  say("");
  if (name === "check") nextClip();
}
$("tab-speak").onclick = () => tab("speak");
$("tab-check").onclick = () => tab("check");

// ── speak ──
let promptId = null, wavBuf = null, rec = null;

async function nextPrompt() {
  try {
    const p = await (await api("/v1/prompts/next?task=read")).json();
    promptId = p.id; $("prompt").textContent = p.text;
  } catch (e) { $("prompt").textContent = "…"; say(e.message, true); }
}
$("skip").onclick = nextPrompt;

function resetPreview() { $("preview").hidden = true; $("sendrow").hidden = true; wavBuf = null; $("level").style.width = "0"; }

$("rec").onclick = async () => {
  if (rec) return stopRec();
  resetPreview(); say("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
    const ctx = new AudioContext(), src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => {
      const d = new Float32Array(e.inputBuffer.getChannelData(0)); chunks.push(d);
      $("level").style.width = Math.min(100, Math.max(...d.map(Math.abs)) * 100) + "%";
    };
    src.connect(proc); proc.connect(ctx.destination);
    rec = { stream, ctx, src, proc, chunks };
    $("rec").textContent = "ನಿಲ್ಲಿಸಿ · Stop"; $("skip").disabled = true;
  } catch (e) { say("Microphone not available: " + e.message, true); }
};

function stopRec() {
  const { stream, ctx, src, proc, chunks } = rec; rec = null;
  src.disconnect(); proc.disconnect(); stream.getTracks().forEach((t) => t.stop());
  const rate = ctx.sampleRate; ctx.close();
  $("rec").textContent = "ರೆಕಾರ್ಡ್ ಮಾಡಿ · Record"; $("skip").disabled = false; $("level").style.width = "0";
  const all = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
  const pcm = KNWav.resample(all, rate), bad = KNWav.check(pcm);
  if (bad) return say(REASONS[bad], true);
  wavBuf = KNWav.encode(pcm);
  $("preview").src = URL.createObjectURL(new Blob([wavBuf], { type: "audio/wav" }));
  $("preview").hidden = false; $("sendrow").hidden = false;
}

$("redo").onclick = resetPreview;
$("send").onclick = async () => {
  const fd = new FormData();
  fd.append("audio", new Blob([wavBuf], { type: "audio/wav" }), "clip.wav");
  fd.append("prompt_id", promptId); fd.append("district", $("district").value); fd.append("dialect", $("dialect").value);
  // The prompt text is the transcript for read-aloud tasks, so it is not sent separately.
  $("send").disabled = true;
  try { await api("/v1/clips", { method: "POST", body: fd }); say("Thank you! / ಧನ್ಯವಾದಗಳು!"); resetPreview(); nextPrompt(); stats(); }
  catch (e) { say(e.message, true); }
  $("send").disabled = false;
};

// ── validate ──
let clipId = null;
async function nextClip() {
  clipId = null; $("vtext").textContent = "…"; $("vaudio").removeAttribute("src");
  try {
    const c = await (await api("/v1/validate/next")).json();
    clipId = c.id; $("vtext").textContent = c.text;
    const blob = await (await api(`/v1/clips/${c.id}/audio`)).blob();
    $("vaudio").src = URL.createObjectURL(blob);
  } catch (e) { $("vtext").textContent = "—"; say(e.message, true); }
}
document.querySelectorAll("[data-vote]").forEach((b) => b.onclick = async () => {
  if (!clipId) return;
  try { await api("/v1/validate", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip_id: clipId, vote: b.dataset.vote }) }); say(""); nextClip(); stats(); }
  catch (e) { say(e.message, true); }
});

// ── misc ──
async function stats() {
  try { const s = await (await api("/v1/stats")).json();
    $("stats").textContent = `${s.clips} clips so far · ${s.hours.accepted} h validated`; } catch (e) {}
}
$("forget").onclick = async (ev) => {
  ev.preventDefault();
  if (!confirm("Delete all your recordings and votes? This cannot be undone.")) return;
  try { await api("/v1/contributions", { method: "DELETE" });
    token = null; try { localStorage.removeItem("kn_token"); } catch (e) {}
    show(false); say("Your contributions were deleted."); }
  catch (e) { say(e.message, true); }
};

show(Boolean(token));

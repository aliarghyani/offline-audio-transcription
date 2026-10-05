"use strict";
let state, index = 0, busy = false, stopAt = null;
const el = id => document.getElementById(id);
const labels = {UNREVIEWED:"Unreviewed",USE_A:"Use original A",USE_B:"Use retry B",UNCLEAR:"Unclear"};
const mergeLabels = {SAFE:"Safe automatic merge",AUTO_TRIMMED:"Auto-trimmed to target",
  MANUAL_BOUNDARY_REQUIRED:"Manual boundary review required",UNUSABLE:"Unusable"};
const issueLabels = {
  B_TARGET_EMPTY:"B has no usable target text.",
  A_SEGMENT_CROSSES_TARGET:"An original segment crosses the replacement boundary.",
  A_WORD_CROSSES_TARGET:"An original word crosses the replacement boundary.",
  B_WORD_CROSSES_TARGET:"A retry word crosses the replacement boundary.",
  B_UNTIMED_SEGMENT_CROSSES_TARGET:"A retry segment crosses the boundary without word timing.",
  B_AMBIGUOUS_WORD_TIMING:"A target word has ambiguous timing.",
  B_OVERLAPPING_WORD_TIMING:"Target word timestamps overlap."
};
function time(seconds) {
  const h = Math.floor(seconds/3600), m = Math.floor(seconds%3600/60), s = (seconds%60).toFixed(3).padStart(6,"0");
  return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${s}`;
}
function message(text, error=false) { el("message").textContent=text; el("message").classList.toggle("error",error); }
function diagnostics(segments) {
  return segments.map(s => {
    const probs=(s.words||[]).map(w=>w.probability).filter(Number.isFinite);
    return {start:s.start,end:s.end,temperature:s.temperature??"UNKNOWN",avg_logprob:s.avg_logprob??"UNKNOWN",
      no_speech_prob:s.no_speech_prob??"UNKNOWN",compression_ratio:s.compression_ratio??"UNKNOWN",
      word_count:(s.words||[]).length,word_probability_mean:probs.length?probs.reduce((a,b)=>a+b,0)/probs.length:"UNKNOWN",
      word_probability_min:probs.length?Math.min(...probs):"UNKNOWN"};
  });
}
function updateDecisions() {
  el("window").disabled=busy;
  const entries=state.bundle.entries, counts={UNREVIEWED:0,USE_A:0,USE_B:0,UNCLEAR:0};
  for(const value of Object.values(state.decisions.decisions)) counts[value.choice]++;
  el("progress").textContent=`${entries.length-counts.UNREVIEWED} / ${entries.length} ${state.bundle.read_only?"policy decisions":"reviewed"} · A ${counts.USE_A} · B ${counts.USE_B} · Unclear ${counts.UNCLEAR}`;
  const entry=entries[index], choice=state.decisions.decisions[entry.review_id].choice;
  el("current-decision").textContent=`Current decision: ${labels[choice]}`;
  for(const button of document.querySelectorAll("[data-choice]")) {
    button.setAttribute("aria-pressed",String(button.dataset.choice===choice));
    button.disabled=busy || state.bundle.read_only || (button.dataset.choice==="USE_B" && !entry.b_selectable);
  }
  for(let i=0;i<entries.length;i++) el("window").options[i].textContent=`${i+1} · ${time(entries[i].target_interval[0])} · ${labels[state.decisions.decisions[entries[i].review_id].choice]}`;
}
function render() {
  const entry=state.bundle.entries[index];
  el("audio").pause(); stopAt=null;
  el("audio").src=entry.audio_url;
  el("interval").textContent=`${time(entry.target_interval[0])} — ${time(entry.target_interval[1])} (${entry.target_interval.map(s=>s.toFixed(3)).join(" – ")} seconds)`;
  el("flags").textContent=entry.quality_flags.join(" · ");
  el("audio-context").textContent=`Listening context: ${time(entry.decoded_interval[0])} — ${time(entry.decoded_interval[1])}. Audio begins before the target.`;
  el("text-a").textContent=entry.text_a || "[No original text in this target]";
  el("text-b").textContent=entry.text_b || "[No retry text assigned to this target]";
  el("candidate-source").textContent=entry.candidate_source==="RECOVERED_FROM_CONTEXT"
    ? "Recovered from retry context" : entry.candidate_source || "";
  el("review-mode").textContent=state.bundle.read_only
    ? `${state.bundle.review_mode}: automatic policy evidence, not human verified. This view is read-only.`
    : "Original A and retry B. Your choice is saved locally.";
  el("decision-heading").textContent=state.bundle.read_only
    ? "Automatic policy decision — not human verified" : "Human candidate quality — your decision";
  if(entry.effective_interval) el("interval").textContent+=` · Effective whole-word span: ${entry.effective_interval.map(time).join(" — ")}`;
  el("context-b").textContent=entry.candidate_b.raw_segments.map(s=>s.text).join("\n");
  el("diagnostics-a").textContent=JSON.stringify(diagnostics(entry.candidate_a.segments),null,2);
  el("diagnostics-b").textContent=JSON.stringify({target_segments:diagnostics(entry.candidate_b.segments),retry:entry.candidate_b.diagnostics},null,2);
  el("merge-status").textContent=mergeLabels[entry.merge_status];
  const reasons=entry.merge_issues.map(s=>issueLabels[s]||s).join(" ");
  el("boundary-note").textContent=entry.merge_status==="MANUAL_BOUNDARY_REQUIRED"
    ? reasons+" Candidate B can be selected, but this interval requires boundary resolution before final merge. This does not mean the spoken content is unclear."
    : entry.merge_status==="UNUSABLE"?reasons+" B has no usable target text and cannot be selected."
    : entry.merge_status==="AUTO_TRIMMED"?"Only words isolated by target timestamps will be inserted. Padded context will not be merged.":"";
  el("previous").disabled=index===0; el("next").disabled=index===state.bundle.entries.length-1;
  el("window").value=String(index); updateDecisions();
  try {localStorage.setItem(`review-position:${state.bundle.review_fingerprint}`,String(index));} catch {}
}
async function load() {
  const response=await fetch("/api/state",{cache:"no-store"});
  state=await response.json(); if(!response.ok) throw new Error(state.error||"Review could not load");
  el("window").replaceChildren();
  state.bundle.entries.forEach((entry,i)=>{const option=document.createElement("option");option.value=String(i);el("window").append(option);});
  try {index=Number(localStorage.getItem(`review-position:${state.bundle.review_fingerprint}`))||0;} catch {}
  index=Math.max(0,Math.min(state.bundle.entries.length-1,index)); render(); message("Loaded. Decisions are saved on this computer.");
}
async function choose(choice) {
  if(busy || !state || state.bundle.read_only) return;
  const entry=state.bundle.entries[index];
  if(choice==="USE_B" && !entry.b_selectable) {message("B has no usable target text and cannot be selected.",true);return;}
  busy=true;updateDecisions();message("Saving decision…");
  try {
    const response=await fetch("/api/decision",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":state.save_token},
      body:JSON.stringify({review_id:entry.review_id,choice,expected_checksum:state.decisions.artifact_checksum})});
    const result=await response.json();
    if(!response.ok) {
      if(response.status===409){await load();throw new Error("Another tab changed decisions. Reloaded; choose again.");}
      throw new Error(result.error||"Save failed. Your previous decision remains.");
    }
    state.decisions=result.decisions;message(`Saved: ${labels[choice]}.`+
      (choice==="USE_B"&&entry.merge_status==="MANUAL_BOUNDARY_REQUIRED"?" Boundary resolution is still required before merge.":""));
  } catch(error) {message(error.message,true);} finally {busy=false;updateDecisions();}
}
function navigate(delta) {if(!busy&&state){index=Math.max(0,Math.min(state.bundle.entries.length-1,index+delta));render();message("Loaded window.");}}
el("previous").addEventListener("click",()=>navigate(-1));el("next").addEventListener("click",()=>navigate(1));
el("window").addEventListener("change",()=>{if(!busy){index=Number(el("window").value);render();}});
for(const button of document.querySelectorAll("[data-choice]")) button.addEventListener("click",()=>choose(button.dataset.choice));
el("play-target").addEventListener("click",()=>{
  if(!state)return; const e=state.bundle.entries[index];
  const span=e.effective_interval||e.target_interval;
  el("audio").currentTime=span[0]-e.decoded_interval[0];
  stopAt=span[1]-e.decoded_interval[0];
  el("audio").play().catch(()=>message("Playback could not start; use the audio controls.",true));
});
el("audio").addEventListener("timeupdate",()=>{if(stopAt!==null&&el("audio").currentTime>=stopAt){el("audio").pause();stopAt=null;}});
el("audio").addEventListener("error",()=>message("Audio could not load. Check the local server and review clip.",true));
document.addEventListener("keydown",event=>{
  if(event.ctrlKey||event.altKey||event.metaKey||event.repeat||["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName))return;
  const key=event.key.toLowerCase(), shortcuts={a:"USE_A",b:"USE_B",u:"UNCLEAR"};
  if(shortcuts[key]){event.preventDefault();choose(shortcuts[key]);}
  else if(key==="n"||key==="p"){event.preventDefault();navigate(key==="n"?1:-1);}
});
load().catch(error=>message(error.message,true));

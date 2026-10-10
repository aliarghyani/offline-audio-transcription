"use strict";
let state;
const labels = ["SPEAKER_00", "SPEAKER_01"];
const choices = {};
const status = document.getElementById("status");
function eligible() {
  document.getElementById("save").disabled = !(
    choices.SPEAKER_00 && choices.SPEAKER_01 &&
    choices.SPEAKER_00 !== choices.SPEAKER_01 &&
    document.getElementById("confirmed").checked
  );
}
function time(seconds) {
  const s = Math.floor(seconds);
  return `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
async function load() {
  const response = await fetch("/api/state");
  if (!response.ok) throw new Error("بارگذاری اطلاعات محلی ناموفق بود.");
  state = await response.json();
  document.getElementById("speakers").replaceChildren();
  document.getElementById("confirmed").checked = false;
  for (const speaker of labels) {
    choices[speaker] = state.mapping?.mapping[speaker] || "";
    const card = document.createElement("section");
    const title = document.createElement("h2");
    title.dir = "ltr";
    title.textContent = speaker;
    card.append(title);
    for (const [index, sample] of state.bundle.samples.filter(s => s.speaker === speaker).entries()) {
      const row = document.createElement("div");
      const caption = document.createElement("p");
      caption.textContent = `نمونهٔ ${index + 1} · `;
      const interval = document.createElement("bdi");
      interval.dir = "ltr";
      interval.textContent = `${time(sample.start)} — ${time(sample.end)}`;
      caption.append(interval);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `/audio/${sample.sample_id}`;
      audio.setAttribute("aria-label", `${speaker} نمونهٔ ${index + 1}`);
      audio.addEventListener("play", () => {
        for (const other of document.querySelectorAll("audio")) if (other !== audio) other.pause();
      });
      row.append(caption, audio);
      card.append(row);
    }
    const label = document.createElement("label");
    label.textContent = "هویت این گوینده ";
    const select = document.createElement("select");
    select.id = `choice-${speaker}`;
    for (const [value, text] of [["", "انتخاب کنید"], ["علی", "علی"], ["مشاور", "مشاور"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      select.append(option);
    }
    select.value = choices[speaker];
    select.addEventListener("change", () => {
      choices[speaker] = select.value;
      document.getElementById("confirmed").checked = false;
      eligible();
    });
    label.append(select);
    card.append(label);
    document.getElementById("speakers").append(card);
  }
  status.textContent = state.mapping ? "هویت‌های تأییدشده بارگذاری شدند." : "هویت هیچ گوینده‌ای هنوز انتخاب نشده است.";
  eligible();
}
document.getElementById("confirmed").addEventListener("change", eligible);
document.getElementById("mapping-form").addEventListener("submit", async event => {
  event.preventDefault();
  document.getElementById("save").disabled = true;
  try {
    const response = await fetch("/api/mapping", {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Review-Token": state.save_token},
      body: JSON.stringify({mapping: choices, expected_checksum: state.mapping?.artifact_checksum || null,
        human_confirmed: document.getElementById("confirmed").checked})
    });
    if (response.status === 409) {
      await load();
      status.textContent = "انتخاب‌ها در پنجره‌ای دیگر تغییر کرده‌اند. دوباره بررسی و تأیید کنید.";
      return;
    }
    if (!response.ok) throw new Error("ثبت ناموفق بود؛ دو نام متفاوت و تأیید انسانی لازم است.");
    state.mapping = (await response.json()).mapping;
    status.textContent = "هویت گویندگان ثبت شد. متن نهایی تولید نشده است.";
    document.getElementById("confirmed").checked = false;
  } catch (error) { status.textContent = error.message; }
  eligible();
});
load().catch(error => {status.textContent = error.message;});

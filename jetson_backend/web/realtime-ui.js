(function bootRealtimeUi() {
  "use strict";

  const parser = window.JetsonRealtimeParser;
  const SAMPLE_LINES = [
    "[12:00:00] 예측: construction (76.4%) | status=감지 | level=-22.4 dBFS | enhanced=-15.1 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=76.4%, gunshot=2.1%, alarm_siren=8.4%, horn=1.2%, water=6.6%, knock=3.1%, appliances=1.0%, baby_cry=0.8%, animal_cry=1.5%, glass_shatter=2.5%",
    "[12:00:02] 예측: horn (84.8%) | status=감지 | level=-18.9 dBFS | enhanced=-12.0 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=3.3%, gunshot=2.0%, alarm_siren=16.2%, horn=84.8%, water=0.7%, knock=1.5%, appliances=0.8%, baby_cry=1.0%, animal_cry=4.1%, glass_shatter=1.7%",
    "[12:00:04] 예측: water (31.6%) | status=감지 | level=-24.8 dBFS | enhanced=-18.7 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=8.8%, gunshot=0.7%, alarm_siren=2.0%, horn=1.1%, water=31.6%, knock=2.6%, appliances=10.3%, baby_cry=2.2%, animal_cry=4.5%, glass_shatter=0.9%",
    "[12:00:06] 예측: glass_shatter (91.2%) | status=감지 | level=-16.3 dBFS | enhanced=-10.6 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=5.1%, gunshot=3.5%, alarm_siren=7.1%, horn=1.3%, water=0.8%, knock=4.0%, appliances=2.4%, baby_cry=0.9%, animal_cry=1.2%, glass_shatter=91.2%",
    "[12:00:08] 예측: animal_cry (3.0%) | status=낮음(점수낮음 3.0%<5.0%) | level=-35.7 dBFS | enhanced=-29.1 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=1.1%, gunshot=0.4%, alarm_siren=1.8%, horn=2.0%, water=3.5%, knock=1.0%, appliances=0.8%, baby_cry=2.2%, animal_cry=3.0%, glass_shatter=0.6%"
  ];

  const els = {
    body: document.body,
    sourceForm: document.getElementById("sourceForm"),
    sourceMode: document.getElementById("sourceMode"),
    sourceUrl: document.getElementById("sourceUrl"),
    fileInput: document.getElementById("fileInput"),
    sampleButton: document.getElementById("sampleButton"),
    connection: document.querySelector(".connection"),
    connectionText: document.getElementById("connectionText"),
    stateBadge: document.getElementById("stateBadge"),
    statusText: document.getElementById("statusText"),
    soundIcon: document.getElementById("soundIcon"),
    soundLabel: document.getElementById("soundLabel"),
    soundScore: document.getElementById("soundScore"),
    scoreMeterBar: document.getElementById("scoreMeterBar"),
    timeValue: document.getElementById("timeValue"),
    dbValue: document.getElementById("dbValue"),
    enhancedValue: document.getElementById("enhancedValue"),
    inferValue: document.getElementById("inferValue"),
    thresholdValue: document.getElementById("thresholdValue"),
    candidateList: document.getElementById("candidateList"),
    sourceValue: document.getElementById("sourceValue"),
    rawLine: document.getElementById("rawLine")
  };

  let eventSource = null;
  let pollTimer = null;
  let fileHandle = null;
  let sampleIndex = 0;

  function scoreThreshold() {
    return parser.DEFAULT_SCORE_THRESHOLD;
  }

  function thresholdLabel() {
    return parser.formatPercent(scoreThreshold());
  }

  function scoreModeLabel() {
    return "10개 정규화";
  }

  function setConnection(state, text) {
    els.connection.dataset.connection = state;
    els.connectionText.textContent = text;
  }

  function closeSources() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function parseMessage(raw) {
    if (!raw) return null;

    try {
      return parser.normalizeResult(JSON.parse(raw), { scoreThreshold: scoreThreshold() });
    } catch (_) {
      return parser.normalizeResult(raw, { scoreThreshold: scoreThreshold() });
    }
  }

  function clampPercent(score) {
    const value = Number(score);
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, value * 100));
  }

  function statusLabel(data) {
    if (data.display_state === "detected") return "감지";
    if (data.reason === "low_score") return "확신도 낮음";
    if (data.reason === "low_status") return data.status_text || "입력 낮음";
    return "소리를 기다리는 중";
  }

  function renderCandidates(items) {
    els.candidateList.textContent = "";

    if (!items || items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "표시할 후보 없음";
      els.candidateList.appendChild(empty);
      return;
    }

    items.slice(0, 10).forEach((item) => {
      const row = document.createElement("div");
      const name = document.createElement("div");
      const score = document.createElement("div");
      const bar = document.createElement("div");
      const fill = document.createElement("span");

      row.className = "candidate";
      name.className = "candidate-name";
      score.className = "candidate-score";
      bar.className = "candidate-bar";

      name.textContent = item.display_label || parser.normalizeLabel(item.label);
      score.textContent = parser.formatPercent(item.score);
      fill.style.width = `${clampPercent(item.score)}%`;

      bar.appendChild(fill);
      row.appendChild(name);
      row.appendChild(score);
      row.appendChild(bar);
      els.candidateList.appendChild(row);
    });
  }

  function render(data) {
    if (!data) return;

    const level = data.display_state === "detected" ? data.level || "info" : "listening";
    els.body.dataset.state = data.display_state || "listening";
    els.body.dataset.level = level;

    els.stateBadge.textContent = data.display_state === "detected" ? level.toUpperCase() : "LISTENING";
    els.statusText.textContent = statusLabel(data);
    els.soundIcon.textContent = data.display_icon || "〰️";
    els.soundLabel.textContent = data.display_label || "소리를 듣고 있어요";
    els.soundScore.textContent =
      data.display_score === null || data.display_score === undefined
        ? `${thresholdLabel()} 이상 감지되면 표시`
        : `${scoreModeLabel()} ${parser.formatPercent(data.display_score)}`;
    els.scoreMeterBar.style.width = `${clampPercent(data.display_score)}%`;

    els.timeValue.textContent = data.time || "-";
    els.dbValue.textContent = parser.formatDb(data.db, data.db_scale);
    els.enhancedValue.textContent = parser.formatDb(data.enhanced_db, "dbfs");
    els.inferValue.textContent =
      data.infer_sec === null || data.infer_sec === undefined ? "-" : `${Number(data.infer_sec).toFixed(3)}s`;
    els.thresholdValue.textContent = scoreModeLabel();
    els.sourceValue.textContent = data.source || "-";
    els.rawLine.textContent = data.display_raw || data.raw || "";

    renderCandidates(data.items || data.strong_items || []);
  }

  function handleRaw(raw) {
    const parsed = parseMessage(raw);
    if (parsed) {
      render(parsed);
      setConnection("connected", "연결됨");
    }
  }

  function findLatestInferenceLine(text) {
    const lines = String(text || "").split(/\r?\n/).filter(Boolean);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      if (parser.parseLine(lines[index], { scoreThreshold: scoreThreshold() })) {
        return lines[index];
      }
    }
    return "";
  }

  async function pollLogUrl(url) {
    try {
      const response = await fetch(url, {
        cache: "no-store",
        headers: {
          Range: "bytes=-65536"
        }
      });
      if (!response.ok && response.status !== 206) throw new Error(`HTTP ${response.status}`);
      const latestLine = findLatestInferenceLine(await response.text());
      if (latestLine) handleRaw(latestLine);
    } catch (error) {
      setConnection("error", error.message || "오류");
    }
  }

  async function pollFileHandle() {
    if (!fileHandle) return;
    try {
      const file = await fileHandle.getFile();
      const latestLine = findLatestInferenceLine(await file.text());
      if (latestLine) handleRaw(latestLine);
    } catch (error) {
      setConnection("error", error.message || "파일 오류");
    }
  }

  function connectSse(url) {
    closeSources();
    setConnection("connecting", "연결 중");
    eventSource = new EventSource(url);
    eventSource.onopen = () => setConnection("connected", "연결됨");
    eventSource.onmessage = (event) => handleRaw(event.data);
    eventSource.onerror = () => setConnection("error", "SSE 오류");
  }

  function connectLog(url) {
    closeSources();
    setConnection("connecting", "폴링 중");
    pollLogUrl(url);
    pollTimer = setInterval(() => pollLogUrl(url), 1000);
  }

  async function connectFile() {
    closeSources();
    setConnection("connecting", "파일");

    if ("showOpenFilePicker" in window) {
      const handles = await window.showOpenFilePicker({
        multiple: false,
        types: [
          {
            description: "Log files",
            accept: {
              "text/plain": [".txt", ".log"]
            }
          }
        ]
      });
      fileHandle = handles[0];
      pollFileHandle();
      pollTimer = setInterval(pollFileHandle, 1000);
      return;
    }

    els.fileInput.click();
  }

  function applyModeDefaults() {
    if (els.sourceMode.value === "sse" && (!els.sourceUrl.value || els.sourceUrl.value.endsWith(".log"))) {
      els.sourceUrl.value = "/events";
    }
    if (els.sourceMode.value === "log" && (!els.sourceUrl.value || els.sourceUrl.value === "/events")) {
      els.sourceUrl.value = "./realtime_inference.log";
    }
  }

  function connectFromForm() {
    const mode = els.sourceMode.value;
    const url = els.sourceUrl.value.trim();
    localStorage.setItem("jetson-audio-source-mode", mode);
    localStorage.setItem("jetson-audio-source-url", url);

    if (mode === "sse") {
      connectSse(url || "/events");
      return;
    }
    if (mode === "log") {
      connectLog(url || "./realtime_inference.log");
      return;
    }
    connectFile().catch((error) => setConnection("error", error.message || "파일 오류"));
  }

  function loadSettings() {
    const params = new URLSearchParams(window.location.search);
    const sourceParam = params.get("source");
    const urlParam = params.get("url") || params.get("events");
    const savedMode = localStorage.getItem("jetson-audio-source-mode");
    const savedUrl = localStorage.getItem("jetson-audio-source-url");

    els.sourceMode.value = sourceParam || savedMode || "sse";
    els.sourceUrl.value = urlParam || savedUrl || "/events";
    applyModeDefaults();
  }

  els.sourceMode.addEventListener("change", applyModeDefaults);
  els.sourceForm.addEventListener("submit", (event) => {
    event.preventDefault();
    connectFromForm();
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) return;

    closeSources();
    setConnection("connected", "파일");
    file.text().then((text) => {
      const latestLine = findLatestInferenceLine(text);
      if (latestLine) handleRaw(latestLine);
    });
  });

  els.sampleButton.addEventListener("click", () => {
    closeSources();
    const now = new Date().toTimeString().slice(0, 8);
    const line = SAMPLE_LINES[sampleIndex % SAMPLE_LINES.length].replace(/\[[0-9:]+\]/, `[${now}]`);
    sampleIndex += 1;
    setConnection("connected", "샘플");
    handleRaw(line);
  });

  loadSettings();
  render(parser.initialResult());
  connectFromForm();
})();

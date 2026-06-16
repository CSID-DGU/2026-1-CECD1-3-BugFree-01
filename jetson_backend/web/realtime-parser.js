(function attachParser(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.JetsonRealtimeParser = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildParser() {
  "use strict";

  const DEFAULT_SCORE_THRESHOLD = 0.05;
  const ANSI_RE = /\u001b\[[0-9;]*m/g;

  const KOREAN_LABELS = {
    construction: "공사장 소리",
    gunshot: "총소리",
    alarm_siren: "경보/사이렌 소리",
    "alarm siren": "경보/사이렌 소리",
    horn: "차량 경적 소리",
    water: "물소리",
    knock: "노크 소리",
    appliances: "가전제품 소리",
    baby_cry: "아기 울음소리",
    animal_cry: "동물 소리",
    "animal cry": "동물 소리",
    glass_shatter: "유리 깨지는 소리",
    "glass shatter": "유리 깨지는 소리",

    cat: "고양이 소리",
    meow: "고양이 소리",
    caterwaul: "고양이 울음소리",
    dog: "강아지 소리",
    bark: "강아지 짖는 소리",
    "crying baby": "아기 울음소리",
    "baby cry": "아기 울음소리",
    "infant cry": "아기 울음소리",
    cry: "울음소리",
    crying: "울음소리",
    scream: "비명 소리",
    screaming: "비명 소리",
    shout: "고함 소리",
    yell: "고함 소리",
    horn: "차량 경적 소리",
    "vehicle horn": "차량 경적 소리",
    "car horn": "차량 경적 소리",
    honking: "차량 경적 소리",
    "vehicle horn, car horn, honking": "차량 경적 소리",
    siren: "사이렌 소리",
    alarm: "경보 소리",
    "alarm clock": "알람 시계 소리",
    gunfire: "총소리",
    explosion: "폭발음",
    fire: "불소리",
    "crackling fire": "불타는 소리",
    jackhammer: "착암기 소리",
    drill: "드릴 소리",
    rain: "빗소리",
    raindrop: "빗방울 소리",
    "water tap": "수돗물 소리",
    pour: "물 따르는 소리",
    "pouring water": "물 따르는 소리",
    "door knock": "노크 소리",
    door: "문 소리",
    "door, wood creaks": "문 삐걱이는 소리",
    "vacuum cleaner": "청소기 소리",
    "washing machine": "세탁기 소리",
    glass: "유리 소리",
    shatter: "깨지는 소리",
    "glass breaking": "유리 깨지는 소리",
    breaking: "깨지는 소리",
    bicycle: "자전거 소리",
    "bicycle bell": "자전거 벨 소리",
    engine: "엔진 소리",
    vehicle: "차량 소리",
    car: "자동차 소리",
    truck: "트럭 소리",
    listening: "소리를 듣고 있어요"
  };

  const KEYWORD_LABELS = [
    [["vehicle horn", "car horn", "honking", "horn"], "차량 경적 소리"],
    [["siren"], "사이렌 소리"],
    [["alarm"], "경보 소리"],
    [["scream", "screaming"], "비명 소리"],
    [["shout", "yell"], "고함 소리"],
    [["gunshot", "gunfire", "gun"], "총소리"],
    [["explosion", "explosive"], "폭발음"],
    [["glass", "shatter", "breaking"], "유리 깨지는 소리"],
    [["fire", "smoke"], "화재/불소리"],
    [["cat", "meow", "caterwaul"], "고양이 소리"],
    [["dog", "bark"], "강아지 소리"],
    [["baby", "infant"], "아기 울음소리"],
    [["cry", "crying"], "울음소리"],
    [["water", "rain", "raindrop", "tap", "pour"], "물소리"],
    [["knock"], "노크 소리"],
    [["door"], "문 소리"],
    [["vacuum", "washing", "appliance"], "가전제품 소리"],
    [["construction", "jackhammer", "drill"], "공사장 소리"],
    [["bicycle"], "자전거 소리"],
    [["engine", "vehicle", "car", "truck"], "차량 소리"]
  ];

  const ICONS = [
    [["animal", "cat", "meow", "caterwaul"], "🐱"],
    [["dog", "bark"], "🐶"],
    [["baby", "infant"], "👶"],
    [["horn", "vehicle", "car"], "🚗"],
    [["siren", "alarm"], "⚠️"],
    [["gun", "gunshot", "explosion"], "🚨"],
    [["glass", "shatter"], "🔔"],
    [["water", "rain", "tap"], "💧"],
    [["knock", "door"], "🚪"],
    [["appliance", "vacuum", "washing"], "🔌"],
    [["construction", "drill", "jackhammer"], "🏗️"]
  ];

  const DANGER_WORDS = [
    "horn",
    "siren",
    "alarm",
    "scream",
    "gun",
    "gunshot",
    "explosion",
    "glass",
    "fire",
    "smoke",
    "crash",
    "shout"
  ];

  const CAUTION_WORDS = [
    "construction",
    "vehicle",
    "car",
    "truck",
    "dog",
    "cry",
    "baby",
    "knock",
    "door",
    "water",
    "engine",
    "cat",
    "animal",
    "appliances"
  ];

  function stripAnsi(text) {
    return String(text || "").replace(ANSI_RE, "");
  }

  function normalizeKey(label) {
    return String(label || "")
      .trim()
      .toLowerCase()
      .replace(/[_-]/g, " ")
      .replace(/\s+/g, " ");
  }

  function normalizeLabel(label) {
    const key = normalizeKey(label);
    if (!key) return "";
    if (KOREAN_LABELS[key]) return KOREAN_LABELS[key];

    for (const sep of [",", "/", "|"]) {
      if (key.includes(sep)) {
        const parts = key.split(sep).map((part) => part.trim()).filter(Boolean);
        for (const part of parts) {
          if (KOREAN_LABELS[part]) return KOREAN_LABELS[part];
        }
      }
    }

    for (const [words, korean] of KEYWORD_LABELS) {
      if (words.some((word) => key.includes(word))) return korean;
    }

    return "알 수 없는 소리";
  }

  function iconForLabel(label) {
    const key = normalizeKey(label);
    for (const [words, icon] of ICONS) {
      if (words.some((word) => key.includes(word))) return icon;
    }
    return "🔊";
  }

  function riskLevel(label) {
    const key = normalizeKey(label);
    if (DANGER_WORDS.some((word) => key.includes(word))) return "danger";
    if (CAUTION_WORDS.some((word) => key.includes(word))) return "caution";
    return "info";
  }

  function toNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function formatPercent(score) {
    const num = toNumber(score);
    if (num === null) return "-";
    return `${(num * 100).toFixed(1)}%`;
  }

  function formatDb(value, scale) {
    const num = toNumber(value);
    if (num === null) return "-";
    return `${num.toFixed(1)} ${scale === "dbfs" ? "dBFS" : "dB"}`;
  }

  function parseScoreItems(scoresText) {
    if (!scoresText) return [];

    return scoresText
      .split(/\s*,\s*/)
      .map((part) => {
        const match = part.trim().match(/^(.+?)=([-+]?[0-9]*\.?[0-9]+)%$/u);
        if (!match) return null;
        const label = match[1].trim();
        const score = Number(match[2]) / 100;
        return {
          label,
          display_label: normalizeLabel(label),
          score
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);
  }

  function parseBridgeItems(itemsText) {
    if (!itemsText) return [];

    const text = itemsText.split("| DOA", 1)[0].trim();
    return text
      .split(" / ")
      .map((part) => {
        const match = part.trim().match(/^(.+?)\s+([-+]?[0-9]*\.?[0-9]+)$/u);
        if (!match) return null;
        const label = match[1].trim();
        const score = Number(match[2]);
        return {
          label,
          display_label: normalizeLabel(label),
          score
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);
  }

  function normalizeDisplayItems(items) {
    if (!Array.isArray(items) || !items.length) return [];

    const rawItems = items.map((item) => {
      const rawScore = Number(item.score);
      return Object.assign({}, item, {
        raw_score: Number.isFinite(rawScore) ? rawScore : 0,
        display_label: item.display_label || normalizeLabel(item.label)
      });
    });
    const total = rawItems.reduce((sum, item) => sum + Math.max(0, item.raw_score), 0);
    if (!(total > 0)) {
      return rawItems.map((item) => Object.assign({}, item, { score: 0, normalized_score: 0 }));
    }

    return rawItems
      .map((item) => {
        const normalizedScore = Math.max(0, item.raw_score) / total;
        return Object.assign({}, item, {
          score: normalizedScore,
          normalized_score: normalizedScore
        });
      })
      .sort((a, b) => b.score - a.score);
  }

  function buildDisplayRaw(parsed, shownItems) {
    const chunks = [];
    chunks.push(`[${parsed.time || "--:--:--"}]`);
    if (parsed.infer_sec !== null && parsed.infer_sec !== undefined) {
      chunks.push(`추론=${Number(parsed.infer_sec).toFixed(3)}초`);
    }
    if (parsed.total_sec !== null && parsed.total_sec !== undefined) {
      chunks.push(`전체=${Number(parsed.total_sec).toFixed(3)}초`);
    }
    if (parsed.db !== null && parsed.db !== undefined) {
      chunks.push(`음량=${formatDb(parsed.db, parsed.db_scale)}`);
    }
    if (parsed.enhanced_db !== null && parsed.enhanced_db !== undefined) {
      chunks.push(`강조=${formatDb(parsed.enhanced_db, "dbfs")}`);
    }

    const visible = Array.isArray(shownItems) ? shownItems : [];
    if (visible.length) {
      chunks.push(
        visible
          .map((item) => `${item.display_label || normalizeLabel(item.label)} ${formatPercent(item.score)}`)
          .join(" / ")
      );
    } else {
      chunks.push("소리를 듣고 있어요");
    }

    if (parsed.status_text) chunks.push(`상태=${parsed.status_text}`);
    return chunks.join(" | ");
  }

  function statusMeansLowSignal(statusText) {
    const text = String(statusText || "").trim();
    if (!text) return false;
    if (text.startsWith("낮음")) return true;
    return /low|quiet|error/i.test(text);
  }

  function statusMeansDetected(statusText) {
    const text = String(statusText || "").trim();
    if (!text) return true;
    if (text.startsWith("감지")) return true;
    return /detected|ok/i.test(text);
  }

  function applyDisplayPolicy(input, options) {
    const threshold = toNumber(options && options.scoreThreshold);
    const scoreThreshold = threshold === null ? DEFAULT_SCORE_THRESHOLD : threshold;
    const parsed = Object.assign({}, input);
    const allItems = Array.isArray(parsed.items) ? parsed.items.slice() : [];

    for (const item of allItems) {
      item.display_label = item.display_label || normalizeLabel(item.label);
    }

    const topItem =
      allItems.find((item) => normalizeKey(item.label) === normalizeKey(parsed.label)) ||
      allItems[0] ||
      null;
    const topScore = topItem ? Number(topItem.score) : Number(parsed.score || 0);
    const topLabel = topItem ? topItem.label : parsed.label || "";
    const strongItems = allItems.filter((item) => Number(item.score) >= scoreThreshold);
    const displayItems = normalizeDisplayItems(allItems);
    const topDisplayItem =
      displayItems.find((item) => normalizeKey(item.label) === normalizeKey(topLabel)) ||
      displayItems[0] ||
      null;
    const displayScore = topDisplayItem ? Number(topDisplayItem.score) : Number(parsed.score || 0);
    const statusText = String(parsed.status_text || "").trim();
    const hasStatusText = statusText.length > 0;
    const detectedStatus = hasStatusText && statusMeansDetected(statusText);
    const lowStatus = statusMeansLowSignal(statusText) || (hasStatusText && !detectedStatus);
    const lowScore = !(topScore >= scoreThreshold);
    const forcedListening = normalizeKey(topLabel) === "listening";

    parsed.status = parsed.status || "ok";
    parsed.label = topLabel;
    parsed.score = Number.isFinite(topScore) ? topScore : 0;
    parsed.display_label = normalizeLabel(topLabel);
    parsed.display_icon = iconForLabel(topLabel);
    parsed.all_items = allItems;
    parsed.strong_items = strongItems;
    parsed.thresholds = Object.assign({}, parsed.thresholds || {}, {
      score: scoreThreshold
    });
    parsed.raw_inference = parsed.raw || "";
    parsed.display_score_mode = "10_class_normalized";

    if (lowStatus || (lowScore && !detectedStatus) || forcedListening) {
      parsed.display_state = "listening";
      parsed.display_label = "소리를 듣고 있어요";
      parsed.display_icon = "〰️";
      parsed.display_score = null;
      parsed.items = [];
      parsed.level = "listening";
      parsed.reason = lowStatus ? "low_status" : lowScore ? "low_score" : "listening";
      parsed.display_raw = buildDisplayRaw(parsed, []);
      return parsed;
    }

    parsed.display_state = "detected";
    parsed.display_score = Number.isFinite(displayScore) ? displayScore : parsed.score;
    parsed.items = displayItems.length ? displayItems.slice(0, 10) : strongItems.length ? strongItems : topItem ? [topItem] : [];
    parsed.level = riskLevel(topLabel);
    parsed.reason = "detected";
    parsed.display_raw = buildDisplayRaw(parsed, parsed.items);
    return parsed;
  }

  function parseRealtimeInferenceLine(line, options) {
    const raw = stripAnsi(line).trim();
    const match = raw.match(
      /^\[([0-9]{2}:[0-9]{2}:[0-9]{2})\]\s+예측:\s+(.+?)\s+\(([-+]?[0-9]*\.?[0-9]+)%\)\s+\|\s+(.+)$/u
    );
    if (!match) return null;

    const tail = match[4];
    const statusMatch = tail.match(/(?:^|\|\s*)status=([^|]+)/u);
    const levelMatch = tail.match(/(?:^|\|\s*)level=([-+]?[0-9]*\.?[0-9]+)\s*dBFS/u);
    const enhancedMatch = tail.match(/(?:^|\|\s*)enhanced=([-+]?[0-9]*\.?[0-9]+)\s*dBFS/u);
    const allMatch = tail.match(/(?:^|\|\s*)전체:\s*(.+)$/u);

    const topLabel = match[2].trim();
    const topScore = Number(match[3]) / 100;
    const items = parseScoreItems(allMatch ? allMatch[1] : `${topLabel}=${match[3]}%`);

    return applyDisplayPolicy(
      {
        status: "ok",
        source: "realtime_inference",
        time: match[1],
        label: topLabel,
        score: topScore,
        infer_sec: null,
        total_sec: null,
        db: levelMatch ? Number(levelMatch[1]) : null,
        db_scale: "dbfs",
        enhanced_db: enhancedMatch ? Number(enhancedMatch[1]) : null,
        status_text: statusMatch ? statusMatch[1].trim() : "",
        level: riskLevel(topLabel),
        items,
        raw
      },
      options || {}
    );
  }

  function parseNanoBridgeLine(line, options) {
    const raw = stripAnsi(line).trim();
    const match = raw.match(/^\[([0-9:]+)\]\s+infer=([0-9.]+)s(.*?)\|\s+(.*)$/u);
    if (!match) return null;

    const middle = match[3] || "";
    const totalMatch = middle.match(/total=([0-9.]+)s/u);
    const dbMatch = middle.match(/db=([-+]?[0-9]*\.?[0-9]+)\s*dB/iu);
    const items = parseBridgeItems(match[4]);
    if (!items.length) return null;

    return applyDisplayPolicy(
      {
        status: "ok",
        source: "jetson_wifi_bridge",
        time: match[1],
        label: items[0].label,
        score: items[0].score,
        infer_sec: Number(match[2]),
        total_sec: totalMatch ? Number(totalMatch[1]) : null,
        db: dbMatch ? Number(dbMatch[1]) : null,
        db_scale: "db",
        enhanced_db: null,
        status_text: "",
        level: riskLevel(items[0].label),
        items,
        raw
      },
      options || {}
    );
  }

  function parseLine(line, options) {
    return parseRealtimeInferenceLine(line, options) || parseNanoBridgeLine(line, options);
  }

  function initialResult() {
    return {
      status: "starting",
      source: "none",
      display_state: "listening",
      raw: "",
      time: "",
      label: "",
      display_label: "소리를 듣고 있어요",
      display_icon: "〰️",
      score: 0,
      display_score: null,
      infer_sec: null,
      total_sec: null,
      db: null,
      db_scale: "dbfs",
      enhanced_db: null,
      level: "listening",
      items: [],
      all_items: [],
      strong_items: [],
      status_text: "",
      thresholds: {
        score: DEFAULT_SCORE_THRESHOLD
      },
      reason: "starting"
    };
  }

  function normalizeResult(data, options) {
    if (data === null || data === undefined) return initialResult();

    if (typeof data === "string") {
      return parseLine(data, options);
    }

    if (typeof data !== "object") return null;

    if (data.display_state && data.display_label) {
      return Object.assign({}, initialResult(), data);
    }

    if (data.raw) {
      const parsed = parseLine(data.raw, options);
      if (parsed) return parsed;
    }

    return applyDisplayPolicy(
      Object.assign({}, initialResult(), data, {
        items: Array.isArray(data.items) ? data.items : []
      }),
      options || {}
    );
  }

  return {
    DEFAULT_SCORE_THRESHOLD,
    applyDisplayPolicy,
    formatDb,
    formatPercent,
    iconForLabel,
    initialResult,
    normalizeLabel,
    normalizeResult,
    parseLine,
    parseNanoBridgeLine,
    parseRealtimeInferenceLine,
    riskLevel,
    stripAnsi
  };
});

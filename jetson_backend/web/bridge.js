#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const readline = require("node:readline");
const { spawn } = require("node:child_process");

const parser = require("./realtime-parser.js");

const publicDir = __dirname;
const defaultCwd = path.resolve(__dirname, "..");
const clients = new Set();

let latest = parser.initialResult();
let childProcess = null;
let logStream = null;

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".log": "text/plain; charset=utf-8"
};
const noCacheExts = new Set([".html", ".css", ".js"]);

const demoLines = [
  "[12:00:00] 예측: construction (76.4%) | status=감지 | level=-22.4 dBFS | enhanced=-15.1 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=76.4%, gunshot=2.1%, alarm_siren=8.4%, horn=1.2%, water=6.6%, knock=3.1%, appliances=1.0%, baby_cry=0.8%, animal_cry=1.5%, glass_shatter=2.5%",
  "[12:00:02] 예측: horn (84.8%) | status=감지 | level=-18.9 dBFS | enhanced=-12.0 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=3.3%, gunshot=2.0%, alarm_siren=16.2%, horn=84.8%, water=0.7%, knock=1.5%, appliances=0.8%, baby_cry=1.0%, animal_cry=4.1%, glass_shatter=1.7%",
  "[12:00:04] 예측: water (31.6%) | status=감지 | level=-24.8 dBFS | enhanced=-18.7 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=8.8%, gunshot=0.7%, alarm_siren=2.0%, horn=1.1%, water=31.6%, knock=2.6%, appliances=10.3%, baby_cry=2.2%, animal_cry=4.5%, glass_shatter=0.9%",
  "[12:00:06] 예측: glass_shatter (91.2%) | status=감지 | level=-16.3 dBFS | enhanced=-10.6 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=5.1%, gunshot=3.5%, alarm_siren=7.1%, horn=1.3%, water=0.8%, knock=4.0%, appliances=2.4%, baby_cry=0.9%, animal_cry=1.2%, glass_shatter=91.2%",
  "[12:00:08] 예측: animal_cry (3.0%) | status=낮음(점수낮음 3.0%<5.0%) | level=-35.7 dBFS | enhanced=-29.1 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=1.1%, gunshot=0.4%, alarm_siren=1.8%, horn=2.0%, water=3.5%, knock=1.0%, appliances=0.8%, baby_cry=2.2%, animal_cry=3.0%, glass_shatter=0.6%"
];

function parseArgs(argv) {
  const options = {
    port: 8765,
    host: "0.0.0.0",
    cwd: defaultCwd,
    log: "",
    demo: false,
    command: []
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") {
      options.command = argv.slice(index + 1);
      break;
    }
    if (arg === "--port") {
      options.port = Number(argv[++index]);
      continue;
    }
    if (arg === "--host") {
      options.host = argv[++index];
      continue;
    }
    if (arg === "--cwd") {
      options.cwd = path.resolve(argv[++index]);
      continue;
    }
    if (arg === "--log") {
      options.log = argv[++index];
      continue;
    }
    if (arg === "--demo") {
      options.demo = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    throw new Error(`Unknown option: ${arg}`);
  }

  if (!Number.isFinite(options.port) || options.port <= 0) {
    throw new Error("--port must be a positive number");
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
  node web/bridge.js --port 8765 -- .venv/bin/python -u realtime_inference.py
  node web/bridge.js --port 8765 --demo

Options:
  --port <port>   HTTP port, default 8765
  --host <host>   HTTP host, default 0.0.0.0
  --cwd <path>    Command working directory, default project root
  --log <path>    Append raw inference output to a log file
  --demo          Serve rotating sample events without starting inference
`);
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Range");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
}

function broadcast(data) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    try {
      res.write(payload);
    } catch (_) {
      clients.delete(res);
    }
  }
}

function updateLatest(data) {
  latest = Object.assign({}, parser.initialResult(), data);
  broadcast(latest);
}

function handleLine(line, streamName) {
  const text = parser.stripAnsi(line).trim();
  if (!text) return;

  if (streamName === "stderr") {
    console.error(text);
  } else {
    console.log(text);
  }

  if (logStream) {
    logStream.write(`${text}\n`);
  }

  const parsed = parser.parseLine(text);
  if (parsed) {
    updateLatest(parsed);
  }
}

function streamLines(stream, streamName) {
  const reader = readline.createInterface({ input: stream });
  reader.on("line", (line) => handleLine(line, streamName));
}

function startCommand(options) {
  if (!options.command.length) {
    console.log("No inference command was provided. Serving the page only.");
    return;
  }

  const [command, ...args] = options.command;
  console.log(`Starting inference command in ${options.cwd}:`);
  console.log([command, ...args].join(" "));

  childProcess = spawn(command, args, {
    cwd: options.cwd,
    stdio: ["ignore", "pipe", "pipe"]
  });

  streamLines(childProcess.stdout, "stdout");
  streamLines(childProcess.stderr, "stderr");

  childProcess.on("exit", (code, signal) => {
    const raw = `inference process exited: code=${code} signal=${signal || "-"}`;
    console.log(raw);
    updateLatest(
      Object.assign({}, latest, {
        status: "stopped",
        display_state: "listening",
        display_label: "추론이 종료되었습니다",
        display_icon: "〰️",
        display_score: null,
        items: [],
        level: "listening",
        reason: "process_exit",
        raw,
        display_raw: raw
      })
    );
  });
}

function startDemo() {
  let index = 0;
  console.log("Demo event mode enabled.");

  function tick() {
    const time = new Date().toTimeString().slice(0, 8);
    const line = demoLines[index % demoLines.length].replace(/\[[0-9:]+\]/, `[${time}]`);
    const parsed = parser.parseLine(line);
    if (parsed) updateLatest(parsed);
    index += 1;
  }

  tick();
  setInterval(tick, 2200);
}

function sendJson(res, data) {
  setCors(res);
  res.writeHead(200, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-cache"
  });
  res.end(JSON.stringify(data));
}

function sendEvents(req, res) {
  setCors(res);
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive"
  });
  res.write("retry: 1000\n\n");
  res.write(`data: ${JSON.stringify(latest)}\n\n`);
  clients.add(res);
  req.on("close", () => clients.delete(res));
}

function safeStaticPath(requestPath) {
  const decoded = decodeURIComponent(requestPath.split("?", 1)[0]);
  const pathname = decoded === "/" ? "/index.html" : decoded;
  const resolved = path.resolve(publicDir, `.${pathname}`);
  const relative = path.relative(publicDir, resolved);
  if (relative.startsWith("..") || path.isAbsolute(relative)) return null;
  return resolved;
}

function sendStatic(req, res) {
  const filePath = safeStaticPath(req.url || "/");
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.stat(filePath, (statError, stat) => {
    if (statError || !stat.isFile()) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const ext = path.extname(filePath);
    setCors(res);
    res.writeHead(200, {
      "Content-Type": mimeTypes[ext] || "application/octet-stream",
      "Content-Length": stat.size,
      "Cache-Control": noCacheExts.has(ext) ? "no-cache" : "public, max-age=60"
    });
    fs.createReadStream(filePath).pipe(res);
  });
}

function createServer() {
  return http.createServer((req, res) => {
    if (req.method === "OPTIONS") {
      setCors(res);
      res.writeHead(204);
      res.end();
      return;
    }

    const pathname = (req.url || "/").split("?", 1)[0];
    if (pathname === "/events") {
      sendEvents(req, res);
      return;
    }
    if (pathname === "/latest") {
      sendJson(res, latest);
      return;
    }
    if (pathname === "/health") {
      sendJson(res, { ok: true, clients: clients.size, status: latest.status });
      return;
    }
    sendStatic(req, res);
  });
}

function setupLog(options) {
  if (!options.log) return;
  const logPath = path.resolve(options.cwd, options.log);
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  logStream = fs.createWriteStream(logPath, { flags: "a" });
  console.log(`Writing raw output to ${logPath}`);
}

function shutdown() {
  for (const res of clients) {
    try {
      res.end();
    } catch (_) {
      // ignore shutdown errors
    }
  }
  clients.clear();

  if (childProcess && childProcess.exitCode === null) {
    childProcess.kill("SIGTERM");
  }
  if (logStream) {
    logStream.end();
  }
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  setupLog(options);

  const server = createServer();
  server.listen(options.port, options.host, () => {
    console.log(`Web page: http://<JETSON_IP>:${options.port}`);
    console.log(`Local URL: http://127.0.0.1:${options.port}`);
  });

  if (options.demo) {
    startDemo();
  } else {
    startCommand(options);
  }

  process.on("SIGINT", () => {
    shutdown();
    server.close(() => process.exit(0));
  });
  process.on("SIGTERM", () => {
    shutdown();
    server.close(() => process.exit(0));
  });
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}

(function () {
  const root = document.getElementById("marllenAssistant");
  if (!root) return;

  // Static URL base:
  // - Avoid hardcoded "/static/..." so it works behind reverse proxies with a path prefix.
  // - Prefer same-origin assets for CN/offline environments.
  let STATIC_BASE = "/static/";
  try {
    const cur = document.currentScript && document.currentScript.src ? String(document.currentScript.src) : "";
    if (cur) STATIC_BASE = new URL("./", cur).toString();
  } catch (e) {}

  const canWrite = String((root.dataset && root.dataset.canWrite) ? root.dataset.canWrite : "") === "1";
  const readOnly = String((root.dataset && root.dataset.readOnly) ? root.dataset.readOnly : "") === "1";
  const userRole = (root.dataset && root.dataset.userRole) ? String(root.dataset.userRole) : "";

  const fab = document.getElementById("marllenAssistantFab");
  const panel = document.getElementById("marllenAssistantPanel");
  const header = document.getElementById("marllenAssistantHeader");
  const btnMin = document.getElementById("marllenAssistantMin");
  const btnNew = document.getElementById("marllenAssistantNew");
  const btnThreads = document.getElementById("marllenAssistantThreads");
  const msgWrap = document.getElementById("marllenAssistantMessages");
  const statusEl = document.getElementById("marllenAssistantStatus");
  const errEl = document.getElementById("marllenAssistantError");
  const input = document.getElementById("marllenAssistantInput");
  const btnSend = document.getElementById("marllenAssistantSend");
  const modeBadge = document.getElementById("marllenAssistantMode");
  const tip = document.getElementById("marllenAssistantTip");
  const tipClose = document.getElementById("marllenAssistantTipClose");

  const execWrap = document.getElementById("marllenAssistantExec");
  const execLogEl = document.getElementById("marllenAssistantExecLog");
  const execMetaEl = document.getElementById("marllenAssistantExecMeta");
  const btnExecToggle = document.getElementById("marllenAssistantExecToggle");
  const btnCancel = document.getElementById("marllenAssistantCancel");

  const threadsOverlay = document.getElementById("marllenAssistantThreadsOverlay");
  const btnThreadsClose = document.getElementById("marllenAssistantThreadsClose");
  const threadsSearch = document.getElementById("marllenAssistantThreadsSearch");
  const threadsListEl = document.getElementById("marllenAssistantThreadsList");
  const threadsHintEl = document.getElementById("marllenAssistantThreadsHint");

  const SEND_LABEL = (btnSend && btnSend.textContent) ? String(btnSend.textContent) : "发送";
  const NEW_LABEL = (btnNew && btnNew.textContent) ? String(btnNew.textContent) : "新对话";

  const LS_POS = "marllenAssistant.pos.v1";
  const LS_OPEN = "marllenAssistant.open.v1";
  const LS_TIP_CLOSED = "marllenAssistant.tipClosed.v1";
  const LS_EXEC_OPEN = "marllenAssistant.execOpen.v1";

  let threadId = null;
  let pendingJobId = null;
  let busy = false;
  let pollTimer = null;
  let threadsCache = [];
  let echartsPromise = null;
  let echartsInstances = [];
  let suppressFabClick = false;
  let posBeforeOpen = null;
  let newThreadLock = false;
  let execOpen = true;

  // Prefer same-origin ECharts for China/offline environments; keep CDNs as fallback.
  const ECHARTS_SOURCES = [
    STATIC_BASE + "vendor/echarts.min.js",
    "https://unpkg.com/echarts@5.5.1/dist/echarts.min.js",
    "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js",
  ];

  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function readPos() {
    try {
      const raw = localStorage.getItem(LS_POS);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj || typeof obj.x !== "number" || typeof obj.y !== "number") return null;
      return { x: obj.x, y: obj.y };
    } catch (e) {
      return null;
    }
  }

  function writePos(x, y) {
    try {
      localStorage.setItem(LS_POS, JSON.stringify({ x: x, y: y }));
    } catch (e) {}
  }

  function isExpanded() {
    try {
      return !!(panel && !panel.classList.contains("hidden"));
    } catch (e) {
      return false;
    }
  }

  function lockPageScroll() {
    try { window.__marllenScrollLock && window.__marllenScrollLock.lock(); } catch (e) {}
  }

  function unlockPageScroll() {
    try { window.__marllenScrollLock && window.__marllenScrollLock.unlock(); } catch (e) {}
  }

  function setOpen(open) {
    if (!panel || !fab) return;
    if (open) {
      if (!isExpanded()) {
        try {
          const r = root.getBoundingClientRect();
          posBeforeOpen = { x: r.left, y: r.top };
        } catch (e) {
          posBeforeOpen = null;
        }
      }
      panel.classList.remove("hidden");
      fab.classList.add("hidden");
      // Mobile: prevent background page scroll while chatting.
      lockPageScroll();
      // Ensure expanded panel stays within viewport.
      setTimeout(function () {
        ensureOnScreen({ persist: false });
      }, 0);
    } else {
      panel.classList.add("hidden");
      fab.classList.remove("hidden");
      unlockPageScroll();
      if (posBeforeOpen && typeof posBeforeOpen.x === "number" && typeof posBeforeOpen.y === "number") {
        root.style.left = posBeforeOpen.x + "px";
        root.style.top = posBeforeOpen.y + "px";
        root.style.right = "auto";
        root.style.bottom = "auto";
      }
      posBeforeOpen = null;
      setTimeout(function () {
        ensureOnScreen({ persist: true });
      }, 0);
    }
    try {
      localStorage.setItem(LS_OPEN, open ? "1" : "0");
    } catch (e) {}
  }

  function isOpen() {
    try {
      return localStorage.getItem(LS_OPEN) === "1";
    } catch (e) {
      return false;
    }
  }

  function safeSetStatus(text) {
    if (!statusEl) return;
    statusEl.textContent = text || "";
    statusEl.classList.toggle("hidden", !text);
  }

  function safeSetError(text) {
    if (!errEl) return;
    errEl.textContent = text || "";
    errEl.classList.toggle("hidden", !text);
  }

  function setExecOpen(open) {
    execOpen = !!open;
    try { localStorage.setItem(LS_EXEC_OPEN, execOpen ? "1" : "0"); } catch (e) {}
    if (btnExecToggle) btnExecToggle.textContent = execOpen ? "收起" : "展开";
    if (execLogEl) execLogEl.classList.toggle("hidden", !execOpen);
    if (execMetaEl) execMetaEl.classList.toggle("hidden", !execOpen || !(execMetaEl.textContent || "").trim());
  }

  function updateExecLog(exec) {
    if (!execWrap) return;
    // Show header while busy; hide entirely otherwise.
    execWrap.classList.toggle("hidden", !busy);

    if (btnCancel) {
      btnCancel.disabled = !busy || !pendingJobId;
    }

    if (!execLogEl && !execMetaEl) return;

    const tail = exec && exec.tail ? String(exec.tail) : "";
    const bytes = exec && typeof exec.bytes === "number" ? exec.bytes : Number(exec && exec.bytes ? exec.bytes : 0);
    const truncated = !!(exec && exec.truncated);

    if (execMetaEl) {
      let meta = "";
      if (bytes > 0) {
        const kb = Math.max(1, Math.round(bytes / 1024));
        meta = truncated ? ("日志太长，已截断：当前显示最后 24KB / 总计 " + String(kb) + "KB") : ("日志大小：" + String(kb) + "KB");
      }
      execMetaEl.textContent = meta;
      execMetaEl.classList.toggle("hidden", !execOpen || !meta);
    }

    if (execLogEl) {
      const atBottom = (execLogEl.scrollTop + execLogEl.clientHeight) >= (execLogEl.scrollHeight - 24);
      execLogEl.textContent = tail || (busy ? "（暂无输出）" : "");
      if (atBottom) {
        try { execLogEl.scrollTop = execLogEl.scrollHeight; } catch (e) {}
      }
      execLogEl.classList.toggle("hidden", !execOpen);
    }
  }

  function setBusy(nextBusy, statusText) {
    busy = !!nextBusy;
    if (input) input.disabled = busy || !canWrite;
    if (btnSend) btnSend.disabled = busy || !canWrite;
    // While busy, block "新对话" to keep one coherent timeline (no forced recovery).
    if (btnNew) btnNew.disabled = busy || !canWrite;
    if (btnSend) btnSend.textContent = busy ? "处理中…" : SEND_LABEL;
    if (btnNew) btnNew.textContent = NEW_LABEL;
    safeSetStatus(statusText || (busy ? "分析中…（复杂问题可能较慢，最长 30 分钟）" : ""));

    // Exec panel visibility + cancel availability are coupled to busy state.
    if (execWrap) execWrap.classList.toggle("hidden", !busy);
    if (btnCancel) btnCancel.disabled = !busy || !pendingJobId;
    if (!busy) {
      try { if (execLogEl) execLogEl.textContent = ""; } catch (e) {}
      try { if (execMetaEl) execMetaEl.textContent = ""; } catch (e) {}
    }
    if (btnExecToggle) btnExecToggle.textContent = execOpen ? "收起" : "展开";
    if (execLogEl) execLogEl.classList.toggle("hidden", !execOpen);
    if (execMetaEl) execMetaEl.classList.toggle("hidden", !execOpen || !(execMetaEl.textContent || "").trim());
  }

  function escapeText(s) {
    return (s == null ? "" : String(s));
  }

  function ensureEchartsLoaded() {
    try {
      if (window.echarts) return Promise.resolve(window.echarts);
    } catch (e) {}
    if (echartsPromise) return echartsPromise;

    function loadScript(src, timeoutMs) {
      return new Promise(function (resolve, reject) {
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.referrerPolicy = "no-referrer";
        script.dataset.marllenEcharts = "1";

        let done = false;
        const timer = setTimeout(function () {
          if (done) return;
          done = true;
          try { script.remove(); } catch (e) {}
          reject(new Error("echarts load timeout: " + src));
        }, Math.max(1500, Number(timeoutMs || 6000)));

        script.onload = function () {
          if (done) return;
          done = true;
          clearTimeout(timer);
          try {
            if (window.echarts) resolve(window.echarts);
            else reject(new Error("echarts not available after load: " + src));
          } catch (e) {
            reject(e);
          }
        };
        script.onerror = function () {
          if (done) return;
          done = true;
          clearTimeout(timer);
          try { script.remove(); } catch (e) {}
          reject(new Error("failed to load echarts: " + src));
        };
        document.head.appendChild(script);
      });
    }

    echartsPromise = (async function () {
      let lastErr = null;
      for (const src of ECHARTS_SOURCES) {
        try {
          return await loadScript(src, 6500);
        } catch (e) {
          lastErr = e;
        }
      }
      throw lastErr || new Error("failed to load echarts");
    })().catch(function (e) {
      // Allow retry later (e.g. network recovers, user opens assistant again).
      echartsPromise = null;
      throw e;
    });

    return echartsPromise;
  }

  function safeChartsFromMeta(meta) {
    try {
      if (!meta) return [];
      const charts = meta.charts;
      if (!Array.isArray(charts)) return [];
      return charts.filter((x) => x && typeof x === "object").slice(0, 3);
    } catch (e) {
      return [];
    }
  }

  function buildChartOption(spec) {
    const type = spec && spec.type ? String(spec.type) : "bar";
    const title = spec && spec.title ? String(spec.title) : "";
    const labels = Array.isArray(spec && spec.labels) ? spec.labels.map((x) => String(x)) : [];
    const values = Array.isArray(spec && spec.values) ? spec.values.map((x) => Number(x)) : [];
    const unit = spec && spec.unit ? String(spec.unit) : "";

    const base = {
      backgroundColor: "transparent",
      animationDuration: 450,
      title: title ? { text: title, left: "center", top: 6, textStyle: { fontSize: 12, fontWeight: 700, color: "#0f172a" } } : undefined,
      tooltip: { trigger: (type === "pie" ? "item" : "axis") },
    };

    if (type === "pie") {
      const data = [];
      for (let i = 0; i < Math.min(labels.length, values.length); i++) {
        data.push({ name: labels[i], value: values[i] });
      }
      return Object.assign({}, base, {
        legend: { top: 28, left: "center", textStyle: { fontSize: 10, color: "#64748b" } },
        series: [
          {
            type: "pie",
            radius: ["38%", "70%"],
            center: ["50%", "64%"],
            itemStyle: { borderRadius: 8, borderColor: "#fff", borderWidth: 2 },
            label: { color: "#334155", fontSize: 10, formatter: "{b}: {c}" + unit },
            data: data,
          }
        ],
      });
    }

    if (type === "line") {
      return Object.assign({}, base, {
        grid: { left: 10, right: 12, top: title ? 34 : 16, bottom: 20, containLabel: true },
        xAxis: { type: "category", data: labels, axisLabel: { color: "#64748b", fontSize: 10 } },
        yAxis: { type: "value", axisLabel: { color: "#64748b", fontSize: 10 }, splitLine: { lineStyle: { color: "#e2e8f0" } } },
        series: [
          {
            type: "line",
            data: values,
            smooth: true,
            symbol: "circle",
            symbolSize: 6,
            lineStyle: { width: 3, color: "#6366f1" },
            itemStyle: { color: "#6366f1" },
            areaStyle: { color: "rgba(99, 102, 241, 0.12)" },
          }
        ],
      });
    }

    // default: bar
    const rotate = labels.length >= 7 ? 24 : 0;
    return Object.assign({}, base, {
      grid: { left: 10, right: 12, top: title ? 34 : 16, bottom: rotate ? 40 : 22, containLabel: true },
      xAxis: {
        type: "category",
        data: labels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "#cbd5e1" } },
        axisLabel: { color: "#64748b", fontSize: 10, rotate: rotate },
      },
      yAxis: {
        type: "value",
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#64748b", fontSize: 10 },
        splitLine: { lineStyle: { color: "#e2e8f0" } },
      },
      series: [
        {
          type: "bar",
          data: values,
          barMaxWidth: 28,
          itemStyle: { color: "#6366f1", borderRadius: [8, 8, 4, 4] },
          label: {
            show: true,
            position: "top",
            color: "#334155",
            fontSize: 10,
            formatter: function (p) {
              const v = (p && typeof p.value === "number") ? p.value : (p ? p.value : "");
              return String(v) + unit;
            },
          },
        }
      ],
    });
  }

  function disposeCharts() {
    try {
      for (const inst of echartsInstances || []) {
        try { inst && inst.dispose && inst.dispose(); } catch (e) {}
      }
    } catch (e) {}
    echartsInstances = [];
  }

  function renderMessages(messages, pendingJob) {
    if (!msgWrap) return;
    disposeCharts();
    msgWrap.innerHTML = "";

    const arr = Array.isArray(messages) ? messages : [];
    for (const m of arr) {
      const role = (m && m.role) ? String(m.role) : "user";
      const content = escapeText(m && m.content);
      const contentHtml = (m && m.content_html) ? String(m.content_html) : "";
      const charts = safeChartsFromMeta(m && m.meta ? m.meta : null);
      const row = document.createElement("div");
      row.className = "flex";

      const isUser = role === "user";
      row.classList.add(isUser ? "justify-end" : "justify-start");

      const bubble = document.createElement("div");
      bubble.className = isUser
        ? "max-w-[85%] rounded-2xl px-3 py-2 bg-slate-900 text-white text-sm whitespace-pre-wrap shadow"
        : "max-w-[85%] rounded-2xl px-3 py-2 bg-white border text-slate-900 text-sm shadow-sm";

      const contentEl = document.createElement("div");
      if (!isUser && contentHtml) {
        contentEl.className = "marllen-md";
        contentEl.innerHTML = contentHtml;
      } else {
        contentEl.className = "whitespace-pre-wrap";
        contentEl.textContent = content || (isUser ? "" : "（空）");
      }
      bubble.appendChild(contentEl);

      if (!isUser && charts.length) {
        const chartsWrap = document.createElement("div");
        chartsWrap.className = "mt-2 space-y-2";

        charts.forEach(function (spec, idx) {
          const card = document.createElement("div");
          card.className = "rounded-xl border border-slate-200 bg-slate-50/80 overflow-hidden";

          const canvas = document.createElement("div");
          canvas.className = "marllen-chart";
          canvas.style.height = "180px";
          canvas.style.width = "100%";

          card.appendChild(canvas);

          const note = spec && spec.note ? String(spec.note) : "";
          if (note) {
            const noteEl = document.createElement("div");
            noteEl.className = "px-3 py-2 text-[11px] text-slate-600 border-t bg-white/70";
            noteEl.textContent = note;
            card.appendChild(noteEl);
          }

          chartsWrap.appendChild(card);

          ensureEchartsLoaded().then(function (echarts) {
            try {
              const inst = echarts.init(canvas, null, { renderer: "canvas" });
              echartsInstances.push(inst);
              inst.setOption(buildChartOption(spec), { notMerge: true, lazyUpdate: true });
              try {
                const ro = new ResizeObserver(function () {
                  try { inst.resize(); } catch (e) {}
                });
                ro.observe(canvas);
              } catch (e) {
                // ignore
              }
            } catch (e) {
              canvas.textContent = "（图表渲染失败）";
            }
          }).catch(function () {
            canvas.textContent = "（图表组件加载失败）";
          });
        });

        bubble.appendChild(chartsWrap);
      }

      row.appendChild(bubble);
      msgWrap.appendChild(row);
    }

    if (pendingJob && pendingJob.status && pendingJob.status !== "done" && pendingJob.status !== "error") {
      const row = document.createElement("div");
      row.className = "flex justify-start";
      const bubble = document.createElement("div");
      bubble.className = "max-w-[85%] rounded-2xl px-3 py-2 bg-white/80 backdrop-blur border border-slate-200 text-slate-700 text-sm shadow-sm";

      const st = String(pendingJob.status || "");
      const hint = (st === "pending")
        ? "已提交，正在排队…"
        : "正在检索并分析数据…";

      const line = document.createElement("div");
      line.className = "flex items-center gap-2";

      const dots = document.createElement("span");
      dots.className = "marllen-typing-dots";
      for (let i = 0; i < 3; i++) {
        const d = document.createElement("span");
        d.className = "marllen-typing-dot";
        dots.appendChild(d);
      }

      const text = document.createElement("div");
      text.className = "leading-relaxed";
      text.textContent = hint + "（复杂问题可能需要几十秒～几分钟，请耐心等待）";

      line.appendChild(dots);
      line.appendChild(text);
      bubble.appendChild(line);
      row.appendChild(bubble);
      msgWrap.appendChild(row);
    }

    // Scroll to bottom
    try {
      msgWrap.scrollTop = msgWrap.scrollHeight;
    } catch (e) {}
  }

  function ensureOnScreen(opts) {
    const persist = (opts && typeof opts.persist === "boolean") ? opts.persist : true;
    try {
      const r = root.getBoundingClientRect();
      const maxX = Math.max(8, window.innerWidth - r.width - 8);
      const maxY = Math.max(8, window.innerHeight - r.height - 8);
      const x = clamp(r.left, 8, maxX);
      const y = clamp(r.top, 8, maxY);
      root.style.left = x + "px";
      root.style.top = y + "px";
      root.style.right = "auto";
      root.style.bottom = "auto";
      if (persist) writePos(x, y);
    } catch (e) {}
  }

  async function apiGet(url) {
    const resp = await fetch(url, { credentials: "same-origin" });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = (data && data.detail) ? String(data.detail) : ("请求失败: " + resp.status);
      const err = new Error(msg);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  async function apiPost(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = (data && data.detail) ? String(data.detail) : ("请求失败: " + resp.status);
      const err = new Error(msg);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  async function loadThread() {
    safeSetError("");
    const data = await apiGet("/api/marllen-assistant/thread?limit=120");
    threadId = data && data.thread ? data.thread.id : null;
    const pending = data ? data.pending_job : null;
    pendingJobId = pending && pending.id ? pending.id : null;
    renderMessages(data.messages || [], pending);

    if (pending && pending.id && pending.status && pending.status !== "done" && pending.status !== "error") {
      setBusy(true, "分析中…（复杂问题可能较慢）");
      startPoll(pending.id);
    } else {
      setBusy(false, "");
    }
  }

  async function loadThreadById(tid) {
    safeSetError("");
    const data = await apiGet("/api/marllen-assistant/threads/" + String(tid) + "?limit=120");
    threadId = data && data.thread ? data.thread.id : null;
    const pending = data ? data.pending_job : null;
    pendingJobId = pending && pending.id ? pending.id : null;
    renderMessages(data.messages || [], pending);

    if (pending && pending.id && pending.status && pending.status !== "done" && pending.status !== "error") {
      setBusy(true, "分析中…（复杂问题可能较慢）");
      startPoll(pending.id);
    } else {
      setBusy(false, "");
    }
  }

  async function probePendingAndMaybeOpen() {
    // UX: user refreshes while waiting → keep them in the dialog.
    try {
      const data = await apiGet("/api/marllen-assistant/thread?limit=1");
      const pending = data ? data.pending_job : null;
      if (pending && pending.id && pending.status && pending.status !== "done" && pending.status !== "error") {
        setOpen(true);
        await loadThread();
      }
    } catch (e) {
      // ignore
    }
  }

  function stopPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function startPoll(jobId) {
    stopPoll();
    if (!jobId) return;

    let backoffMs = 800;
    const startedMs0 = Date.now();
    const tick = async function () {
      try {
        const data = await apiGet("/api/marllen-assistant/jobs/" + String(jobId));
        const job = data && data.job ? data.job : null;
        if (!job) {
          setBusy(false, "");
          pendingJobId = null;
          updateExecLog(null);
          return;
        }
        const st = String(job.status || "");
        if (st === "done") {
          pendingJobId = null;
          setBusy(false, "");
          updateExecLog(null);
          await loadThread();
          return;
        }
        if (st === "error") {
          pendingJobId = null;
          setBusy(false, "");
          updateExecLog(null);
          safeSetError("生成失败：" + (job.last_error || "未知错误"));
          await loadThread();
          return;
        }

        // Better UX: show elapsed time & recovery hint for "stuck" jobs.
        let elapsedS = 0;
        try {
          const ts = job.started_at || job.created_at;
          const ms = ts ? (new Date(String(ts))).getTime() : null;
          if (ms && !Number.isNaN(ms)) {
            elapsedS = Math.max(0, Math.floor((Date.now() - ms) / 1000));
          } else {
            elapsedS = Math.max(0, Math.floor((Date.now() - startedMs0) / 1000));
          }
        } catch (e) {
          elapsedS = Math.max(0, Math.floor((Date.now() - startedMs0) / 1000));
        }
        let statusText = "分析中…（复杂问题可能较慢）";
        if (elapsedS >= 60) {
          const mins = Math.max(1, Math.round(elapsedS / 60));
          statusText = "分析中…（已等待 " + String(mins) + " 分钟）";
        }
        setBusy(true, statusText);
        updateExecLog(job.exec || null);
        backoffMs = clamp(Math.floor(backoffMs * 1.25), 800, 4000);
        pollTimer = setTimeout(tick, backoffMs);
      } catch (e) {
        // transient network errors: keep polling, but slower
        backoffMs = clamp(Math.floor(backoffMs * 1.5), 1200, 6000);
        pollTimer = setTimeout(tick, backoffMs);
      }
    };
    pollTimer = setTimeout(tick, backoffMs);
  }

  async function cancelPendingJob() {
    if (!pendingJobId) return;
    safeSetError("");
    safeSetStatus("正在终止…");
    try {
      await apiPost("/api/marllen-assistant/jobs/" + String(pendingJobId) + "/cancel", {});
      stopPoll();
      pendingJobId = null;
      setBusy(false, "");
      updateExecLog(null);
      await loadThread();
    } catch (e) {
      safeSetStatus("");
      safeSetError(e && e.message ? String(e.message) : "终止失败");
    }
  }

  async function send() {
    if (busy) return;
    if (!canWrite) {
      safeSetError("当前账号无提问权限。");
      return;
    }
    if (!threadId) await loadThread();
    if (!threadId) {
      safeSetError("无法初始化线程，请刷新页面重试。");
      return;
    }

    const text = (input && input.value ? String(input.value) : "").trim();
    if (!text) return;

    safeSetError("");
    setBusy(true, "提交中…");
    try {
      const data = await apiPost("/api/marllen-assistant/threads/" + String(threadId) + "/messages", { text: text });
      if (input) input.value = "";
      const jobId = data && data.job_id ? data.job_id : null;
      pendingJobId = jobId;
      updateExecLog(null);
      await loadThread();
      if (jobId) startPoll(jobId);
    } catch (e) {
      setBusy(false, "");
      safeSetError(e && e.message ? String(e.message) : "提交失败");
    }
  }

  async function newThread() {
    if (newThreadLock) return;
    if (busy) return;
    if (!canWrite) {
      safeSetError("当前账号无新建对话权限。");
      return;
    }
    safeSetError("");
    setBusy(true, "新建对话…");
    newThreadLock = true;
    try {
      await apiPost("/api/marllen-assistant/threads/new", {});
      pendingJobId = null;
      stopPoll();
      await loadThread();
      setBusy(false, "");
    } catch (e) {
      setBusy(false, "");
      safeSetError(e && e.message ? String(e.message) : "新建失败");
    } finally {
      newThreadLock = false;
    }
  }

  function setThreadsOverlayOpen(open) {
    if (!threadsOverlay) return;
    threadsOverlay.classList.toggle("hidden", !open);
    if (open) {
      try {
        if (threadsSearch) threadsSearch.focus();
      } catch (e) {}
    }
  }

  function formatTs(iso) {
    if (!iso) return "";
    try {
      const d = new Date(String(iso));
      if (Number.isNaN(d.getTime())) return String(iso);
      return d.toLocaleString();
    } catch (e) {
      return String(iso);
    }
  }

  function renderThreadsList(threads, query) {
    if (!threadsListEl) return;
    const q = (query || "").trim().toLowerCase();
    const arr = Array.isArray(threads) ? threads : [];
    const filtered = q
      ? arr.filter((t) => String((t && t.title) ? t.title : "").toLowerCase().includes(q))
      : arr;

    threadsListEl.innerHTML = "";
    if (threadsHintEl) threadsHintEl.classList.add("hidden");

    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "px-3 py-8 text-center text-sm text-slate-500";
      empty.textContent = q ? "没有匹配的对话" : "暂无对话记录";
      threadsListEl.appendChild(empty);
      return;
    }

    for (const t of filtered) {
      const tid = t && t.id ? Number(t.id) : 0;
      const title = String((t && t.title) ? t.title : "Marllen小助手");
      const isArchived = !!(t && t.is_archived);
      const updatedAt = t && t.updated_at ? String(t.updated_at) : (t && t.created_at ? String(t.created_at) : "");

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "w-full text-left px-3 py-2 rounded-2xl border bg-white hover:bg-slate-50 active:scale-[0.99]";
      if (tid && threadId && String(tid) === String(threadId)) {
        btn.classList.add("ring-2", "ring-indigo-200", "border-indigo-200");
      } else {
        btn.classList.add("border-slate-200");
      }

      const row1 = document.createElement("div");
      row1.className = "flex items-center justify-between gap-2";
      const titleEl = document.createElement("div");
      titleEl.className = "font-semibold text-sm truncate";
      titleEl.textContent = title || "Marllen小助手";
      const badge = document.createElement("span");
      badge.className = "text-[10px] px-2 py-0.5 rounded-full whitespace-nowrap " + (isArchived ? "bg-slate-100 text-slate-600" : "bg-emerald-100 text-emerald-700");
      badge.textContent = isArchived ? "历史" : "当前";
      row1.appendChild(titleEl);
      row1.appendChild(badge);

      const row2 = document.createElement("div");
      row2.className = "mt-0.5 text-[11px] text-slate-500";
      row2.textContent = updatedAt ? ("更新于 " + formatTs(updatedAt)) : "";

      btn.appendChild(row1);
      btn.appendChild(row2);

      btn.addEventListener("click", function () {
        if (!tid) return;
        stopPoll();
        setBusy(false, "");
        loadThreadById(tid).catch(function (e) {
          safeSetError(e && e.message ? String(e.message) : "加载失败");
        });
        setThreadsOverlayOpen(false);
      });

      threadsListEl.appendChild(btn);
    }

    if (threadsHintEl) {
      if (!canWrite) {
        threadsHintEl.textContent = "当前账号为只读权限：仅可查看历史对话。";
      } else if (readOnly) {
        threadsHintEl.textContent = "提示：发送消息到历史对话会自动切换为当前对话。只读模式：不会自动改系统数据/配置，需要改动请联系管理员。";
      } else {
        threadsHintEl.textContent = "提示：发送消息到历史对话会自动切换为当前对话。";
      }
      threadsHintEl.classList.remove("hidden");
    }
  }

  async function loadThreads() {
    const data = await apiGet("/api/marllen-assistant/threads?limit=80");
    threadsCache = (data && data.threads) ? data.threads : [];
    return threadsCache;
  }

  // Dragging (panel header)
  (function setupDrag() {
    if (!header) return;

    let dragging = false;
    let moved = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    function getRect() {
      try {
        return root.getBoundingClientRect();
      } catch (e) {
        return { left: 0, top: 0, width: 0, height: 0 };
      }
    }

    header.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      // Do not start dragging when interacting with buttons/inputs inside the header.
      try {
        const target = e && e.target ? e.target : null;
        if (target && target.closest && target.closest("button, a, input, textarea, select, [role='button']")) {
          return;
        }
      } catch (err) {}
      dragging = true;
      moved = false;
      safeSetError("");
      const r = getRect();
      startX = e.clientX;
      startY = e.clientY;
      startLeft = r.left;
      startTop = r.top;
      try {
        header.setPointerCapture(e.pointerId);
      } catch (err) {}
    });

    header.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (!moved && (Math.abs(dx) + Math.abs(dy) > 3)) {
        moved = true;
        // User explicitly moved the expanded panel: don't snap back to the pre-open FAB position.
        posBeforeOpen = null;
      }

      const r = getRect();
      const maxX = Math.max(8, window.innerWidth - r.width - 8);
      const maxY = Math.max(8, window.innerHeight - r.height - 8);
      const x = clamp(startLeft + dx, 8, maxX);
      const y = clamp(startTop + dy, 8, maxY);

      root.style.left = x + "px";
      root.style.top = y + "px";
      root.style.right = "auto";
      root.style.bottom = "auto";
    });

    header.addEventListener("pointerup", function (e) {
      if (!dragging) return;
      dragging = false;
      if (moved) {
        const r = getRect();
        writePos(r.left, r.top);
      }
      moved = false;
      try {
        header.releasePointerCapture(e.pointerId);
      } catch (err) {}
    });

    header.addEventListener("pointercancel", function () {
      dragging = false;
      moved = false;
    });
  })();

  // Dragging (collapsed FAB)
  (function setupFabDrag() {
    if (!fab) return;

    let dragging = false;
    let moved = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    function getRect() {
      try {
        return root.getBoundingClientRect();
      } catch (e) {
        return { left: 0, top: 0, width: 0, height: 0 };
      }
    }

    fab.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      dragging = true;
      moved = false;
      suppressFabClick = false;
      safeSetError("");
      const r = getRect();
      startX = e.clientX;
      startY = e.clientY;
      startLeft = r.left;
      startTop = r.top;
      try {
        fab.setPointerCapture(e.pointerId);
      } catch (err) {}
    });

    fab.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (!moved && (Math.abs(dx) + Math.abs(dy) > 6)) {
        moved = true;
        suppressFabClick = true;
      }
      if (!moved) return;
      try { e.preventDefault(); } catch (err) {}

      const r = getRect();
      const maxX = Math.max(8, window.innerWidth - r.width - 8);
      const maxY = Math.max(8, window.innerHeight - r.height - 8);
      const x = clamp(startLeft + dx, 8, maxX);
      const y = clamp(startTop + dy, 8, maxY);

      root.style.left = x + "px";
      root.style.top = y + "px";
      root.style.right = "auto";
      root.style.bottom = "auto";
    });

    function end(e) {
      if (!dragging) return;
      dragging = false;
      if (moved) {
        const r = getRect();
        writePos(r.left, r.top);
      }
      try {
        fab.releasePointerCapture(e.pointerId);
      } catch (err) {}
    }

    fab.addEventListener("pointerup", end);
    fab.addEventListener("pointercancel", function () {
      dragging = false;
      moved = false;
    });
  })();

  // Wiring
  fab && fab.addEventListener("click", function (e) {
    if (suppressFabClick) {
      suppressFabClick = false;
      try { e.preventDefault(); e.stopPropagation(); } catch (err) {}
      return;
    }
    setOpen(true);
    loadThread().catch(function (e) {
      safeSetError(e && e.message ? String(e.message) : "加载失败");
    });
  });
  btnMin && btnMin.addEventListener("click", function () {
    setOpen(false);
  });
  btnNew && btnNew.addEventListener("click", function () {
    newThread();
  });
  btnThreads && btnThreads.addEventListener("click", function () {
    safeSetError("");
    setThreadsOverlayOpen(true);
    loadThreads().then(function (rows) {
      renderThreadsList(rows, threadsSearch && threadsSearch.value ? String(threadsSearch.value) : "");
    }).catch(function (e) {
      safeSetError(e && e.message ? String(e.message) : "加载对话列表失败");
    });
  });
  btnThreadsClose && btnThreadsClose.addEventListener("click", function () {
    setThreadsOverlayOpen(false);
  });
  btnSend && btnSend.addEventListener("click", function () {
    send();
  });

  input && input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // Tip close
  (function initTip() {
    if (!tip) return;
    try {
      const closed = localStorage.getItem(LS_TIP_CLOSED) === "1";
      if (closed) tip.classList.add("hidden");
    } catch (e) {}
    tipClose && tipClose.addEventListener("click", function () {
      try { tip.classList.add("hidden"); } catch (e) {}
      try { localStorage.setItem(LS_TIP_CLOSED, "1"); } catch (e) {}
    });
  })();

  // Initial placement + open state
  (function initPosAndState() {
    if (modeBadge) modeBadge.classList.toggle("hidden", !readOnly);
    try {
      if (modeBadge && readOnly) {
        modeBadge.textContent = "只读";
        if (userRole) modeBadge.title = "当前角色：" + userRole;
      }
    } catch (e) {}

    const pos = readPos();
    if (pos) {
      root.style.left = pos.x + "px";
      root.style.top = pos.y + "px";
      root.style.right = "auto";
      root.style.bottom = "auto";
    } else {
      // Default: bottom-right-ish (measured after layout)
      setTimeout(function () {
        try {
          const r = root.getBoundingClientRect();
          const x = Math.max(8, window.innerWidth - r.width - 16);
          const y = Math.max(8, window.innerHeight - r.height - 16);
          root.style.left = x + "px";
          root.style.top = y + "px";
          root.style.right = "auto";
          root.style.bottom = "auto";
          writePos(x, y);
        } catch (e) {}
      }, 0);
    }

    const open = isOpen();
    setOpen(open);
    if (open) {
      loadThread().catch(function (e) {
        safeSetError(e && e.message ? String(e.message) : "加载失败");
      });
    } else {
      probePendingAndMaybeOpen();
    }
  })();

  // Exec panel prefs + actions
  (function initExecPanel() {
    try {
      execOpen = localStorage.getItem(LS_EXEC_OPEN) !== "0";
    } catch (e) {
      execOpen = true;
    }
    setExecOpen(execOpen);
    btnExecToggle && btnExecToggle.addEventListener("click", function () {
      setExecOpen(!execOpen);
    });
    btnCancel && btnCancel.addEventListener("click", function () {
      cancelPendingJob();
    });
  })();

  threadsSearch && threadsSearch.addEventListener("input", function () {
    renderThreadsList(threadsCache, threadsSearch.value ? String(threadsSearch.value) : "");
  });

  window.addEventListener("resize", function () {
    // Keep it on screen when viewport changes (mobile rotation, etc.)
    ensureOnScreen({ persist: !isExpanded() });
  });
})();

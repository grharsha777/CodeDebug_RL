/* ─── State ─────────────────────────────────────────────────── */
const S = {
  bootstrap: null,
  session: null,
  selectedStep: null,
  submitting: false,
  editorDirty: false,
  lastError: null,
};

const el = {};

/* ─── Boot ──────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", async () => {
  cacheEls();
  bindEvents();
  await boot();
});

function cacheEls() {
  const ids = [
    "hero-title","hero-subtitle","proof-strip","hero-task-count",
    "hero-baseline-score","hero-validation","summary-bar",
    "task-select","difficulty-select","patch-format",
    "code-editor","reasoning-input","commit-input",
    "reset-button","submit-button","editor-task-label","editor-difficulty-label",
    "run-status-chip","evaluation-root","chart-root","history-root","metadata-root",
    "benchmark-overview","task-table-root","baseline-root","recent-episodes-root",
    "compliance-root","api-root","spec-root",
    "breadcrumb","topbar-episode","env-status-pill","env-dot","env-status-text",
    "sb-task-count","sb-baseline","sb-health","refresh-btn",
  ];
  ids.forEach(id => {
    const key = id.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    el[key] = document.getElementById(id);
  });
}

function bindEvents() {
  // Sidebar nav
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  // Editor
  el.codeEditor && el.codeEditor.addEventListener("input", () => { S.editorDirty = true; });
  // Actions
  el.resetButton && el.resetButton.addEventListener("click", () => resetEpisode());
  el.submitButton && el.submitButton.addEventListener("click", () => submitStep());
  el.refreshBtn && el.refreshBtn.addEventListener("click", () => refreshAll());
}

async function boot() {
  setEnvStatus("Initializing", "idle");
  try {
    const data = await api("ui/bootstrap");
    S.bootstrap = data;
    S.session = data.session;
    renderAll(false);
    setEnvStatus("Ready", "ok");
    updateSidebar();
    if (!data.session?.initialized) {
      const task = pickTask(data.benchmark?.tasks || []);
      await resetEpisode(task?.task_id || null);
    }
    healthCheck();
  } catch (e) {
    S.lastError = e.message;
    setEnvStatus("Error", "error");
    renderFatal(e.message);
  }
}

function pickTask(tasks) {
  return tasks.find(t => t.difficulty === "medium") || tasks[0] || null;
}

async function refreshAll() {
  try {
    await Promise.all([refreshSession(), refreshMetrics()]);
    renderAll(false);
  } catch (e) { S.lastError = e.message; }
}

async function refreshSession() {
  S.session = await api("ui/session");
}
async function refreshMetrics() {
  if (!S.bootstrap) return;
  try { S.bootstrap.metrics = await api("metrics"); } catch (_) {}
}

async function resetEpisode(taskId = null) {
  setBusy(true, "Resetting\u2026");
  S.lastError = null;
  S.selectedStep = null;
  const tid = taskId || el.taskSelect?.value || "";
  const diff = el.difficultySelect?.value || "";
  const body = {};
  if (tid) body.task_id = tid;
  else if (diff) body.difficulty = diff;
  try {
    await api("reset", { method: "POST", body: JSON.stringify(body) });
    if (el.reasoningInput) el.reasoningInput.value = "";
    if (el.commitInput) el.commitInput.value = "";
    S.editorDirty = false;
    await Promise.all([refreshSession(), refreshMetrics()]);
    renderAll(false);
    setEnvStatus("Active", "ok");
    updateSidebar();
  } catch (e) {
    S.lastError = e.message;
    renderAll(true);
    setEnvStatus("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function submitStep() {
  setBusy(true, "Running\u2026");
  S.lastError = null;
  const payload = {
    action: {
      patched_code: el.codeEditor?.value || "",
      patch_format: el.patchFormat?.value || "full_replace",
    },
  };
  const reasoning = el.reasoningInput?.value.trim();
  const commit = el.commitInput?.value.trim();
  if (reasoning) payload.action.reasoning = reasoning;
  if (commit) payload.action.commit_message = commit;
  try {
    await api("step", { method: "POST", body: JSON.stringify(payload) });
    S.editorDirty = false;
    await Promise.all([refreshSession(), refreshMetrics()]);
    renderAll(false);
    updateSidebar();
  } catch (e) {
    S.lastError = e.message;
    renderAll(true);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy, label = "Run step") {
  S.submitting = busy;
  if (el.submitButton) {
    el.submitButton.disabled = busy;
    el.submitButton.innerHTML = busy
      ? `<svg width="14" height="14" viewBox="0 0 14 14" style="animation:spin .8s linear infinite"><circle cx="7" cy="7" r="5" stroke="currentColor" stroke-width="1.5" fill="none" stroke-dasharray="20" stroke-dashoffset="10"/></svg> ${label}`
      : `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><polygon points="3,2 13,7 3,12" fill="currentColor"/></svg> Run step`;
  }
  if (el.resetButton) el.resetButton.disabled = busy;
}

/* ─── Tab switching ─────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("is-active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("is-active", p.dataset.panel === tab));
  if (el.breadcrumb) el.breadcrumb.textContent = labelize(tab);
}

/* ─── Status helpers ─────────────────────────────────────────── */
function setEnvStatus(text, state) {
  if (el.envStatusText) el.envStatusText.textContent = text;
  if (el.envDot) el.envDot.className = `dot dot--${state}`;
}

function updateSidebar() {
  const tasks = S.bootstrap?.benchmark?.tasks || [];
  const baseline = S.bootstrap?.benchmark?.baseline || {};
  if (el.sbTaskCount) el.sbTaskCount.textContent = tasks.length || "-";
  if (el.sbBaseline) el.sbBaseline.textContent = fmt(baseline.average_score);
}

async function healthCheck() {
  try {
    const h = await api("health");
    if (el.sbHealth) {
      const ok = h.status === "healthy";
      el.sbHealth.innerHTML = `<span class="dot dot--${ok ? 'ok' : 'error'}"></span> ${ok ? "Healthy" : "Degraded"}`;
    }
  } catch (_) {
    if (el.sbHealth) el.sbHealth.innerHTML = `<span class="dot dot--error"></span> Unreachable`;
  }
}

/* ─── Render ────────────────────────────────────────────────── */
function renderAll(preserveEditor) {
  if (!S.bootstrap) return;
  renderHero();
  renderTaskSelect();
  renderSummaryBar();
  renderWorkbench(preserveEditor);
  renderBenchmark();
  renderSystem();
}

function renderHero() {
  const { product, benchmark, compliance } = S.bootstrap;
  const baseline = benchmark?.baseline || {};
  const v = compliance?.validator_status || {};

  setText("heroTitle", product?.title || "CodeDebug-RL");
  setText("heroSubtitle", product?.subtitle || "Enterprise RL environment for training self-correcting code agents.");
  setText("heroTaskCount", benchmark?.task_count ?? "—");
  setText("heroBaselineScore", fmt(baseline.average_score));

  const ready = v.openenv_yaml_present && v.dockerfile_present && v.inference_script_present;
  if (el.heroValidation) {
    el.heroValidation.innerHTML = ready
      ? `<span class="dot dot--ok"></span> Valid`
      : `<span class="dot dot--warn"></span> Incomplete`;
  }
  if (el.proofStrip) {
    const proofs = product?.proof_points || ["OpenEnv-compatible API","3+ graded tasks","HF Space + Docker ready","Structured reward + telemetry"];
    el.proofStrip.innerHTML = proofs.map(p => `<span class="proof-badge"><span class="dot dot--ok"></span>${esc(p)}</span>`).join("");
  }
}

function renderTaskSelect() {
  if (!el.taskSelect) return;
  const tasks = S.bootstrap?.benchmark?.tasks || [];
  const curTask = S.session?.task?.task_id || "";
  el.taskSelect.innerHTML = [
    '<option value="">Random task</option>',
    ...tasks.map(t =>
      `<option value="${esc(t.task_id)}"${t.task_id === curTask ? " selected" : ""}>${esc(t.task_id)} · ${esc(t.difficulty)}</option>`
    ),
  ].join("");
}

function renderSummaryBar() {
  if (!el.summaryBar) return;
  const session = S.session;
  const sum = session?.summary;
  const task = session?.task;

  if (!session?.initialized || !sum || !task) {
    el.summaryBar.innerHTML = `
      <div class="summary-cell"><div class="sc-label">Status</div><div class="sc-value">Idle</div></div>
      <div class="summary-cell"><div class="sc-label">Score</div><div class="sc-value">—</div></div>
      <div class="summary-cell"><div class="sc-label">Reward</div><div class="sc-value">—</div></div>
      <div class="summary-cell"><div class="sc-label">Steps</div><div class="sc-value">—</div></div>
    `;
    return;
  }

  const topClass = sum.solved ? "positive" : (sum.done ? "negative" : "");

  el.summaryBar.innerHTML = `
    <div class="summary-cell">
      <div class="sc-label">Task</div>
      <div class="sc-value" style="font-size:14px;">${esc(task.task_id)}</div>
      <div class="sc-sub">${esc(task.difficulty)} · ${(task.tags||[]).map(t=>`<span class="tag-chip">${esc(t)}</span>`).join(" ")}</div>
    </div>
    <div class="summary-cell ${topClass}">
      <div class="sc-label">Score</div>
      <div class="sc-value">${fmt(sum.current_score)}</div>
      <div class="sc-sub">Best: ${fmt(sum.best_score)}</div>
    </div>
    <div class="summary-cell">
      <div class="sc-label">Reward</div>
      <div class="sc-value">${fmtSigned(sum.latest_reward)}</div>
      <div class="sc-sub">Cumulative: ${fmtSigned(sum.cumulative_reward)}</div>
    </div>
    <div class="summary-cell">
      <div class="sc-label">Tests</div>
      <div class="sc-value">${sum.current_passed}/${sum.current_total}</div>
      <div class="sc-sub">Pass rate: ${fmtPct(sum.peak_pass_rate)}</div>
    </div>
    <div class="summary-cell">
      <div class="sc-label">Steps</div>
      <div class="sc-value">${sum.step_index}/${sum.max_steps}</div>
      <div class="sc-sub">${esc(labelize(sum.status))}</div>
    </div>
  `;

  // Update topbar episode
  if (el.topbarEpisode) {
    el.topbarEpisode.textContent = session.episode_id ? `ep:${session.episode_id.slice(0,8)}` : "";
  }
}

function renderWorkbench(preserveEditor) {
  const session = S.session;
  const task = session?.task;
  const sum = session?.summary;

  if (session?.initialized && (!preserveEditor || !S.editorDirty)) {
    if (el.codeEditor) el.codeEditor.value = session.code?.current || "";
  }

  if (el.editorTaskLabel) el.editorTaskLabel.textContent = task?.task_id || "No active task";
  if (el.editorDifficultyLabel) {
    const diff = task?.difficulty || "";
    el.editorDifficultyLabel.textContent = labelize(diff || "Idle");
    el.editorDifficultyLabel.className = `diff-badge diff-badge--${diff || "idle"}`;
  }
  if (task?.task_id && el.taskSelect) el.taskSelect.value = task.task_id;

  // Status chip
  if (el.runStatusChip) {
    const st = sum?.execution_status || "idle";
    const tone = chipTone(st);
    el.runStatusChip.textContent = labelize(st);
    el.runStatusChip.className = `status-chip ${tone}`;
  }

  const run = getSelectedRun();
  renderEvalPanel(run, sum);
  renderChart();
  renderHistory();
  renderMetadataPanel(run);
}

/* ─── Evaluation inspector ──────────────────────────────────── */
function renderEvalPanel(run, sum) {
  if (!el.evaluationRoot) return;

  if (!run) {
    el.evaluationRoot.innerHTML = `
      <div class="eval-empty">
        <div class="eval-empty-icon">⬡</div>
        <div class="eval-empty-title">No run data</div>
        <div class="eval-empty-sub">Reset the episode to load a task and inspect baseline grader output.</div>
      </div>
    `;
    return;
  }

  const failures = run.failures || [];
  const rb = run.reward_breakdown || {};
  const deltaVsBaseline = (run.score || 0) - (sum?.baseline_score || 0);
  const isBaseline = run.kind === "baseline";

  el.evaluationRoot.innerHTML = `
    ${S.lastError ? `<div style="background:var(--red-dim);border:1px solid rgba(240,90,90,.3);border-radius:var(--radius);padding:10px 14px;font-size:12px;color:var(--red);margin-bottom:8px;">${esc(S.lastError)}</div>` : ""}

    <div class="eval-run-header">
      <div>
        <div class="eval-run-title">${isBaseline ? "Baseline run" : `Step ${run.step_index}`}</div>
        <div class="eval-run-meta">${esc(run.execution_status || run.status || "—")}</div>
      </div>
      <div class="eval-pill-row">
        <span class="status-chip ${chipTone(run.status)}">${esc(labelize(run.status || "unknown"))}</span>
        <span class="status-chip">${run.syntax_valid ? "Syntax valid" : "Syntax error"}</span>
        ${run.duration_ms ? `<span class="status-chip">${fmtMs(run.duration_ms)}</span>` : ""}
      </div>
    </div>

    <div class="eval-kpi-strip">
      <div class="eval-kpi">
        <div class="eval-kpi-label">Score</div>
        <div class="eval-kpi-value ${scoreClass(run.score)}">${fmt(run.score)}</div>
      </div>
      <div class="eval-kpi">
        <div class="eval-kpi-label">Reward</div>
        <div class="eval-kpi-value ${run.reward_delta >= 0 ? 'green' : 'red'}">${isBaseline ? "—" : fmtSigned(run.reward_delta)}</div>
      </div>
      <div class="eval-kpi">
        <div class="eval-kpi-label">Tests passed</div>
        <div class="eval-kpi-value">${run.passed ?? 0}/${run.total ?? 0}</div>
      </div>
      <div class="eval-kpi">
        <div class="eval-kpi-label">Patch lines</div>
        <div class="eval-kpi-value">${isBaseline ? "—" : (run.diff_lines || 0)}</div>
      </div>
    </div>

    <div class="eval-delta-row">
      <div class="eval-delta-card">
        <div class="eval-delta-label">Δ vs baseline</div>
        <div class="eval-delta-value ${deltaVsBaseline >= 0 ? 'pos' : 'neg'}">${fmtSigned(deltaVsBaseline)}</div>
      </div>
      <div class="eval-delta-card">
        <div class="eval-delta-label">Cumulative reward</div>
        <div class="eval-delta-value ${(run.cumulative_reward||0) >= 0 ? 'pos' : 'neg'}">${fmtSigned(run.cumulative_reward || 0)}</div>
      </div>
      <div class="eval-delta-card">
        <div class="eval-delta-label">Execution status</div>
        <div class="eval-delta-value" style="font-size:12px;font-weight:600;color:var(--text-2)">${esc(labelize(run.execution_status || run.status || "—"))}</div>
      </div>
    </div>

    <div>
      <div class="eval-section-title">Failure groups</div>
      ${failures.length
        ? failures.map((f, i) => renderFailGroup(f, i === 0)).join("")
        : `<div class="eval-no-fails"><span class="dot dot--ok"></span> No failures captured for this run.</div>`}
    </div>

    ${Object.keys(rb).length ? `
    <div>
      <div class="eval-section-title">Reward breakdown</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        ${Object.entries(rb).filter(([,v]) => v !== 0).map(([k,v]) => `
          <div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:8px 10px;display:flex;justify-content:space-between;align-items:center;font-size:12px;">
            <span style="color:var(--text-2)">${esc(labelize(k))}</span>
            <strong style="color:${v>=0?'var(--green)':'var(--red)'};">${fmtSigned(v)}</strong>
          </div>
        `).join("")}
      </div>
    </div>
    ` : ""}

    ${run.stdout || run.stderr || run.diff_preview ? `
    <div>
      <div class="eval-section-title">Artifacts</div>
      ${run.stdout ? renderExpandable("Stdout", run.stdout) : ""}
      ${run.stderr ? renderExpandable("Stderr", run.stderr) : ""}
      ${run.diff_preview ? renderExpandable("Diff preview", run.diff_preview) : ""}
    </div>
    ` : ""}
  `;
}

function renderFailGroup(f, open) {
  return `
    <details class="fail-group" ${open ? "open" : ""}>
      <summary style="cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:8px;">
        <div class="fail-header">
          <span class="fail-title">${esc(f.title || f.test_name || "Failure")}</span>
          <span class="fail-body">${esc(f.assertion || f.root_cause || "")}</span>
        </div>
        <span class="status-chip error" style="flex-shrink:0;">${esc(labelize(f.severity || "error"))}</span>
      </summary>
      ${f.root_cause ? `<div class="fail-body" style="color:var(--text-3)"><strong>Root cause</strong><br>${esc(f.root_cause)}</div>` : ""}
      ${f.trace_excerpt ? `<pre class="fail-trace">${esc(f.trace_excerpt)}</pre>` : ""}
    </details>
  `;
}

function renderExpandable(title, content) {
  return `
    <details style="margin-bottom:6px;">
      <summary style="cursor:pointer;font-size:12px;font-weight:600;color:var(--text-2);padding:6px 0;">${esc(title)}</summary>
      <pre class="fail-trace">${esc(content)}</pre>
    </details>
  `;
}

/* ─── Chart ─────────────────────────────────────────────────── */
function renderChart() {
  if (!el.chartRoot) return;
  const session = S.session;
  if (!session?.initialized) {
    el.chartRoot.innerHTML = '<div class="chart-empty">Reset and run steps to populate the trajectory chart.</div>';
    return;
  }

  const baselineScore = session.summary?.baseline_score || 0;
  const history = session.history || [];
  const pts = [{ step: 0, score: baselineScore, reward: 0 }];
  history.forEach(r => pts.push({ step: r.step_index, score: r.score || 0, reward: r.cumulative_reward || 0 }));

  const W = 560, H = 120, PAD = 28;
  const maxStep = Math.max(1, ...pts.map(p => p.step));
  const rewardVals = pts.map(p => p.reward);
  const rMin = Math.min(0, ...rewardVals), rMax = Math.max(0.01, ...rewardVals);

  const sx = step => PAD + ((W - PAD*2) * step) / maxStep;
  const sy = (v, lo, hi) => hi === lo ? H/2 : H - PAD - ((H - PAD*2) * (v - lo)) / (hi - lo);

  const scorePts  = pts.map(p => `${sx(p.step)},${sy(p.score, 0, 1)}`).join(" ");
  const rewardPts = pts.map(p => `${sx(p.step)},${sy(p.reward, rMin, rMax)}`).join(" ");

  el.chartRoot.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Score/reward trajectory">
      <defs>
        <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--accent)" stop-opacity=".3"/>
          <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <line x1="${PAD}" y1="${H-PAD}" x2="${W-PAD}" y2="${H-PAD}" stroke="var(--border)" stroke-width="1"/>
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H-PAD}" stroke="var(--border)" stroke-width="1"/>
      <text x="${PAD}" y="${PAD-5}" fill="var(--text-3)" font-size="9" font-family="Inter">1.0</text>
      <text x="${PAD}" y="${H-10}" fill="var(--text-3)" font-size="9" font-family="Inter">0.0</text>
      <polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" points="${scorePts}"/>
      <polyline fill="none" stroke="var(--green)" stroke-width="2" stroke-linejoin="round" stroke-dasharray="4,3" points="${rewardPts}"/>
      ${pts.map(p => `<circle cx="${sx(p.step)}" cy="${sy(p.score,0,1)}" r="3.5" fill="var(--accent)" />`).join("")}
    </svg>
    <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:var(--text-3);">
      <span><span style="display:inline-block;width:12px;height:2px;background:var(--accent);vertical-align:middle;margin-right:5px;"></span>Score</span>
      <span><span style="display:inline-block;width:12px;height:2px;background:var(--green);vertical-align:middle;margin-right:5px;border-top:2px dashed var(--green);"></span>Cumulative reward</span>
    </div>
  `;
}

/* ─── History ───────────────────────────────────────────────── */
function renderHistory() {
  if (!el.historyRoot) return;
  const history = S.session?.history || [];
  if (!history.length) {
    el.historyRoot.innerHTML = '<div style="color:var(--text-3);font-size:12px;padding:20px 0;text-align:center;">No steps yet. Submit a patch to start.</div>';
    return;
  }
  el.historyRoot.innerHTML = history.map(r => `
    <div class="history-row" data-step="${r.step_index}" style="cursor:pointer;${S.selectedStep===r.step_index?'border-color:var(--accent);':''}" role="button" tabindex="0">
      <span class="hr-step">#${r.step_index}</span>
      <span class="status-chip ${chipTone(r.status)} hr-badge" style="font-size:10px;padding:2px 7px;">${esc(labelize(r.status))}</span>
      <span style="font-size:11px;color:var(--text-3);flex:1;">${r.passed}/${r.total} tests</span>
      <span class="hr-score" style="color:${scoreColor(r.score)}">${fmt(r.score)}</span>
      <span class="hr-reward" style="color:${r.reward_delta>=0?'var(--green)':'var(--red)'}">${fmtSigned(r.reward_delta)}</span>
    </div>
  `).join("");

  el.historyRoot.querySelectorAll(".history-row").forEach(row => {
    const activate = () => {
      const step = Number(row.dataset.step);
      S.selectedStep = S.selectedStep === step ? null : step;
      renderWorkbench(true);
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") activate(); });
  });
}

/* ─── Metadata ──────────────────────────────────────────────── */
function renderMetadataPanel(run) {
  if (!el.metadataRoot) return;
  const session = S.session;
  if (!session?.initialized) {
    el.metadataRoot.innerHTML = '<div style="color:var(--text-3);font-size:12px;padding:20px 0;text-align:center;">Reset to load task metadata.</div>';
    return;
  }
  const task = session.task;
  const sum = session.summary;
  const act = session.latest_action || {};

  el.metadataRoot.innerHTML = `
    <div class="meta-row"><span class="meta-key">Episode</span><span class="meta-val">${(session.episode_id||"").slice(0,10)}</span></div>
    <div class="meta-row"><span class="meta-key">Task</span><span class="meta-val">${esc(task.task_id)}</span></div>
    <div class="meta-row"><span class="meta-key">Difficulty</span><span class="meta-val"><span class="diff-badge diff-badge--${task.difficulty}">${esc(labelize(task.difficulty))}</span></span></div>
    <div style="margin:8px 0 4px;font-size:11px;color:var(--text-3);">${esc(task.instruction||"")}</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;">${(task.tags||[]).map(t=>`<span class="tag-chip">${esc(t)}</span>`).join("")}</div>
    <div style="border-top:1px solid var(--border);padding-top:10px;margin-top:4px;">
      <div class="meta-row"><span class="meta-key">Baseline</span><span class="meta-val">${sum.baseline_passed}/${sum.baseline_total}</span></div>
      <div class="meta-row"><span class="meta-key">Best passed</span><span class="meta-val">${sum.best_passed}/${sum.baseline_total}</span></div>
      <div class="meta-row"><span class="meta-key">Regressions</span><span class="meta-val">${sum.regression_count||0}</span></div>
      <div class="meta-row"><span class="meta-key">Syntax errors</span><span class="meta-val">${sum.syntax_error_count||0}</span></div>
    </div>
    ${session.hint ? `<div style="background:var(--amber-dim);border:1px solid rgba(245,166,35,.3);border-radius:var(--radius);padding:10px 12px;font-size:12px;color:var(--amber);margin-top:10px;">${esc(session.hint)}</div>` : ""}
    ${act.commit_message ? `<div style="border-top:1px solid var(--border);padding-top:10px;margin-top:10px;"><div class="meta-row"><span class="meta-key">Last commit</span><span class="meta-val" style="font-size:11px">${esc(act.commit_message)}</span></div></div>` : ""}
  `;
}

/* ─── Benchmark tab ─────────────────────────────────────────── */
function renderBenchmark() {
  const bench = S.bootstrap?.benchmark;
  const metrics = S.bootstrap?.metrics;
  if (!bench) return;

  // Overview stats
  if (el.benchmarkOverview) {
    const dd = bench.difficulty_distribution || {};
    el.benchmarkOverview.innerHTML = [
      statCard("Tasks", bench.task_count || 0, "accent"),
      statCard("Easy / Med / Hard", `${dd.easy||0} / ${dd.medium||0} / ${dd.hard||0}`, ""),
      statCard("Baseline avg", fmt(bench.baseline?.average_score), "ok"),
    ].join("");
  }

  // Task table
  if (el.taskTableRoot) {
    const tasks = bench.tasks || [];
    el.taskTableRoot.innerHTML = `
      <table>
        <thead><tr><th>Task ID</th><th>Difficulty</th><th>Objective</th><th>Tags</th><th></th></tr></thead>
        <tbody>
          ${tasks.map(t => `
            <tr>
              <td style="font-family:var(--mono)">${esc(t.task_id)}</td>
              <td><span class="diff-badge diff-badge--${t.difficulty}">${esc(labelize(t.difficulty))}</span></td>
              <td style="color:var(--text-2)">${esc(t.objective || "")}</td>
              <td>${(t.tags||[]).map(x=>`<span class="tag-chip">${esc(x)}</span>`).join(" ")}</td>
              <td><button class="btn btn-ghost btn-sm" data-load-task="${esc(t.task_id)}">Load</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    el.taskTableRoot.querySelectorAll("[data-load-task]").forEach(btn => {
      btn.addEventListener("click", async e => {
        e.stopPropagation();
        switchTab("workbench");
        if (el.taskSelect) el.taskSelect.value = btn.dataset.loadTask;
        await resetEpisode(btn.dataset.loadTask);
      });
    });
  }

  // Baseline & recent episodes
  if (el.baselineRoot) {
    const baseline = bench.baseline || {};
    const results = baseline.results || [];
    el.baselineRoot.innerHTML = `
      <div style="padding:14px 18px;display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid var(--border);">
        <div><div style="font-size:10px;color:var(--text-3)">Model</div><div style="font-size:13px;font-weight:600">${esc(baseline.model||"Not recorded")}</div></div>
        <div><div style="font-size:10px;color:var(--text-3)">Average score</div><div style="font-size:13px;font-weight:700;color:var(--accent)">${fmt(baseline.average_score)}</div></div>
        <div><div style="font-size:10px;color:var(--text-3)">Source</div><div style="font-size:13px;font-weight:600">${esc(labelize(baseline.source||"missing"))}</div></div>
      </div>
      ${results.length ? `
        <table>
          <thead><tr><th>Task</th><th>Score</th><th>Success</th><th>Steps</th></tr></thead>
          <tbody>
            ${results.map(r => `
              <tr>
                <td style="font-family:var(--mono)">${esc(r.task_id||"—")}</td>
                <td style="color:var(--accent)">${fmt(r.score)}</td>
                <td><span class="dot dot--${r.success?'ok':'error'}"></span> ${r.success?"Yes":"No"}</td>
                <td>${r.steps??"-"}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      ` : '<div style="padding:14px 18px;font-size:12px;color:var(--text-3);">Run <code>python inference.py</code> to generate baseline data.</div>'}
    `;
  }

  if (el.recentEpisodesRoot) {
    const eps = metrics?.recent_episodes || [];
    el.recentEpisodesRoot.innerHTML = `
      <table>
        <thead><tr><th>Episode</th><th>Task</th><th>Solved</th><th>Reward</th><th>Peak pass</th></tr></thead>
        <tbody>
          ${eps.length ? eps.map(ep => `
            <tr>
              <td style="font-family:var(--mono);color:var(--text-3)">${esc(ep.episode_id?.slice(0,8)||"—")}</td>
              <td>${esc(ep.task_id)}</td>
              <td><span class="dot dot--${ep.solved?'ok':'idle'}"></span> ${ep.solved?"Yes":"No"}</td>
              <td style="color:${ep.reward>=0?'var(--green)':'var(--red)'}">${fmtSigned(ep.reward)}</td>
              <td>${fmtPct(ep.peak_pass_rate)}</td>
            </tr>
          `).join("") : '<tr><td colspan="5" style="text-align:center;color:var(--text-3);padding:14px">No completed episodes yet.</td></tr>'}
        </tbody>
      </table>
    `;
  }
}

/* ─── System tab ─────────────────────────────────────────────── */
function renderSystem() {
  const compliance = S.bootstrap?.compliance;
  if (!compliance) return;

  const v = compliance.validator_status || {};
  if (el.complianceRoot) {
    const items = [
      { label: "openenv.yaml", sub: "Spec definition file", ok: v.openenv_yaml_present },
      { label: "Dockerfile", sub: "Port 7860, non-root user", ok: v.dockerfile_present },
      { label: "inference.py", sub: "START/STEP/END structured log", ok: v.inference_script_present },
      { label: "HF Space UI", sub: "Web interface enabled", ok: v.web_interface_enabled },
      { label: "State endpoint", sub: "/state returns current episode", ok: true },
      { label: "Reset endpoint", sub: "POST /reset returns observation", ok: true },
    ];
    el.complianceRoot.innerHTML = items.map(item => `
      <div class="compliance-item">
        <div><div class="ci-label">${esc(item.label)}</div><div class="ci-sub">${esc(item.sub)}</div></div>
        <span class="ci-badge ${item.ok ? 'ok' : 'fail'}">${item.ok ? "Pass" : "Fail"}</span>
      </div>
    `).join("");
  }

  if (el.apiRoot) {
    const endpoints = [
      { method: "GET",  path: "/health",  desc: "Health check and server status" },
      { method: "POST", path: "/reset",   desc: "Start new episode, returns observation" },
      { method: "POST", path: "/step",    desc: "Submit patch action, returns (obs, reward, done, info)" },
      { method: "GET",  path: "/state",   desc: "Current episode state for OpenEnv spec" },
      { method: "GET",  path: "/tasks",   desc: "List all available graded tasks" },
      { method: "GET",  path: "/metrics", desc: "Telemetry — recent episodes and counters" },
    ];
    el.apiRoot.innerHTML = endpoints.map(e => `
      <div class="api-card">
        <div class="api-method ${e.method.toLowerCase()}">${e.method}</div>
        <div class="api-path">${esc(e.path)}</div>
        <div class="api-desc">${esc(e.desc)}</div>
      </div>
    `).join("");
  }

  if (el.specRoot) {
    const payload = {
      openenv: compliance.openenv || {},
      docker: compliance.docker || {},
      tasks: compliance.tasks || {},
      reward: compliance.reward || {},
    };
    el.specRoot.innerHTML = `<pre class="spec-json">${esc(JSON.stringify(payload, null, 2))}</pre>`;
  }
}

/* ─── Helpers ───────────────────────────────────────────────── */
function getSelectedRun() {
  const session = S.session;
  if (!session?.initialized) return null;
  if (S.selectedStep != null) {
    const r = (session.history||[]).find(r => r.step_index === S.selectedStep);
    if (r) return { ...r, kind: "step" };
  }
  if (session.latest_run) return { ...session.latest_run, kind: "step" };
  if (session.baseline) return {
    ...session.baseline,
    status: session.baseline.status,
    execution_status: session.baseline.status,
    reward_delta: 0,
    cumulative_reward: 0,
    diff_lines: 0,
    reward_breakdown: {},
    diff_preview: "",
    step_index: 0,
    kind: "baseline",
  };
  return null;
}

function statCard(label, value, tone) {
  return `<div class="stat-card ${tone}"><div class="stat-card-value">${esc(String(value))}</div><div class="stat-card-label">${esc(label)}</div></div>`;
}

function setText(key, val) {
  const e = key.startsWith("el.") ? eval(key) : el[key];
  if (e) e.textContent = String(val ?? "");
}

function chipTone(st) {
  if (!st) return "";
  if (["solved","success","improved","easy","healthy","active"].includes(st)) return "ok";
  if (["steady","medium","warning"].includes(st)) return "warn";
  if (["syntax_error","timeout","crash","runtime_error","regressed","hard","error"].includes(st)) return "error";
  if (["running"].includes(st)) return "running";
  return "";
}
function scoreClass(score) {
  if (score === undefined || score === null) return "";
  if (score >= 0.8) return "green";
  if (score <= 0.3) return "red";
  return "accent";
}
function scoreColor(score) {
  if (!score && score !== 0) return "var(--text)";
  if (score >= 0.8) return "var(--green)";
  if (score <= 0.3) return "var(--red)";
  return "var(--accent)";
}
function labelize(v) {
  return String(v||"").replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
}
function fmt(v) { return typeof v === "number" ? v.toFixed(3) : "—"; }
function fmtSigned(v) { return typeof v === "number" ? `${v>=0?"+":""}${v.toFixed(3)}` : "—"; }
function fmtPct(v) { return typeof v === "number" ? `${(v*100).toFixed(0)}%` : "—"; }
function fmtMs(v) { const n=Number(v||0); return n>=1000?`${(n/1000).toFixed(2)}s`:`${n.toFixed(0)}ms`; }

function esc(v) {
  return String(v??"")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

async function api(url, opts = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers||{}) },
    ...opts,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `${resp.status} ${resp.statusText}`);
  return data;
}

function renderFatal(msg) {
  document.body.innerHTML = `
    <div style="max-width:640px;margin:60px auto;padding:28px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-lg);font-family:var(--font);">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-3);margin-bottom:8px;">Bootstrap Error</div>
      <h2 style="font-size:18px;font-weight:700;color:var(--text);margin-bottom:12px;">CodeDebug-RL could not initialize</h2>
      <pre style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;font-size:12px;color:var(--text-2);white-space:pre-wrap;">${esc(msg)}</pre>
    </div>
  `;
}

/* CSS spin anim */
const style = document.createElement("style");
style.textContent = "@keyframes spin{to{transform:rotate(360deg)}}";
document.head.appendChild(style);

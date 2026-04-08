const appState = {
  bootstrap: null,
  session: null,
  selectedStep: null,
  submitting: false,
  editorDirty: false,
  lastError: null,
};

const el = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindEvents();
  await loadBootstrap();
});

function cacheElements() {
  el.heroTitle = document.getElementById("hero-title");
  el.heroSubtitle = document.getElementById("hero-subtitle");
  el.proofStrip = document.getElementById("proof-strip");
  el.heroTaskCount = document.getElementById("hero-task-count");
  el.heroBaselineScore = document.getElementById("hero-baseline-score");
  el.heroValidation = document.getElementById("hero-validation");
  el.summaryBar = document.getElementById("summary-bar");
  el.taskSelect = document.getElementById("task-select");
  el.difficultySelect = document.getElementById("difficulty-select");
  el.patchFormat = document.getElementById("patch-format");
  el.codeEditor = document.getElementById("code-editor");
  el.reasoningInput = document.getElementById("reasoning-input");
  el.commitInput = document.getElementById("commit-input");
  el.resetButton = document.getElementById("reset-button");
  el.submitButton = document.getElementById("submit-button");
  el.editorTaskLabel = document.getElementById("editor-task-label");
  el.editorDifficultyLabel = document.getElementById("editor-difficulty-label");
  el.runStatusChip = document.getElementById("run-status-chip");
  el.evaluationRoot = document.getElementById("evaluation-root");
  el.chartRoot = document.getElementById("chart-root");
  el.historyRoot = document.getElementById("history-root");
  el.metadataRoot = document.getElementById("metadata-root");
  el.benchmarkOverview = document.getElementById("benchmark-overview");
  el.taskTableRoot = document.getElementById("task-table-root");
  el.baselineRoot = document.getElementById("baseline-root");
  el.recentEpisodesRoot = document.getElementById("recent-episodes-root");
  el.complianceRoot = document.getElementById("compliance-root");
  el.apiRoot = document.getElementById("api-root");
  el.specRoot = document.getElementById("spec-root");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });

  el.codeEditor.addEventListener("input", () => {
    appState.editorDirty = true;
  });

  el.resetButton.addEventListener("click", async () => {
    await resetEpisode();
  });

  el.submitButton.addEventListener("click", async () => {
    await submitStep();
  });
}

async function loadBootstrap() {
  try {
    const data = await fetchJson("/ui/bootstrap");
    appState.bootstrap = data;
    appState.session = data.session;
    renderAll(false);

    if (!data.session || !data.session.initialized) {
      const mediumTask = pickDefaultTask(data.benchmark?.tasks || []);
      await resetEpisode(mediumTask?.task_id || null);
    }
  } catch (error) {
    appState.lastError = error.message;
    renderFatal(error.message);
  }
}

function pickDefaultTask(tasks) {
  const medium = tasks.find((task) => task.difficulty === "medium");
  return medium || tasks[0] || null;
}

async function refreshSession() {
  appState.session = await fetchJson("/ui/session");
}

async function refreshMetrics() {
  if (!appState.bootstrap) {
    return;
  }
  const metrics = await fetchJson("/metrics");
  appState.bootstrap.metrics = metrics;
}

async function resetEpisode(taskId = null) {
  setSubmitting(true, "Resetting...");
  appState.lastError = null;
  appState.selectedStep = null;

  const selectedTaskId = taskId || el.taskSelect.value || "";
  const difficulty = el.difficultySelect.value || "";
  const payload = {};

  if (selectedTaskId) {
    payload.task_id = selectedTaskId;
  } else if (difficulty) {
    payload.difficulty = difficulty;
  }

  try {
    await fetchJson("/reset", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    appState.editorDirty = false;
    el.reasoningInput.value = "";
    el.commitInput.value = "";
    await Promise.all([refreshSession(), refreshMetrics()]);
    renderAll(false);
  } catch (error) {
    appState.lastError = error.message;
    renderAll(true);
  } finally {
    setSubmitting(false);
  }
}

async function submitStep() {
  setSubmitting(true, "Running step...");
  appState.lastError = null;

  const payload = {
    action: {
      patched_code: el.codeEditor.value,
      patch_format: el.patchFormat.value,
    },
  };

  if (el.reasoningInput.value.trim()) {
    payload.action.reasoning = el.reasoningInput.value.trim();
  }
  if (el.commitInput.value.trim()) {
    payload.action.commit_message = el.commitInput.value.trim();
  }

  try {
    await fetchJson("/step", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    appState.editorDirty = false;
    await Promise.all([refreshSession(), refreshMetrics()]);
    renderAll(false);
  } catch (error) {
    appState.lastError = error.message;
    renderAll(true);
  } finally {
    setSubmitting(false);
  }
}

function setSubmitting(isSubmitting, label) {
  appState.submitting = isSubmitting;
  el.submitButton.disabled = isSubmitting;
  el.resetButton.disabled = isSubmitting;
  el.submitButton.textContent = isSubmitting ? label : "Run step";
}

function activateTab(tabName) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === tabName);
  });
}

function renderAll(preserveEditor) {
  if (!appState.bootstrap) {
    return;
  }

  renderHero();
  renderTaskSelect();
  renderSummary();
  renderWorkbench(preserveEditor);
  renderBenchmark();
  renderSystem();
}

function renderHero() {
  const { product, benchmark, compliance } = appState.bootstrap;
  const baseline = benchmark?.baseline || {};
  const validator = compliance?.validator_status || {};
  el.heroTitle.textContent = product?.title || "CodeDebug-RL";
  el.heroSubtitle.textContent = product?.subtitle || "";
  el.proofStrip.innerHTML = (product?.proof_points || [])
    .map((item) => `<span class="proof-pill">${escapeHtml(item)}</span>`)
    .join("");
  el.heroTaskCount.textContent = String(benchmark?.task_count ?? "-");
  el.heroBaselineScore.textContent = formatScore(baseline?.average_score);
  el.heroValidation.textContent = validator.openenv_yaml_present && validator.dockerfile_present && validator.inference_script_present
    ? "Ready"
    : "Incomplete";
}

function renderTaskSelect() {
  const tasks = appState.bootstrap?.benchmark?.tasks || [];
  const currentTaskId = appState.session?.task?.task_id || "";
  const options = [
    '<option value="">Random task</option>',
    ...tasks.map((task) => {
      const selected = task.task_id === currentTaskId ? " selected" : "";
      return `<option value="${escapeAttr(task.task_id)}"${selected}>${escapeHtml(task.task_id)} · ${escapeHtml(task.difficulty)}</option>`;
    }),
  ];
  el.taskSelect.innerHTML = options.join("");
}

function renderSummary() {
  const session = appState.session;
  const summary = session?.summary;
  const task = session?.task;

  if (!session || !session.initialized || !summary || !task) {
    el.summaryBar.innerHTML = `
      <div class="summary-primary">
        <div class="summary-title">
          <h2>No active episode</h2>
        </div>
      </div>
      <div class="summary-kpis">
        ${renderKpiCard("Score", "0.00")}
        ${renderKpiCard("Reward", "0.00")}
        ${renderKpiCard("Status", "Idle")}
      </div>
    `;
    setRunChip("Idle", "neutral");
    return;
  }

  el.summaryBar.innerHTML = `
    <div class="summary-primary">
      <div class="summary-title">
        <h2>${escapeHtml(task.task_id)}</h2>
        <span class="badge ${statusTone(summary.status)}">${escapeHtml(labelize(summary.status))}</span>
        <span class="badge info">${escapeHtml(task.difficulty)}</span>
      </div>
      <div class="summary-meta">${escapeHtml(task.instruction || "No task description available.")}</div>
      <div class="pill-row">
        ${(task.tags || []).map((tag) => `<span class="badge neutral">${escapeHtml(tag)}</span>`).join("")}
      </div>
    </div>
    <div class="summary-kpis">
      ${renderKpiCard("Score", formatScore(summary.current_score))}
      ${renderKpiCard("Best", formatScore(summary.best_score))}
      ${renderKpiCard("Reward", formatSigned(summary.latest_reward))}
      ${renderKpiCard("Cumulative", formatSigned(summary.cumulative_reward))}
      ${renderKpiCard("Pass rate", `${summary.current_passed}/${summary.current_total}`)}
      ${renderKpiCard("Steps", `${summary.step_index}/${summary.max_steps}`)}
    </div>
  `;

  setRunChip(labelize(summary.execution_status || summary.status), statusTone(summary.status));
}

function renderWorkbench(preserveEditor) {
  const session = appState.session;
  const task = session?.task;
  const summary = session?.summary;

  if (session?.initialized && (!preserveEditor || !appState.editorDirty)) {
    el.codeEditor.value = session.code?.current || "";
  }

  el.editorTaskLabel.textContent = task?.task_id || "No active task";
  el.editorDifficultyLabel.textContent = task?.difficulty ? labelize(task.difficulty) : "Idle";
  el.editorDifficultyLabel.className = `badge ${statusTone(task?.difficulty || "neutral")}`;

  if (task?.task_id) {
    el.taskSelect.value = task.task_id;
  }

  const selectedRun = getSelectedRun();
  renderEvaluation(selectedRun, summary);
  renderChart();
  renderHistory();
  renderMetadata(selectedRun);
}

function renderEvaluation(run, summary) {
  if (!run) {
    el.evaluationRoot.innerHTML = '<div class="empty-state">Reset the environment to load a task and inspect the baseline grader output.</div>';
    return;
  }

  const failures = run.failures || [];
  const rewardBreakdown = run.reward_breakdown || {};
  const diffPreview = run.diff_preview || "";
  const rewardRows = Object.entries(rewardBreakdown).length
    ? Object.entries(rewardBreakdown).map(([key, value]) => `<div class="segment"><span class="label">${escapeHtml(labelize(key))}</span><strong>${formatSigned(value)}</strong></div>`).join("")
    : '<div class="segment"><span class="label">Reward breakdown</span><strong>Baseline only</strong></div>';

  const logsHtml = [
    renderLogBlock("Stdout", run.stdout),
    renderLogBlock("Stderr", run.stderr),
    diffPreview ? renderLogBlock("Diff preview", diffPreview) : "",
  ].join("");

  el.evaluationRoot.innerHTML = `
    <div class="evaluation-card">
      ${appState.lastError ? `<div class="verdict-card"><h3>Last request error</h3><div class="subtle">${escapeHtml(appState.lastError)}</div></div>` : ""}
      <div class="verdict-row">
        <div class="verdict-card">
          <div class="eyebrow">${run.kind === "baseline" ? "Baseline run" : `Step ${run.step_index}`}</div>
          <h3>${escapeHtml(labelize(run.status))}</h3>
          <div class="subtle">${escapeHtml(run.execution_status || run.status)}</div>
          <div class="pill-row">
            <span class="badge ${statusTone(run.status)}">${escapeHtml(labelize(run.status))}</span>
            <span class="badge neutral">${escapeHtml(run.syntax_valid ? "Syntax valid" : "Syntax invalid")}</span>
            <span class="badge neutral">${escapeHtml(formatMs(run.duration_ms || 0))}</span>
          </div>
        </div>
        <div class="metric-row">
          ${renderMetricCard("Score", formatScore(run.score))}
          ${renderMetricCard("Reward", run.kind === "baseline" ? "0.00" : formatSigned(run.reward_delta))}
          ${renderMetricCard("Passed", `${run.passed}/${run.total}`)}
          ${renderMetricCard("Patch", run.kind === "baseline" ? "-" : `${run.diff_lines || 0} lines`)}
        </div>
      </div>

      <div class="segment-row">
        <div class="segment">
          <span class="label">Delta vs baseline</span>
          <strong>${formatSigned((run.score || 0) - (summary?.baseline_score || 0))}</strong>
        </div>
        <div class="segment">
          <span class="label">Cumulative reward</span>
          <strong>${formatSigned(run.cumulative_reward || 0)}</strong>
        </div>
        <div class="segment">
          <span class="label">Execution</span>
          <strong>${escapeHtml(labelize(run.execution_status || run.status))}</strong>
        </div>
      </div>

      <div class="code-card">
        <h3>Failure groups</h3>
        ${
          failures.length
            ? `<div class="failure-list">${failures.map((failure, index) => renderFailureCard(failure, index === 0)).join("")}</div>`
            : '<div class="empty-state">No failures captured for this run.</div>'
        }
      </div>

      <div class="code-card">
        <h3>Reward components</h3>
        <div class="segment-row">${rewardRows}</div>
      </div>

      <div class="log-card">
        <h3>Artifacts</h3>
        ${logsHtml || '<div class="empty-state">No stdout, stderr, or diff preview captured for this run.</div>'}
      </div>
    </div>
  `;
}

function renderChart() {
  const session = appState.session;
  if (!session?.initialized) {
    el.chartRoot.innerHTML = '<div class="chart-empty">A baseline reset populates score, reward, and trajectory data here.</div>';
    return;
  }

  const baselineScore = session.summary?.baseline_score || 0;
  const history = session.history || [];
  const points = [{ step: 0, score: baselineScore, reward: 0 }];

  history.forEach((run) => {
    points.push({
      step: run.step_index,
      score: run.score || 0,
      reward: run.cumulative_reward || 0,
    });
  });

  const svg = renderTrajectorySvg(points);
  el.chartRoot.innerHTML = `
    <div class="chart-shell">
      ${svg}
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#4f7cff"></span>Normalized score</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#16a34a"></span>Cumulative reward</span>
      </div>
    </div>
  `;
}

function renderHistory() {
  const session = appState.session;
  const history = session?.history || [];

  if (!history.length) {
    el.historyRoot.innerHTML = '<div class="empty-state">No steps yet. The baseline run is ready; submit a patch to start the episode timeline.</div>';
    return;
  }

  el.historyRoot.innerHTML = `
    <table class="history-table">
      <thead>
        <tr>
          <th>Step</th>
          <th>Status</th>
          <th>Score</th>
          <th>Reward</th>
          <th>Pass</th>
          <th>Duration</th>
          <th>Patch</th>
        </tr>
      </thead>
      <tbody>
        ${history.map((run) => `
          <tr data-step="${run.step_index}" class="${appState.selectedStep === run.step_index ? "is-selected" : ""}">
            <td>${run.step_index}</td>
            <td><span class="badge ${statusTone(run.status)}">${escapeHtml(labelize(run.status))}</span></td>
            <td>${formatScore(run.score)}</td>
            <td>${formatSigned(run.reward_delta)}</td>
            <td>${run.passed}/${run.total}</td>
            <td>${escapeHtml(formatMs(run.duration_ms))}</td>
            <td>${run.diff_lines || 0} lines</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  el.historyRoot.querySelectorAll("tbody tr").forEach((row) => {
    row.addEventListener("click", () => {
      const step = Number(row.dataset.step);
      appState.selectedStep = appState.selectedStep === step ? null : step;
      renderWorkbench(true);
    });
  });
}

function renderMetadata(selectedRun) {
  const session = appState.session;
  if (!session?.initialized) {
    el.metadataRoot.innerHTML = '<div class="empty-state">Task metadata and action context appear here after reset.</div>';
    return;
  }

  const task = session.task;
  const latestAction = session.latest_action || {};
  el.metadataRoot.innerHTML = `
    <div class="metadata-section">
      <div class="metadata-card">
        <h3>Task brief</h3>
        <div class="key-value">
          <strong>Instruction</strong><div>${escapeHtml(task.instruction || "No instruction available.")}</div>
          <strong>Tags</strong><div>${(task.tags || []).map((tag) => `<span class="badge neutral">${escapeHtml(tag)}</span>`).join(" ") || "<span class=\"subtle\">None</span>"}</div>
          <strong>Files</strong><div class="mono">${escapeHtml(task.source_filename)} · ${escapeHtml(task.test_filename)}</div>
          <strong>Episode</strong><div class="mono">${escapeHtml(session.episode_id || "-")}</div>
        </div>
      </div>
      <div class="metadata-card">
        <h3>Latest action</h3>
        <div class="key-value">
          <strong>Patch format</strong><div>${escapeHtml(labelize(latestAction.patch_format || "full_replace"))}</div>
          <strong>Commit</strong><div>${escapeHtml(latestAction.commit_message || "Not provided")}</div>
          <strong>Reasoning</strong><div>${escapeHtml(latestAction.reasoning || "Not provided")}</div>
          <strong>Expected impact</strong><div>${escapeHtml(latestAction.expected_test_impact || "Not provided")}</div>
        </div>
      </div>
      <div class="metadata-card">
        <h3>Episode counters</h3>
        <div class="segment-row">
          <div class="segment"><span class="label">Baseline</span><strong>${session.summary.baseline_passed}/${session.summary.baseline_total}</strong></div>
          <div class="segment"><span class="label">Best</span><strong>${session.summary.best_passed}/${session.summary.baseline_total}</strong></div>
          <div class="segment"><span class="label">Peak pass rate</span><strong>${formatPercent(session.summary.peak_pass_rate || 0)}</strong></div>
          <div class="segment"><span class="label">Regressions</span><strong>${session.summary.regression_count || 0}</strong></div>
          <div class="segment"><span class="label">Syntax errors</span><strong>${session.summary.syntax_error_count || 0}</strong></div>
          <div class="segment"><span class="label">Duplicate patches</span><strong>${session.summary.duplicate_patch_count || 0}</strong></div>
        </div>
      </div>
      ${
        selectedRun?.reasoning
          ? `<div class="metadata-card"><h3>Selected step reasoning</h3><pre class="trace-block">${escapeHtml(selectedRun.reasoning)}</pre></div>`
          : ""
      }
      ${
        session.hint
          ? `<div class="metadata-card"><h3>Progressive hint</h3><div>${escapeHtml(session.hint)}</div></div>`
          : ""
      }
    </div>
  `;
}

function renderBenchmark() {
  const benchmark = appState.bootstrap?.benchmark;
  const metrics = appState.bootstrap?.metrics;
  if (!benchmark) {
    return;
  }

  el.benchmarkOverview.innerHTML = [
    renderKpiCard("Tasks", String(benchmark.task_count || 0)),
    renderKpiCard("Difficulty ladder", `${benchmark.difficulty_distribution?.easy || 0}/${benchmark.difficulty_distribution?.medium || 0}/${benchmark.difficulty_distribution?.hard || 0}`),
    renderKpiCard("Baseline avg.", formatScore(benchmark.baseline?.average_score)),
    renderKpiCard("Runtime budget", `${benchmark.constraints?.max_inference_runtime_min || 20} min`),
  ].join("");

  el.taskTableRoot.innerHTML = `
    <table class="task-table">
      <thead>
        <tr>
          <th>Task</th>
          <th>Difficulty</th>
          <th>Objective</th>
          <th>Baseline</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${(benchmark.tasks || []).map((task) => `
          <tr data-task="${escapeAttr(task.task_id)}">
            <td class="mono">${escapeHtml(task.task_id)}</td>
            <td><span class="badge ${statusTone(task.difficulty)}">${escapeHtml(labelize(task.difficulty))}</span></td>
            <td>${escapeHtml(task.objective || "")}</td>
            <td>${formatScore(task.baseline_score)}</td>
            <td><button class="link-button" data-load-task="${escapeAttr(task.task_id)}">Load task</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  el.taskTableRoot.querySelectorAll("[data-load-task]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const taskId = button.dataset.loadTask;
      activateTab("workbench");
      el.taskSelect.value = taskId;
      await resetEpisode(taskId);
    });
  });

  const baseline = benchmark.baseline || {};
  const results = baseline.results || [];
  el.baselineRoot.innerHTML = `
    <div class="baseline-card">
      <div class="segment-row">
        <div class="segment"><span class="label">Baseline source</span><strong>${escapeHtml(labelize(baseline.source || "missing"))}</strong></div>
        <div class="segment"><span class="label">Model</span><strong>${escapeHtml(baseline.model || "Not recorded")}</strong></div>
        <div class="segment"><span class="label">Average score</span><strong>${formatScore(baseline.average_score)}</strong></div>
      </div>
      ${
        results.length
          ? `<table class="recent-table">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Score</th>
                  <th>Success</th>
                  <th>Steps</th>
                </tr>
              </thead>
              <tbody>
                ${results.map((result) => `
                  <tr>
                    <td class="mono">${escapeHtml(result.task_id || "-")}</td>
                    <td>${formatScore(result.score)}</td>
                    <td>${String(Boolean(result.success))}</td>
                    <td>${escapeHtml(String(result.steps ?? "-"))}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
            <div class="failure-list">
              ${results.map((result) => renderTranscriptCard(result)).join("")}
            </div>`
          : '<div class="empty-state">Run <span class="mono">python inference.py</span> to generate a live baseline artifact for this panel.</div>'
      }
    </div>
  `;

  const recentEpisodes = metrics?.recent_episodes || [];
  el.recentEpisodesRoot.innerHTML = `
    <table class="recent-table">
      <thead>
        <tr>
          <th>Episode</th>
          <th>Task</th>
          <th>Solved</th>
          <th>Reward</th>
          <th>Peak pass</th>
          <th>Duration</th>
        </tr>
      </thead>
      <tbody>
        ${
          recentEpisodes.length
            ? recentEpisodes.map((episode) => `
              <tr>
                <td class="mono">${escapeHtml(shortId(episode.episode_id))}</td>
                <td class="mono">${escapeHtml(episode.task_id)}</td>
                <td>${String(Boolean(episode.solved))}</td>
                <td>${formatSigned(episode.reward)}</td>
                <td>${formatPercent(episode.peak_pass_rate)}</td>
                <td>${escapeHtml(formatDuration(episode.duration_s))}</td>
              </tr>
            `).join("")
            : '<tr><td colspan="6" class="subtle">No completed episodes recorded yet.</td></tr>'
        }
      </tbody>
    </table>
  `;
}

function renderSystem() {
  const compliance = appState.bootstrap?.compliance;
  if (!compliance) {
    return;
  }

  const validator = compliance.validator_status || {};
  el.complianceRoot.innerHTML = [
    renderBooleanCard("openenv.yaml", validator.openenv_yaml_present),
    renderBooleanCard("Dockerfile", validator.dockerfile_present),
    renderBooleanCard("inference.py", validator.inference_script_present),
    renderBooleanCard("HF Space UI", validator.web_interface_enabled),
  ].join("");

  el.apiRoot.innerHTML = `
    ${renderSpecBlock("Endpoints", compliance.api)}
    ${renderSpecBlock("Required env", compliance.required_env)}
    ${renderSpecBlock("Typed models", compliance.typed_models)}
    ${renderSpecBlock("Validator status", compliance.validator_status)}
  `;

  el.specRoot.innerHTML = `
    ${renderSpecBlock("OpenEnv", compliance.openenv)}
    ${renderSpecBlock("Docker", compliance.docker)}
    ${renderSpecBlock("Tasks", compliance.tasks)}
    ${renderSpecBlock("Reward", compliance.reward)}
  `;
}

function getSelectedRun() {
  const session = appState.session;
  if (!session?.initialized) {
    return null;
  }

  if (appState.selectedStep != null) {
    const selected = (session.history || []).find((run) => run.step_index === appState.selectedStep);
    if (selected) {
      return { ...selected, kind: "step" };
    }
  }

  if (session.latest_run) {
    return { ...session.latest_run, kind: "step" };
  }

  if (session.baseline) {
    return {
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
  }

  return null;
}

function setRunChip(label, tone) {
  el.runStatusChip.textContent = label;
  el.runStatusChip.className = `status-chip ${tone}`;
}

function renderTrajectorySvg(points) {
  const width = 680;
  const height = 220;
  const padding = 26;
  const scoreRange = { min: 0, max: 1 };
  const rewardValues = points.map((point) => point.reward || 0);
  const rewardMin = Math.min(0, ...rewardValues);
  const rewardMax = Math.max(1, ...rewardValues);
  const maxStep = Math.max(1, ...points.map((point) => point.step));

  const scaleX = (step) => padding + ((width - padding * 2) * step) / maxStep;
  const scaleY = (value, min, max) => {
    if (max === min) {
      return height / 2;
    }
    return height - padding - ((height - padding * 2) * (value - min)) / (max - min);
  };

  const scoreLine = points.map((point) => `${scaleX(point.step)},${scaleY(point.score, scoreRange.min, scoreRange.max)}`).join(" ");
  const rewardLine = points.map((point) => `${scaleX(point.step)},${scaleY(point.reward, rewardMin, rewardMax)}`).join(" ");
  const markers = points.map((point) => `
    <circle cx="${scaleX(point.step)}" cy="${scaleY(point.score, scoreRange.min, scoreRange.max)}" r="4" fill="#4f7cff" />
    <circle cx="${scaleX(point.step)}" cy="${scaleY(point.reward, rewardMin, rewardMax)}" r="3" fill="#16a34a" />
  `).join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="220" role="img" aria-label="Score and reward trajectory">
      <rect x="0" y="0" width="${width}" height="${height}" rx="12" fill="#0f172a" stroke="#263247"></rect>
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#25324a"></line>
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#25324a"></line>
      <text x="${padding}" y="${padding - 8}" fill="#95a3b8" font-size="11">1.0</text>
      <text x="${padding}" y="${height - 8}" fill="#95a3b8" font-size="11">0.0</text>
      <polyline fill="none" stroke="#4f7cff" stroke-width="3" points="${scoreLine}"></polyline>
      <polyline fill="none" stroke="#16a34a" stroke-width="3" points="${rewardLine}"></polyline>
      ${markers}
    </svg>
  `;
}

function renderFailureCard(failure, openByDefault) {
  return `
    <details class="failure-card" ${openByDefault ? "open" : ""}>
      <summary class="failure-header">
        <div>
          <h4>${escapeHtml(failure.title || failure.test_name || "Failure")}</h4>
          <div class="subtle">${escapeHtml(failure.assertion || failure.root_cause || "No assertion summary")}</div>
        </div>
        <span class="badge ${statusTone(failure.severity)}">${escapeHtml(labelize(failure.severity || "failure"))}</span>
      </summary>
      <div class="failure-body">
        ${failure.root_cause ? `<div><strong>Root cause</strong><div class="subtle">${escapeHtml(failure.root_cause)}</div></div>` : ""}
        ${failure.source_line ? `<div><strong>Source line</strong><div class="mono">${failure.source_line}</div></div>` : ""}
        ${failure.trace_excerpt ? `<pre class="trace-block">${escapeHtml(failure.trace_excerpt)}</pre>` : ""}
      </div>
    </details>
  `;
}

function renderTranscriptCard(result) {
  if (!result.transcript || !result.transcript.length) {
    return "";
  }
  return `
    <details class="failure-card">
      <summary class="failure-header">
        <div>
          <h4>${escapeHtml(result.task_id || "Transcript")}</h4>
          <div class="subtle">Structured baseline stdout artifact</div>
        </div>
        <span class="badge neutral">Transcript</span>
      </summary>
      <div class="failure-body">
        <pre class="trace-block">${escapeHtml(result.transcript.join("\n"))}</pre>
      </div>
    </details>
  `;
}

function renderLogBlock(title, content) {
  if (!content) {
    return "";
  }
  return `
    <details class="failure-card">
      <summary class="failure-header">
        <div><h4>${escapeHtml(title)}</h4></div>
        <span class="badge neutral">Expand</span>
      </summary>
      <div class="failure-body">
        <pre class="log-block">${escapeHtml(content)}</pre>
      </div>
    </details>
  `;
}

function renderSpecBlock(title, payload) {
  return `
    <div class="spec-block">
      <h3>${escapeHtml(title)}</h3>
      <pre class="trace-block">${escapeHtml(JSON.stringify(payload || {}, null, 2))}</pre>
    </div>
  `;
}

function renderMetricCard(label, value) {
  return `<div class="metric-card"><span class="label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderKpiCard(label, value) {
  return `<div class="kpi-card"><span class="label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderBooleanCard(label, value) {
  return renderKpiCard(label, value ? "Yes" : "No");
}

function renderFatal(message) {
  document.body.innerHTML = `
    <div style="max-width:760px;margin:48px auto;padding:24px;border:1px solid #263247;border-radius:12px;background:#111827;color:#e5edf7;font:14px/1.5 Segoe UI,system-ui,sans-serif;">
      <div style="font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#95a3b8;">UI bootstrap error</div>
      <h1 style="margin:8px 0 12px;">CodeDebug-RL could not initialize</h1>
      <pre style="white-space:pre-wrap;background:#0f172a;border:1px solid #263247;border-radius:8px;padding:14px;">${escapeHtml(message)}</pre>
    </div>
  `;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

function statusTone(status) {
  if (["solved", "success", "improved", "easy"].includes(status)) {
    return "success";
  }
  if (["medium", "warning", "steady"].includes(status)) {
    return "warning";
  }
  if (["syntax_error", "timeout", "crash", "runtime_error", "regressed", "hard"].includes(status)) {
    return "danger";
  }
  if (["active", "full_replace", "unified_diff"].includes(status)) {
    return "info";
  }
  return "neutral";
}

function labelize(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function formatSigned(value) {
  return typeof value === "number" ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}` : "-";
}

function formatMs(value) {
  const number = Number(value || 0);
  if (number >= 1000) {
    return `${(number / 1000).toFixed(2)}s`;
  }
  return `${number.toFixed(0)}ms`;
}

function formatPercent(value) {
  const number = Number(value || 0);
  return `${(number * 100).toFixed(0)}%`;
}

function formatDuration(value) {
  const number = Number(value || 0);
  return `${number.toFixed(2)}s`;
}

function shortId(value) {
  return String(value || "").slice(0, 8);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/"/g, "&quot;");
}

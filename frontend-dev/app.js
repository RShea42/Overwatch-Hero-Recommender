// Post-submission DEVELOPMENT frontend: points at the separately-named dev
// Modal API, never the submitted production API.
const API_BASE_URL = "https://rshea42--overwatch-hero-recommender-dev-fastapi-app.modal.run";
const RECOMMEND_URL = `${API_BASE_URL}/recommend`;

const roleSelect = document.getElementById("role");
const mainSelect = document.getElementById("main");
const secondarySelect = document.getElementById("secondary");
const rankSelect = document.getElementById("rank");
const inputSelect = document.getElementById("input");
const regionSelect = document.getElementById("region");
const form = document.getElementById("pool-form");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");

const statusEmpty = document.getElementById("status-empty");
const statusLoading = document.getElementById("status-loading");
const statusError = document.getElementById("status-error");
const results = document.getElementById("results");

const contextRole = document.getElementById("context-role");
const contextRank = document.getElementById("context-rank");
const contextInput = document.getElementById("context-input");
const contextRegion = document.getElementById("context-region");

const poolHeading = document.getElementById("pool-heading");
const finalPoolContainer = document.getElementById("final-pool");

const coverageBefore = document.getElementById("coverage-before");
const coverageAfter = document.getElementById("coverage-after");
const statResolvedValue = document.getElementById("stat-resolved-value");
const statSeverityValue = document.getElementById("stat-severity-value");
const coveragePositive = document.getElementById("coverage-positive");
const coverageWorst = document.getElementById("coverage-worst");
const remainingToggle = document.getElementById("remaining-toggle");
const remainingSummary = document.getElementById("remaining-summary");
const remainingList = document.getElementById("remaining-list");
const resolvedToggle = document.getElementById("resolved-toggle");
const resolvedSummary = document.getElementById("resolved-summary");
const resolvedList = document.getElementById("resolved-list");

const wrGrid = document.getElementById("wr-grid");

const unknownCaveat = document.getElementById("unknown-caveat");
const unknownList = document.getElementById("unknown-list");

const searchDetails = document.getElementById("search-details");
const searchDetailsBody = document.getElementById("search-details-body");

function populateHeroSelects() {
  const role = roleSelect.value;
  const heroes = [...(HEROES_BY_ROLE[role] || [])].sort((a, b) => a.name.localeCompare(b.name));

  mainSelect.innerHTML = "";
  secondarySelect.innerHTML = '<option value="">None</option>';

  for (const hero of heroes) {
    const opt1 = document.createElement("option");
    opt1.value = hero.id;
    opt1.textContent = hero.name;
    mainSelect.appendChild(opt1);

    const opt2 = document.createElement("option");
    opt2.value = hero.id;
    opt2.textContent = hero.name;
    secondarySelect.appendChild(opt2);
  }

  // Default starting hero for Damage, matching the already-verified test
  // case; otherwise just leave the first option selected.
  if (role === "Damage") {
    mainSelect.value = "soldier-76";
  }

  // Role changed: the previously selected Secondary may no longer exist in
  // this role's list (or may now equal Main) - reset it rather than leave a
  // stale/invalid selection.
  secondarySelect.value = "";
  updateActionLabel();
}

function updateActionLabel() {
  submitBtn.textContent = secondarySelect.value ? "Complete My Pool" : "Build My Pool";
}

function enforceMainSecondaryDistinct() {
  if (secondarySelect.value && secondarySelect.value === mainSelect.value) {
    secondarySelect.value = "";
    updateActionLabel();
  }
}

function setView(view) {
  statusEmpty.hidden = view !== "empty";
  statusLoading.hidden = view !== "loading";
  statusError.hidden = view !== "error";
  results.hidden = view !== "results";
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function clearResults() {
  finalPoolContainer.innerHTML = "";
  remainingList.innerHTML = "";
  resolvedList.innerHTML = "";
  unknownList.innerHTML = "";
  wrGrid.innerHTML = "";
  coveragePositive.hidden = true;
  coverageWorst.hidden = true;
  unknownCaveat.hidden = true;
}

function titleCase(value) {
  if (!value) return "";
  return value.charAt(0) + value.slice(1).toLowerCase();
}

function formatPercent(value) {
  if (value === null || value === undefined) return "N/A";
  return `${value}%`;
}

function heroCard(id, slotLabel, variant) {
  const card = document.createElement("div");
  card.className = `final-hero-card final-hero-card--${variant}`;
  const label = document.createElement("span");
  label.className = "slot-label";
  label.textContent = slotLabel;
  const name = document.createElement("span");
  name.className = "hero-name";
  name.textContent = heroName(id);
  card.appendChild(label);
  card.appendChild(name);
  return card;
}

function matchupChip(opponentId, value) {
  const span = document.createElement("span");
  span.className = "chip chip-matchup";
  span.textContent = `${heroName(opponentId)} (${value})`;
  return span;
}

function plainChip(text) {
  const span = document.createElement("span");
  span.className = "chip";
  span.textContent = text;
  return span;
}

function renderPool(payload) {
  finalPoolContainer.innerHTML = "";

  if (payload.mode === "builder") {
    poolHeading.textContent = "Your Recommended Pool";
    finalPoolContainer.appendChild(heroCard(payload.main, "Your Main", "main"));
    const backups = [payload.backup_a, payload.backup_b].sort((a, b) =>
      heroName(a).localeCompare(heroName(b))
    );
    backups.forEach((id) => finalPoolContainer.appendChild(heroCard(id, "Recommended", "recommended")));
  } else {
    poolHeading.textContent = "Your Completed Pool";
    finalPoolContainer.appendChild(heroCard(payload.main, "Your Main", "main"));
    finalPoolContainer.appendChild(heroCard(payload.secondary, "Your Secondary", "given"));
    finalPoolContainer.appendChild(
      heroCard(payload.recommended_tertiary, "Recommended", "recommended")
    );
  }
}

function renderCoverage(payload) {
  coverageBefore.textContent = payload.original_vulnerability_count;
  coverageAfter.textContent = payload.final_residual_vulnerability_count;
  statResolvedValue.textContent = payload.vulnerabilities_resolved_count;
  statSeverityValue.textContent = payload.total_residual_severity;

  if (payload.no_known_unfavorable_matchups_remain) {
    coveragePositive.hidden = false;
    coverageWorst.hidden = true;
  } else if (payload.worst_remaining_matchup) {
    coverageWorst.hidden = false;
    coverageWorst.textContent = `Worst remaining matchup: ${heroName(
      payload.worst_remaining_matchup.opponent
    )} (${payload.worst_remaining_matchup.value})`;
  }

  remainingList.innerHTML = "";
  const remaining = payload.remaining_vulnerabilities || [];
  remainingSummary.textContent = `Remaining vulnerabilities (${remaining.length})`;
  if (remaining.length === 0) {
    remainingList.appendChild(plainChip("None"));
  } else {
    remaining.forEach((v) => remainingList.appendChild(matchupChip(v.opponent, v.value)));
  }

  const originalByOpponent = {};
  (payload.original_vulnerabilities || []).forEach((v) => {
    originalByOpponent[v.opponent] = v.value;
  });

  resolvedList.innerHTML = "";
  const resolved = payload.vulnerabilities_resolved_opponents || [];
  resolvedSummary.textContent = `Resolved matchups (${resolved.length})`;
  if (resolved.length === 0) {
    resolvedList.appendChild(plainChip("None"));
  } else {
    resolved.forEach((opponentId) => {
      const originalValue = originalByOpponent[opponentId];
      resolvedList.appendChild(
        plainChip(
          originalValue === undefined
            ? heroName(opponentId)
            : `${heroName(opponentId)} (was ${originalValue})`
        )
      );
    });
  }
}

function wrStatBox(label, value, emphasize) {
  const box = document.createElement("div");
  box.className = `stat-box winrate${emphasize ? " winrate-emphasized" : ""}`;
  const l = document.createElement("span");
  l.className = "stat-label";
  l.textContent = label;
  const v = document.createElement("span");
  v.className = "stat-value";
  v.textContent = formatPercent(value);
  box.appendChild(l);
  box.appendChild(v);
  return box;
}

function renderContextualWR(payload) {
  wrGrid.innerHTML = "";
  if (payload.mode === "builder") {
    wrGrid.appendChild(
      wrStatBox(`${heroName(payload.backup_a)} Win Rate`, payload.contextual_wr.backup_a, true)
    );
    wrGrid.appendChild(
      wrStatBox(`${heroName(payload.backup_b)} Win Rate`, payload.contextual_wr.backup_b, true)
    );
  } else {
    wrGrid.appendChild(
      wrStatBox(
        `${heroName(payload.secondary)} Win Rate (reference)`,
        payload.contextual_wr.secondary,
        false
      )
    );
    wrGrid.appendChild(
      wrStatBox(
        `${heroName(payload.recommended_tertiary)} Win Rate (recommended)`,
        payload.contextual_wr.tertiary,
        true
      )
    );
  }
}

function renderUnknownCaveat(payload) {
  const caveats = payload.unknown_data_caveats || [];
  if (caveats.length === 0) {
    unknownCaveat.hidden = true;
    return;
  }
  unknownCaveat.hidden = false;
  unknownList.innerHTML = "";
  caveats.forEach((c) => {
    const missingNames = (c.backups_missing_data || []).map(heroName).join(", ");
    unknownList.appendChild(
      plainChip(`${heroName(c.opponent)} — no data from ${missingNames || "recommended backups"}`)
    );
  });
}

function renderSearchDetails(payload) {
  const size = payload.search ? payload.search.candidate_pool_size : null;
  const finalists = payload.search ? payload.search.pareto_frontier_size : null;
  searchDetailsBody.textContent =
    size !== null && finalists !== null
      ? `Evaluated ${size} complete pools · ${finalists} finalists compared.`
      : "";
}

function renderResults(payload) {
  contextRole.textContent = `Role: ${titleCase(payload.role)}`;
  contextRank.textContent = `Rank: ${payload.context.rank}`;
  contextInput.textContent = `Input: ${payload.context.input}`;
  contextRegion.textContent = `Region: ${payload.context.region}`;

  renderPool(payload);
  renderCoverage(payload);
  renderContextualWR(payload);
  renderUnknownCaveat(payload);
  renderSearchDetails(payload);

  setView("results");
}

async function handleSubmit(event) {
  event.preventDefault();
  showFormError("");

  const main = mainSelect.value;
  const secondary = secondarySelect.value;

  if (!main) {
    showFormError("Please select a Main.");
    return;
  }
  if (secondary && secondary === main) {
    showFormError("Secondary must be different from Main.");
    return;
  }

  const body = {
    role: roleSelect.value,
    main,
    secondary: secondary || null,
    rank: rankSelect.value,
    input: inputSelect.value,
    region: regionSelect.value,
  };

  submitBtn.disabled = true;
  clearResults();
  setView("loading");

  try {
    const response = await fetch(RECOMMEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let message = `Request failed with status ${response.status}.`;
      try {
        const errorPayload = await response.json();
        message = extractErrorMessage(errorPayload, message);
      } catch (_) {
        // response body wasn't JSON; keep the generic message
      }
      statusError.textContent = message;
      setView("error");
      return;
    }

    const payload = await response.json();
    renderResults(payload);
  } catch (err) {
    statusError.textContent =
      "Could not reach the recommendation API. Check your connection and try again.";
    setView("error");
  } finally {
    submitBtn.disabled = false;
  }
}

function extractErrorMessage(errorPayload, fallback) {
  const detail = errorPayload && errorPayload.detail;
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (typeof detail === "object" && detail.message) return detail.message;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return fallback;
}

populateHeroSelects();
roleSelect.addEventListener("change", populateHeroSelects);
mainSelect.addEventListener("change", enforceMainSecondaryDistinct);
secondarySelect.addEventListener("change", () => {
  enforceMainSecondaryDistinct();
  updateActionLabel();
});
form.addEventListener("submit", handleSubmit);

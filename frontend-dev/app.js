// Post-submission DEVELOPMENT frontend: points at the separately-named dev
// Modal API, never the submitted production API.
const API_BASE_URL = "https://rshea42--overwatch-hero-recommender-dev-fastapi-app.modal.run";
const RECOMMEND_URL = `${API_BASE_URL}/recommend`;

const roleSelect = document.getElementById("role");
const hero1Select = document.getElementById("hero1");
const hero2Select = document.getElementById("hero2");
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
const stepsContainer = document.getElementById("steps");
const finalPoolContainer = document.getElementById("final-pool");
const finalNegativeEl = document.getElementById("final-negative");

function populateHeroSelects() {
  const role = roleSelect.value;
  const heroes = [...(HEROES_BY_ROLE[role] || [])].sort((a, b) => a.name.localeCompare(b.name));

  hero1Select.innerHTML = "";
  hero2Select.innerHTML = '<option value="">None</option>';

  for (const hero of heroes) {
    const opt1 = document.createElement("option");
    opt1.value = hero.id;
    opt1.textContent = hero.name;
    hero1Select.appendChild(opt1);

    const opt2 = document.createElement("option");
    opt2.value = hero.id;
    opt2.textContent = hero.name;
    hero2Select.appendChild(opt2);
  }

  // Default starting hero for Damage, matching the already-verified test
  // case; otherwise just leave the first option selected.
  if (role === "Damage") {
    hero1Select.value = "soldier-76";
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

function heroPill(id, kind) {
  const span = document.createElement("span");
  span.className = `hero-pill ${kind}`;
  span.textContent = heroName(id);
  return span;
}

function statBox(label, value, kind) {
  const box = document.createElement("div");
  box.className = `stat-box ${kind}`;
  const l = document.createElement("span");
  l.className = "stat-label";
  l.textContent = label;
  const v = document.createElement("span");
  v.className = "stat-value";
  v.textContent = value;
  box.appendChild(l);
  box.appendChild(v);
  return box;
}

function heroChip(id) {
  const span = document.createElement("span");
  span.className = "chip";
  span.textContent = heroName(id);
  return span;
}

function matchupToggle(label, heroIds) {
  const details = document.createElement("details");
  details.className = "matchup-toggle";

  const summary = document.createElement("summary");
  summary.textContent = `${label} (${heroIds.length}) – view matchups`;
  details.appendChild(summary);

  const content = document.createElement("div");
  content.className = "chip-list";

  if (heroIds.length === 0) {
    const empty = document.createElement("span");
    empty.className = "chip-list-empty";
    empty.textContent = "None — full known coverage";
    content.appendChild(empty);
  } else {
    [...heroIds]
      .sort((a, b) => heroName(a).localeCompare(heroName(b)))
      .forEach((id) => content.appendChild(heroChip(id)));
  }

  details.appendChild(content);
  return details;
}

function renderStep(stepNumber, rec, isTwoHeroInput) {
  const card = document.createElement("div");
  card.className = "step-card";

  const heading = document.createElement("h4");
  heading.textContent = isTwoHeroInput
    ? "Recommended Hero #3"
    : `Recommended Hero #${stepNumber}`;
  card.appendChild(heading);

  const rosterRow = document.createElement("div");
  rosterRow.className = "step-roster";
  rec.roster_before.forEach((id) => rosterRow.appendChild(heroPill(id, "existing")));
  const arrow = document.createElement("span");
  arrow.className = "arrow";
  arrow.textContent = "→";
  rosterRow.appendChild(arrow);
  rosterRow.appendChild(heroPill(rec.recommended_hero, "recommended"));
  card.appendChild(rosterRow);

  const grid = document.createElement("div");
  grid.className = "stat-grid";
  grid.appendChild(statBox("Vulnerabilities Before", rec.vulnerabilities_before, "coverage"));
  grid.appendChild(statBox("Vulnerabilities Resolved", rec.vulnerabilities_resolved, "coverage"));
  grid.appendChild(statBox("Remaining Unfavorable", rec.remaining_negative_count, "coverage"));
  grid.appendChild(
    statBox("Contextual Win Rate", formatPercent(rec.contextual_winrate), "winrate")
  );
  grid.appendChild(
    statBox("Contextual Pick Rate", formatPercent(rec.contextual_pickrate), "pickrate")
  );
  grid.appendChild(
    statBox("Contextual Ban Rate", formatPercent(rec.contextual_banrate), "banrate")
  );
  card.appendChild(grid);

  const matchupSection = document.createElement("div");
  matchupSection.className = "matchup-toggles";
  matchupSection.appendChild(
    matchupToggle("Vulnerabilities Before", rec.vulnerability_heroes_before)
  );
  matchupSection.appendChild(
    matchupToggle("Vulnerabilities Resolved", rec.vulnerability_heroes_resolved)
  );
  matchupSection.appendChild(
    matchupToggle("Remaining Unfavorable", rec.remaining_negative_heroes)
  );
  card.appendChild(matchupSection);

  return card;
}

function formatPercent(value) {
  if (value === null || value === undefined) return "N/A";
  return `${value}%`;
}

function renderResults(payload, isTwoHeroInput) {
  contextRole.textContent = `Role: ${payload.role}`;
  contextRank.textContent = `Rank: ${payload.rank}`;
  contextInput.textContent = `Input: ${payload.input}`;
  contextRegion.textContent = `Region: ${payload.region}`;

  stepsContainer.innerHTML = "";
  payload.recommendations.forEach((rec, idx) => {
    stepsContainer.appendChild(renderStep(idx + 2, rec, isTwoHeroInput && idx === 0));
  });

  finalPoolContainer.innerHTML = "";
  const labels = ["Starting Hero", "Recommended #2", "Recommended #3"];
  payload.final_roster.forEach((id, idx) => {
    const card = document.createElement("div");
    card.className = "final-hero-card";
    const label = document.createElement("span");
    label.className = "slot-label";
    label.textContent = labels[idx] || `Slot ${idx + 1}`;
    const name = document.createElement("span");
    name.className = "hero-name";
    name.textContent = heroName(id);
    card.appendChild(label);
    card.appendChild(name);
    finalPoolContainer.appendChild(card);
  });

  const lastRec = payload.recommendations[payload.recommendations.length - 1];
  finalNegativeEl.innerHTML = `Final remaining unfavorable matchups: <strong>${lastRec.remaining_negative_count}</strong>`;

  setView("results");
}

async function handleSubmit(event) {
  event.preventDefault();
  showFormError("");

  const hero1 = hero1Select.value;
  const hero2 = hero2Select.value;

  if (!hero1) {
    showFormError("Please select a starting hero.");
    return;
  }
  if (hero2 && hero2 === hero1) {
    showFormError("The second hero must be different from the first.");
    return;
  }

  const heroes = hero2 ? [hero1, hero2] : [hero1];
  const body = {
    heroes,
    role: roleSelect.value,
    rank: rankSelect.value,
    input: inputSelect.value,
    region: regionSelect.value,
  };

  submitBtn.disabled = true;
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
    renderResults(payload, heroes.length === 2);
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
form.addEventListener("submit", handleSubmit);

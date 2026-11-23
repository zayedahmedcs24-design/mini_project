/**
 * Interactive Leaflet Web Viewer for Flood Segmentation & Geospatial Exposure
 * Offline-first, vendored Leaflet, dynamic layer stacking and exposure metrics.
 */

let map = null;
let currentChipId = null;
let allChips = [];
let baseOsmTileLayer = null;

// Layer registry
const layers = {
  sar_vv: { name: "SAR VV Amplitude", active: true, opacity: 1.0, overlay: null },
  ground_truth: { name: "Ground Truth Mask", active: true, opacity: 0.75, overlay: null },
  prediction: { name: "U-Net Ensemble (p>0.50)", active: true, opacity: 0.75, overlay: null },
  prediction_deeplab: { name: "DeepLabV3+ Ensemble (p>0.50)", active: false, opacity: 0.75, overlay: null },
  entropy: { name: "Predictive Uncertainty (Entropy)", active: false, opacity: 0.85, overlay: null },
  error: { name: "Error Analysis (FP/FN)", active: false, opacity: 0.80, overlay: null }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadIndex();
  setupLayerControls();
});

function initMap() {
  map = L.map("map", {
    zoomControl: true,
    attributionControl: false,
    minZoom: 4,
    maxZoom: 18
  }).setView([0, 0], 2);

  // Optional online basemap layer
  try {
    baseOsmTileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      opacity: 0.35
    });
    // Check basemap checkbox
    const bmCheck = document.getElementById("toggle-basemap");
    if (bmCheck && bmCheck.checked) {
      baseOsmTileLayer.addTo(map);
    }
    if (bmCheck) {
      bmCheck.addEventListener("change", (e) => {
        if (e.target.checked) {
          baseOsmTileLayer.addTo(map);
        } else {
          map.removeLayer(baseOsmTileLayer);
        }
      });
    }
  } catch (err) {
    console.warn("Basemap unavailable in offline mode.", err);
  }
}

async function loadIndex() {
  try {
    const resp = await fetch("data/index.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    allChips = data.chips;

    const selectEl = document.getElementById("chip-select");
    selectEl.innerHTML = "";

    // Group chips by their group field
    const groups = {};
    allChips.forEach(c => {
      if (!groups[c.group]) groups[c.group] = [];
      groups[c.group].push(c);
    });

    Object.keys(groups).forEach(gName => {
      const optGroup = document.createElement("optgroup");
      optGroup.label = gName;
      groups[gName].forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.chip_id;
        const expBadge = c.has_exposure ? " [Exposure]" : "";
        opt.textContent = `${c.chip_id} (IoU: ${c.unet_iou}%)${expBadge}`;
        optGroup.appendChild(opt);
      });
      selectEl.appendChild(optGroup);
    });

    selectEl.addEventListener("change", (e) => {
      loadChip(e.target.value);
    });

    // Check URL parameters for chip and layer selection
    const urlParams = new URLSearchParams(window.location.search);
    const chipParam = urlParams.get("chip");
    const layersParam = urlParams.get("layers");

    if (layersParam) {
      const activeKeys = layersParam.split(",").map(k => k.trim());
      Object.keys(layers).forEach(k => {
        layers[k].active = activeKeys.includes(k);
        const chk = document.getElementById(`chk-${k}`);
        if (chk) chk.checked = layers[k].active;
      });
    }

    // Default load chip
    let targetChip = null;
    if (chipParam && allChips.some(c => c.chip_id === chipParam)) {
      targetChip = chipParam;
    } else if (allChips.length > 0) {
      const defaultChip = allChips.find(c => c.chip_id === "Bolivia_290290") || allChips[0];
      targetChip = defaultChip.chip_id;
    }

    if (targetChip) {
      selectEl.value = targetChip;
      loadChip(targetChip);
    }
  } catch (err) {
    console.error("Failed to load data/index.json:", err);
    document.getElementById("chip-meta-container").innerHTML = `<div style="color:var(--accent-red);padding:10px;">Error loading index.json</div>`;
  }
}

async function loadChip(chipId) {
  currentChipId = chipId;

  try {
    const resp = await fetch(`data/${chipId}/meta.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const meta = await resp.json();

    // 1. Update Split Banner
    const bannerEl = document.getElementById("split-banner");
    if (bannerEl) {
      bannerEl.innerHTML = `<span><strong>SPLIT CONTEXT:</strong> ${meta.banner_text}</span> <span class="badge">${meta.split_name}</span>`;
    }

    // 2. Update Chip Meta Bar
    document.getElementById("meta-centroid").textContent = `${meta.centroid.lat}, ${meta.centroid.lon}`;
    document.getElementById("meta-utm").textContent = meta.utm_crs;
    document.getElementById("meta-flood-pct").textContent = `${meta.flood_fraction_valid_pct}%`;
    document.getElementById("meta-ignore-pct").textContent = `${meta.ignore_fraction_pct}%`;

    // 3. Update Model Performance Metrics Card
    updateMetricsCard(meta.metrics);

    // 4. Update Exposure Card
    updateExposureCard(meta.exposure);

    // 5. Update Map Layer Overlays
    const bounds = meta.leaflet_bounds;
    Object.keys(layers).forEach(layerKey => {
      const cfg = layers[layerKey];
      const imgUrl = `data/${chipId}/${layerKey}.png`;

      if (cfg.overlay) {
        map.removeLayer(cfg.overlay);
        cfg.overlay = null;
      }

      cfg.overlay = L.imageOverlay(imgUrl, bounds, {
        opacity: cfg.opacity,
        interactive: false
      });

      if (cfg.active) {
        cfg.overlay.addTo(map);
      }
    });

    // Fit map to chip bounds
    map.fitBounds(bounds, { padding: [30, 30] });

  } catch (err) {
    console.error(`Error loading chip ${chipId}:`, err);
  }
}

function updateMetricsCard(metrics) {
  const container = document.getElementById("metrics-container");
  if (!metrics) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:11px;">No metrics available</div>`;
    return;
  }

  const u = metrics.unet_resnet34 || {};
  const d = metrics.deeplabv3p_resnet34 || {};

  container.innerHTML = `
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="m-title">U-Net Flood IoU</div>
        <div class="m-val">${u.iou !== undefined ? u.iou + "%" : "-"}</div>
        <div class="m-sub">F1: ${u.f1 || "-"}% | Rec: ${u.recall || "-"}%</div>
      </div>
      <div class="metric-box">
        <div class="m-title">DeepLabV3+ Flood IoU</div>
        <div class="m-val" style="color:var(--accent-green);">${d.iou !== undefined ? d.iou + "%" : "-"}</div>
        <div class="m-sub">F1: ${d.f1 || "-"}% | Rec: ${d.recall || "-"}%</div>
      </div>
    </div>
  `;
}

function updateExposureCard(exposure) {
  const card = document.getElementById("exposure-container");

  if (!exposure || !exposure.available) {
    card.innerHTML = `
      <div class="exposure-card">
        <div class="exposure-card-header">
          <span>Disaster Exposure Assessment</span>
          <span style="color:var(--text-muted);font-size:10px;">N/A</span>
        </div>
        <p style="color:var(--text-muted);font-size:11px;line-height:1.4;">
          ${exposure && exposure.note ? exposure.note : "Exposure analysis not computed for this scene."}
        </p>
      </div>
    `;
    return;
  }

  const gt = exposure.ground_truth || {};
  const u = exposure.unet_resnet34 || {};
  const popTotal = exposure.total_footprint_population !== undefined ? exposure.total_footprint_population.toLocaleString(undefined, {maximumFractionDigits: 1}) : "-";

  function formatRoad(val, status) {
    if (val === null || val === undefined) {
      return `<span style="color:#94a3b8;font-style:italic;" title="No roads inside bounding box in extract">null (no roads in extract)</span>`;
    }
    return `${val.toFixed(2)} km`;
  }

  card.innerHTML = `
    <div class="exposure-card">
      <div class="exposure-card-header">
        <span>Disaster Exposure Assessment</span>
        <span style="color:var(--text-main);font-size:11px;">Footprint Pop: <strong>${popTotal}</strong></span>
      </div>
      <table class="exposure-table">
        <thead>
          <tr>
            <th>Scenario / Bracket</th>
            <th>Flooded Area</th>
            <th>Exposed Pop</th>
            <th>Roads Flooded</th>
          </tr>
        </thead>
        <tbody>
          <tr style="font-weight:700;color:#e2e8f0;background:rgba(255,255,255,0.03);">
            <td>Ground Truth (Observed)</td>
            <td>${gt.flooded_area_km2 !== undefined ? gt.flooded_area_km2.toFixed(2) + " km²" : "-"}</td>
            <td>${gt.exposed_population !== undefined ? gt.exposed_population.toFixed(1) : "-"}</td>
            <td>${formatRoad(gt.flooded_road_km, gt.road_status)}</td>
          </tr>
          <tr>
            <td>Conservative (p > 0.70)</td>
            <td>${u.conservative ? u.conservative.flooded_area_km2.toFixed(2) + " km²" : "-"}</td>
            <td>${u.conservative ? u.conservative.exposed_population.toFixed(1) : "-"}</td>
            <td>${u.conservative ? formatRoad(u.conservative.flooded_road_km, u.conservative.road_status) : "-"}</td>
          </tr>
          <tr>
            <td>Expected (p > 0.50)</td>
            <td>${u.expected ? u.expected.flooded_area_km2.toFixed(2) + " km²" : "-"}</td>
            <td>${u.expected ? u.expected.exposed_population.toFixed(1) : "-"}</td>
            <td>${u.expected ? formatRoad(u.expected.flooded_road_km, u.expected.road_status) : "-"}</td>
          </tr>
          <tr>
            <td>Worst-Case (p > 0.30)</td>
            <td>${u.worst_case ? u.worst_case.flooded_area_km2.toFixed(2) + " km²" : "-"}</td>
            <td>${u.worst_case ? u.worst_case.exposed_population.toFixed(1) : "-"}</td>
            <td>${u.worst_case ? formatRoad(u.worst_case.flooded_road_km, u.worst_case.road_status) : "-"}</td>
          </tr>
        </tbody>
      </table>
      <div class="exposure-caveat-box">
        <strong>Methodological Caveats:</strong> ${exposure.caveats}
      </div>
    </div>
  `;
}

function setupLayerControls() {
  const container = document.getElementById("layer-controls-list");
  container.innerHTML = "";

  const layerDefinitions = [
    { key: "sar_vv", label: "SAR VV Amplitude", color: "#94a3b8", defaultActive: true, defaultOpacity: 1.0 },
    { key: "ground_truth", label: "Ground Truth Mask", color: "#2980b9", defaultActive: true, defaultOpacity: 0.75 },
    { key: "prediction", label: "U-Net Prediction (p>0.50)", color: "#00ced1", defaultActive: true, defaultOpacity: 0.75 },
    { key: "prediction_deeplab", label: "DeepLabV3+ (p>0.50)", color: "#2ecc71", defaultActive: false, defaultOpacity: 0.75 },
    { key: "entropy", label: "Predictive Uncertainty", color: "#e040fb", defaultActive: false, defaultOpacity: 0.85 },
    { key: "error", label: "Error Analysis (FP/FN)", color: "#e74c3c", defaultActive: false, defaultOpacity: 0.80 }
  ];

  layerDefinitions.forEach(def => {
    const item = document.createElement("div");
    item.className = "layer-item";

    item.innerHTML = `
      <div class="layer-header">
        <label class="layer-label">
          <input type="checkbox" id="chk-${def.key}" ${def.defaultActive ? "checked" : ""}>
          <span class="layer-color-dot" style="background:${def.color};"></span>
          <span>${def.label}</span>
        </label>
      </div>
      <div class="layer-slider-row">
        <input type="range" id="rng-${def.key}" min="0" max="100" value="${Math.round(def.defaultOpacity * 100)}">
        <span class="opacity-val" id="val-${def.key}">${Math.round(def.defaultOpacity * 100)}%</span>
      </div>
    `;

    container.appendChild(item);

    // Bind checkbox
    const chk = item.querySelector(`#chk-${def.key}`);
    chk.addEventListener("change", (e) => {
      layers[def.key].active = e.target.checked;
      if (layers[def.key].overlay) {
        if (e.target.checked) {
          layers[def.key].overlay.addTo(map);
        } else {
          map.removeLayer(layers[def.key].overlay);
        }
      }
    });

    // Bind slider
    const rng = item.querySelector(`#rng-${def.key}`);
    const valSpan = item.querySelector(`#val-${def.key}`);
    rng.addEventListener("input", (e) => {
      const op = parseFloat(e.target.value) / 100.0;
      layers[def.key].opacity = op;
      valSpan.textContent = `${e.target.value}%`;
      if (layers[def.key].overlay) {
        layers[def.key].overlay.setOpacity(op);
      }
    });
  });
}

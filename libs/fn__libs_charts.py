from __future__ import annotations

import calendar
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from ladybug.color import Colorset
    HAS_LADYBUG = True
except ImportError:
    HAS_LADYBUG = False


def _rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))


def _get_nuanced_colorset():
    if HAS_LADYBUG:
        colorset = Colorset.nuanced()
        return [_rgb_to_hex((c.r, c.g, c.b)) for c in colorset]
    else:
        import plotly.express as px
        return px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


def _get_nuanced_colorscale():
    colors = _get_nuanced_colorset()
    n = len(colors)
    if n == 0:
        return "RdBu_r"
    return [[i / (n - 1) if n > 1 else 0, color] for i, color in enumerate(colors)]


@st.cache_data(show_spinner=False)
def _load_italy_regions_geojson():
    if not HAS_REQUESTS:
        return None
    try:
        url = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def _add_italy_region_outlines_to_plotly(fig):
    regions_geojson = _load_italy_regions_geojson()
    if not regions_geojson or "features" not in regions_geojson:
        return
    for feature in regions_geojson["features"]:
        geom = feature.get("geometry")
        if not geom or geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        exterior_ring = coords[0]
        lons = [coord[0] for coord in exterior_ring]
        lats = [coord[1] for coord in exterior_ring]
        fig.add_trace(
            go.Scattergeo(
                lon=lons, lat=lats, mode="lines",
                line=dict(color="rgba(40,40,40,0.65)", width=0.6),
                showlegend=False, hoverinfo="skip",
            )
        )


def f201__plotly_italy_map(points, metric_key: str, height: int, title: str, *, show_colorbar: bool = True):
    df = pd.DataFrame(points)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=title)
        return fig

    # Δ map: color by magnitude (0..5) using Viridis, keep signed value in hover
    z = pd.to_numeric(df[metric_key], errors="coerce").clip(-5, 5)
    df = df.copy()
    df["_z"] = z
    df["_z_mag"] = z.abs().clip(0, 5)

    customdata = list(
        zip(
            df.get("location_id", "").astype(str),
            df.get("location_name", "").astype(str),
            pd.to_numeric(df["_z"], errors="coerce"),
        )
    )

    fig = go.Figure(
        data=go.Scattergeo(
            lon=pd.to_numeric(df["longitude"], errors="coerce"),
            lat=pd.to_numeric(df["latitude"], errors="coerce"),
            mode="markers",
            text=df.get("location_name", df.get("location_id", "")),
            customdata=customdata,
            marker=dict(
                size=9,
                color=df["_z_mag"],
                cmin=0,
                cmax=5,
                colorscale="Viridis",
                reversescale=True,  # yellow at 0, blue/purple at 5
                showscale=bool(show_colorbar),
                line=dict(color="#111", width=0.7),
                colorbar=(
                    dict(
                        title=dict(text="ΔT (°C)", side="top", font=dict(size=13.2)),
                        tickmode="array",
                        tickvals=[0, 5],
                        len=0.28,
                        thickness=10,
                        x=0.02,
                        y=0.08,
                        xanchor="left",
                        yanchor="bottom",
                        tickfont=dict(size=13.2),
                        ticks="outside",
                        ticklen=8,
                        tickwidth=1,
                        xpad=6,
                        ypad=10,
                    )
                    if show_colorbar
                    else None
                ),
            ),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Δ: %{customdata[2]:.1f} °C<br>"
                "lat/lon: %{lat:.2f}, %{lon:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#f7f7f7",
        showcountries=True,
        countrycolor="#333",
        showcoastlines=True,
        coastlinecolor="#333",
        showframe=False,
        lataxis_range=[35.0, 48.9],
        lonaxis_range=[6.0, 19.9],
    )
    
    # Add Italian region outlines
    _add_italy_region_outlines_to_plotly(fig)
    
    fig.update_layout(
        height=height + 80,
        margin=dict(l=0, r=0, t=34, b=0),
        title=dict(text=title, x=0.0, xanchor="left", y=0.98),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
    )
    return fig


def f202__plotly_italy_map_abs(
    points,
    metric_key: str,
    height: int,
    title: str,
    *,
    show_colorbar: bool = True,
):
    """
    Plotly Italy map for absolute metrics (e.g. Tmax/Tavg), not deltas.
    """
    df = pd.DataFrame(points)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=title)
        return fig

    # Scenario map: fixed 0..40, blue(0) -> red(40)
    z = pd.to_numeric(df.get(metric_key), errors="coerce").clip(0, 40)
    df = df.copy()
    df["_z"] = z

    customdata = list(
        zip(
            df.get("location_id", "").astype(str),
            df.get("location_name", "").astype(str),
        )
    )

    cb_title = "T (°C)"
    if metric_key.lower().startswith("tmax"):
        cb_title = "Tmax (°C)"
    elif metric_key.lower().startswith("tavg"):
        cb_title = "Tavg (°C)"

    fig = go.Figure(
        data=go.Scattergeo(
            lon=pd.to_numeric(df["longitude"], errors="coerce"),
            lat=pd.to_numeric(df["latitude"], errors="coerce"),
            mode="markers",
            text=df.get("location_name", df.get("location_id", "")),
            customdata=customdata,
            marker=dict(
                size=9,
                color=df["_z"],
                cmin=0,
                cmax=40,
                colorscale="RdBu_r",
                showscale=bool(show_colorbar),
                line=dict(color="#111", width=0.7),
                colorbar=(
                    dict(
                        title=dict(text=cb_title, font=dict(size=13.2)),
                        tickmode="array",
                        tickvals=[0, 20, 40],
                        len=0.28,
                        thickness=10,
                        x=0.02,
                        y=0.08,
                        xanchor="left",
                        yanchor="bottom",
                        tickfont=dict(size=13.2),
                    )
                    if show_colorbar
                    else None
                ),
            ),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                f"{cb_title}: %{{marker.color:.1f}}<br>"
                "lat/lon: %{lat:.2f}, %{lon:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#f7f7f7",
        showcountries=True,
        countrycolor="#333",
        showcoastlines=True,
        coastlinecolor="#333",
        showframe=False,
        lataxis_range=[35.0, 48.9],
        lonaxis_range=[6.0, 19.9],
    )
    
    # Add Italian region outlines
    _add_italy_region_outlines_to_plotly(fig)
    
    fig.update_layout(
        height=height + 80,
        margin=dict(l=0, r=0, t=34, b=0),
        title=dict(text=title, x=0.0, xanchor="left", y=0.98),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
    )
    return fig


def f203__parse_plotly_selection(sel: Any) -> Optional[Tuple[str, str]]:
    """Parse Streamlit plotly chart selection to (location_id, location_name)."""
    if sel is None:
        return None
    # Streamlit may return event with .selection or dict with "selection"
    selection = getattr(sel, "selection", None)
    if selection is None and isinstance(sel, dict):
        selection = sel.get("selection")
    # Handle list of events (e.g. [{"selection": {...}}])
    if isinstance(sel, (list, tuple)) and len(sel) > 0:
        first = sel[0]
        selection = getattr(first, "selection", first.get("selection") if isinstance(first, dict) else None)
    if not isinstance(selection, dict):
        return None
    pts = selection.get("points") or []
    if not pts:
        return None
    p0 = pts[0] if isinstance(pts[0], dict) else {}
    cd = p0.get("customdata")
    if isinstance(cd, (list, tuple)) and len(cd) >= 1:
        loc_id = str(cd[0])
        loc_name = str(cd[1]) if len(cd) > 1 else loc_id
        return loc_id, loc_name
    # Fallback: point_index might refer to data row
    if "point_index" in p0 and "pointIndex" in p0:
        pass  # would need dataframe to look up - leave as None
    return None


# -----------------------------
# D3/JS embedded dashboard data
# -----------------------------
@st.cache_data(show_spinner=False)


def f204__d3_dashboard_html_abs(
    *,
    points: List[Dict[str, Any]],
    profiles_bundle: Dict[str, Any],
    metric_key: str,  # "Tmax" or "Tavg"
    width: int = 1100,
    height: int = 640,
    scenario_variant: str,
    percentile: float = 99.0,
) -> str:
    """
    D3/JS dashboard for absolute scenario temperatures (single series).

    - metric_key="Tmax": monthly uses percentile P of daily series
    - metric_key="Tavg": monthly uses mean of daily series
    """
    data_json = json.dumps(points, ensure_ascii=False)
    prof_json = json.dumps(profiles_bundle, ensure_ascii=False)

    return f"""
<!doctype html>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:wght@400;600&display=swap">
<style>
  :root {{
    --font: Inter, Helvetica, Arial, sans-serif;
    --serif-font: 'Lora', 'Source Serif Pro', 'Source Serif 4', Georgia, 'Times New Roman', serif;
    --fs: 13px;
    --fs2: 12px;
    --fg: #222;
    --muted: rgba(0,0,0,0.65);
  }}
  body {{ font-family: var(--font); }}
  .layout {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }}
  .map-panel {{ flex: 0 0 62%; }}
  .charts-panel {{ flex: 1 1 auto; }}
  .panel-title {{
    font: 600 var(--fs) var(--font);
    color: var(--fg);
    margin: 2px 0 2px;
    line-height: 1.2;
  }}
  .chart-title {{
    padding-left: 50px;
  }}
  .chart-title {{
    padding-left: 50px;
  }}
  svg text {{
    font-family: var(--font);
    font-size: var(--fs2);
    fill: var(--fg);
  }}
</style>

<div class="layout">
  <div class="map-panel">
    <div id="map"></div>
  </div>
  <div class="charts-panel">
    <div class="panel-title chart-title" id="cmp"></div>
    <div class="panel-title chart-title" id="cmp2" style="margin-top: 0px;"></div>
    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="bar_title"></div>
    <div id="bar_month"></div>
    <div style="height:8px"></div>
    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="ts_title"></div>
    <div class="panel-title chart-title" id="ts_title2" style="margin-top: 0px;"></div>
    <div class="panel-title chart-title" id="ts_title3" style="margin-top: 0px;"></div>
    <div id="scatter"></div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
(async function() {{
  const W = {width}, H = {height};
  const metricKey = "{metric_key}";
  const points = {data_json};
  const bundle = {prof_json};
  const keys = bundle.keys || [];
  const profiles = bundle.profiles || {{}};

  const PCT = {percentile};
  const P = Math.max(0.95, Math.min(1.0, (PCT / 100.0)));
  const statLabel = (metricKey === "Tmax") ? "Tmax" : "Tavg";

  function scenarioLabel() {{
    const c = "{scenario_variant}";
    const parts = c.split("_");
    if (parts.length >= 2 && parts[0].startsWith("rcp")) {{
      const rcp = parts[0].slice(3);
      const year = parts[1];
      return "RCP " + rcp + " Year " + year;
    }}
    return c.toUpperCase();
  }}

  // --- Map ---
  const mapDiv = d3.select("#map");
  mapDiv.selectAll("*").remove();
  const mapW = 748, mapH = 748;
  const svg = mapDiv.append("svg")
    .attr("viewBox", [0, 0, mapW, mapH])
    .style("width", "100%")
    .style("height", "auto");

  const overviewMap = d3.select("#overview_map");

  const topo = await d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json");
  const countries = topojson.feature(topo, topo.objects.countries);
  const italy = {{
    type: "FeatureCollection",
    features: countries.features.filter(d => +d.id === 380)
  }};
  const pad = 16;
  const projection = d3.geoMercator().fitExtent([[pad, pad], [mapW - pad, mapH - pad]], italy);
  const path = d3.geoPath(projection);

  svg.append("path")
    .datum(italy.features[0])
    .attr("d", path)
    .attr("fill", "#f7f7f7")
    .attr("stroke", "#333")
    .attr("stroke-width", 1);

  // Region outlines (Italy) overlay
  // Source: Openpolis geojson-italy (regions). If this fetch fails (offline/proxy), we simply skip.
  try {{
    const regionsUrl = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson";
    const regions = await d3.json(regionsUrl);
    if (regions && regions.features) {{
      svg.append("g")
        .attr("pointer-events", "none")
        .selectAll("path.region")
        .data(regions.features)
        .join("path")
          .attr("class", "region")
          .attr("d", path)
          .attr("fill", "none")
          .attr("stroke", "rgba(40,40,40,0.65)")
          .attr("stroke-width", 0.6);
    }}
  }} catch (e) {{
    // ignore
  }}

  // Scenario tab map colors: blue(0) -> red(40)
  const MINV = 0, MAXV = 40;
  const clamp = (x) => Math.max(MINV, Math.min(MAXV, x));
  const color = d3.scaleSequential(t => d3.interpolateRdBu(1 - t)).domain([MINV, MAXV]);
  const futureColor = d3.interpolateViridis(0.12);
  const futureColorStrong = d3.color(futureColor).copy({{opacity: 0.75}}).formatRgb();
  const futureColorDim = d3.color(futureColor).copy({{opacity: 0.45}}).formatRgb();

  const pointById = new Map(points.map(d => [String(d.location_id), d]));
  function mapValueForLocation(locId) {{
    const p = pointById.get(String(locId));
    if (!p) return NaN;
    return +p[metricKey];
  }}

  function format1(x) {{
    return (Math.round(x * 10) / 10).toFixed(1);
  }}

  const tip = mapDiv.append("div")
    .style("position", "absolute")
    .style("pointer-events", "none")
    .style("background", "rgba(255,255,255,0.95)")
    .style("border", "1px solid #ddd")
    .style("border-radius", "8px")
    .style("padding", "10px 12px")
    .style("font", "12px/1.3 Inter, Helvetica, Arial, sans-serif")
    .style("display", "none");

  svg.append("g")
    .selectAll("circle")
    .data(points)
    .join("circle")
      .attr("cx", d => projection([+d.longitude, +d.latitude])[0])
      .attr("cy", d => projection([+d.longitude, +d.latitude])[1])
      .attr("r", 5.0)
      .attr("fill", d => {{
        const v = mapValueForLocation(d.location_id);
        if (!Number.isFinite(v)) return "#999";
        return color(clamp(v));
      }})
      .attr("stroke", "#111")
      .attr("stroke-width", 0.7)
      .attr("opacity", 0.95)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {{
        const v = mapValueForLocation(d.location_id);
        tip
          .style("display", "block")
          .style("left", (event.pageX + 12) + "px")
          .style("top", (event.pageY + 12) + "px")
          .html(`
            <div style="font-weight:600; margin-bottom:4px;">${{d.location_name || d.location_id}}</div>
            <div>Annual ${{statLabel}}: <b>${{Number.isFinite(v) ? format1(v) : "n/a"}}</b> °C</div>
          `);
      }})
      .on("mouseleave", () => tip.style("display", "none"))
      .on("click", (_, d) => {{
        renderCharts(String(d.location_id));
      }});

  // --- Charts helpers ---
  function cleanVals(arr) {{
    return (arr || []).filter(v => v != null && Number.isFinite(+v)).map(v => +v);
  }}
  function quant(vals, p) {{
    const v = cleanVals(vals).sort(d3.ascending);
    if (!v.length) return NaN;
    return d3.quantileSorted(v, p);
  }}
  function mean(vals) {{
    const v = cleanVals(vals);
    if (!v.length) return NaN;
    return d3.mean(v);
  }}
  function seriesPairs(locId) {{
    const p = profiles[String(locId)];
    if (!p) return [];
    const s = (p.series) || [];
    const out = [];
    for (let i = 0; i < keys.length; i++) {{
      const v = s[i];
      if (v == null) continue;
      const month = keys[i][0];
      const day = keys[i][1];
      out.push({{ month, day, v: +v, doy: i + 1 }});
    }}
    return out;
  }}
  function monthStat(locId, month) {{
    const pairs = seriesPairs(locId).filter(d => d.month === month).map(d => d.v);
    if (metricKey === "Tmax") return quant(pairs, P);
    return mean(pairs);
  }}

  function hottestMonthByTmax(locId) {{
    const pairs = seriesPairs(locId);
    if (!pairs.length) return null;
    let bestMonth = null;
    let bestVal = -Infinity;
    for (let m = 1; m <= 12; m++) {{
      const vals = pairs.filter(d => d.month === m).map(d => d.v);
      const v = quant(vals, P);
      if (!Number.isFinite(v)) continue;
      if (v > bestVal) {{
        bestVal = v;
        bestMonth = m;
      }}
    }}
    return bestMonth;
  }}

  const stationIcon = `<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F000%2F437%2F977%2Foriginal%2Fvector-location-icon.jpg&f=1&nofb=1&ipt=3a764b6b7a498a7caaacc0b13e3d01da381e4d011fce2fd9d49efa1e7d5c0719" alt="station" style="width:14px;height:14px;vertical-align:-2px;margin-right:6px;">`;
  const scenarioIcon = `<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fas2.ftcdn.net%2Fv2%2Fjpg%2F03%2F54%2F02%2F33%2F1000_F_354023354_YBnwu0FDuRUVmsFgOD1Ihew6PNtNvyFJ.jpg&f=1&nofb=1&ipt=002a4610321645d7046875eca50971cd4f64d12246fb3e3d7d134e2d446a087c" alt="scenario" style="width:14px;height:14px;vertical-align:-2px;margin-right:6px;">`;

  function renderCharts(locId) {{
    const meta = profiles[String(locId)];
    if (!meta) return;

    const cmp = document.getElementById("cmp");
    if (cmp) cmp.innerHTML = (`${{stationIcon}}<b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    const cmp2 = document.getElementById("cmp2");
    if (cmp2) cmp2.innerHTML = (`${{scenarioIcon}}<b>${{scenarioLabel()}}</b>`);

    const barTitle = document.getElementById("bar_title");
    if (barTitle) {{
      if (metricKey === "Tmax") barTitle.textContent = ("Monthly Tmax - " + PCT.toFixed(1) + "th perc.");
      else barTitle.textContent = ("Monthly Tavg (mean)");
    }}

    const tsTitle = document.getElementById("ts_title");
    if (tsTitle) tsTitle.innerHTML = (`${{stationIcon}}<b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    const tsTitle2 = document.getElementById("ts_title2");
    if (tsTitle2) tsTitle2.innerHTML = (`${{scenarioIcon}}<b>${{scenarioLabel()}}</b>`);
    const tsTitle3 = document.getElementById("ts_title3");
    if (tsTitle3) {{
      if (metricKey === "Tmax") tsTitle3.textContent = ("Daily Tmax");
      else tsTitle3.textContent = ("Daily Tavg");
    }}

    const pairs = seriesPairs(locId);

    // --- Scatter (hottest month by Tmax percentile) ---
    const hottestMonth = hottestMonthByTmax(locId);
    const scatterPairs = (hottestMonth != null)
      ? pairs.filter(d => d.month === hottestMonth)
      : pairs;
    const sw = 240, sh = 200;
    const sm = {{t: 26, r: 14, b: 44, l: 50}};
    d3.select("#scatter").selectAll("*").remove();
    const ssvg = d3.select("#scatter").append("svg").attr("viewBox", [0, 0, sw, sh]);

    function labelFromDoy(doy) {{
      const k = keys[(doy|0) - 1];
      if (!k) return String(doy);
      return `${{k[0]}}/${{k[1]}}`;
    }}

    if (!scatterPairs.length) {{
      ssvg.append("text")
        .attr("x", sm.l)
        .attr("y", sm.t + 16)
        .style("font", "12px Inter, Helvetica, Arial, sans-serif")
        .attr("fill", "rgba(0,0,0,0.65)")
        .text("No daily data available for this location.");
    }} else {{
      const x = d3.scaleLinear()
        .domain(d3.extent(scatterPairs, d => d.doy) || [182, 243])
        .range([sm.l, sw - sm.r]);

      const y = d3.scaleLinear()
        .domain([0, 40])
        .range([sh - sm.b, sm.t]);

      const tickVals = scatterPairs
        .map(d => d.doy)
        .filter((d, i, a) => a.indexOf(d) === i)
        .filter((d, i) => i % 7 === 0);

      ssvg.append("g")
        .attr("transform", `translate(0,${{sh - sm.b}})`)
        .call(d3.axisBottom(x).tickValues(tickVals).tickFormat(labelFromDoy))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("g")
        .attr("transform", `translate(${{sm.l}},0)`)
        .call(
          d3.axisLeft(y)
            .tickValues([0, 10, 20, 30, 40])
            .tickFormat(d => `${{d}} ºC`)
        )
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("text")
        .attr("x", (sw / 2))
        .attr("y", sh - 11)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Days");

      ssvg.append("text")
        .attr("x", 6)
        .attr("y", 10)
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("");

      const line = d3.line().x(d => x(d.doy)).y(d => y(d.v));
      const sorted = scatterPairs.slice().sort((a,b)=>a.doy-b.doy);

      ssvg.append("path")
        .datum(sorted)
        .attr("fill", "none")
        .attr("stroke", futureColorStrong)
        .attr("stroke-width", 1.7)
        .attr("d", line);

      ssvg.append("g")
        .selectAll("circle.val")
        .data(sorted)
        .join("circle")
          .attr("cx", d => x(d.doy))
          .attr("cy", d => y(d.v))
          .attr("r", 2.2)
          .attr("fill", futureColorDim)
          .attr("stroke", "rgba(20,20,20,0.15)");
    }}

    // --- Monthly bars (absolute) ---
    function renderBar(selector, data) {{
      const bw = 230, bh = 127;
      const bm = {{t: 10, r: 10, b: 42, l: 50}};
      d3.select(selector).selectAll("*").remove();
      const bsvg = d3.select(selector).append("svg").attr("viewBox", [0, 0, bw, bh]);

      const bx = d3.scaleBand().domain(data.map(d => d.k)).range([bm.l, bw - bm.r]).padding(0.15);
      const by = d3.scaleLinear().domain([0, 40]).range([bh - bm.b, bm.t]);

      bsvg.append("g")
        .attr("transform", `translate(0,${{bh - bm.b}})`)
        .call(d3.axisBottom(bx))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));
      bsvg.append("g")
        .attr("transform", `translate(${{bm.l}},0)`)
        .call(
          d3.axisLeft(by)
            .tickValues([0, 10, 20, 30, 40])
            .tickFormat(d => `${{d}} ºC`)
        )
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      bsvg.append("g").selectAll("rect")
        .data(data)
        .join("rect")
          .attr("x", d => bx(d.k))
          .attr("width", bx.bandwidth())
          .attr("y", d => by(d.v))
          .attr("height", d => (bh - bm.b) - by(d.v))
          .attr("fill", d3.color(futureColor).copy({{opacity: 0.60}}).formatRgb());

      bsvg.append("text")
        .attr("x", (bw / 2))
        .attr("y", bh - 7)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Months");
    }}

    const monthData = d3.range(1, 13).map(m => {{
      const v = monthStat(locId, m);
      const vClamped = Number.isFinite(v) ? Math.max(0, Math.min(40, v)) : 0;
      return {{k: String(m), v: vClamped}};
    }});
    renderBar("#bar_month", monthData);
  }}

  // Legend (bottom-left), +10% text, tiny bit up
  const legendW = 180, legendH = 10;
  const legendX = 16, legendY = mapH - 34;
  const defs = svg.append("defs");
  const lg = defs.append("linearGradient").attr("id", "lg_abs");
  const stops = d3.range(0, 1.00001, 0.1);
  lg.selectAll("stop")
    .data(stops)
    .join("stop")
      .attr("offset", d => (d * 100) + "%")
      .attr("stop-color", d => color(MINV + d * (MAXV - MINV)));

  svg.append("rect")
    .attr("x", legendX)
    .attr("y", legendY)
    .attr("width", legendW)
    .attr("height", legendH)
    .attr("fill", "url(#lg_abs)")
    .attr("stroke", "#333")
    .attr("stroke-width", 0.6);

  const legendFont = 18.17; // +10%
  svg.append("text")
    .attr("x", legendX)
    .attr("y", legendY - 10) // move font a bit further from bar
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("T (°C)");

  svg.append("text")
    .attr("x", legendX)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("0");

  svg.append("text")
    .attr("x", legendX + legendW / 2)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("text-anchor", "middle")
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("20");

  svg.append("text")
    .attr("x", legendX + legendW)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("text-anchor", "end")
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("40");

  if (points.length) {{
    renderCharts(String(points[0].location_id));
  }}
}})();
</script>
"""



def f205__d3_dashboard_html(
    *,
    points: List[Dict[str, Any]],
    profiles_bundle: Dict[str, Any],
    metric_key: str,
    width: int = 1100,
    height: int = 640,
    baseline_variant: str,
    compare_variant: str,
    percentile: float = 99.0,
) -> str:
    data_json = json.dumps(points, ensure_ascii=False)
    prof_json = json.dumps(profiles_bundle, ensure_ascii=False)

    return f"""
<!doctype html>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:wght@400;600&display=swap">
<style>
  :root {{
    --font: Inter, Helvetica, Arial, sans-serif;
    --serif-font: 'Lora', 'Source Serif Pro', 'Source Serif 4', Georgia, 'Times New Roman', serif;
    --fs: 13px;
    --fs2: 12px;
    --fg: #222;
    --muted: rgba(0,0,0,0.65);
  }}
  body {{ font-family: var(--font); }}
  .layout {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }}
  .map-panel {{
    flex: 0 0 62%;
  }}
  .charts-panel {{
    flex: 1 1 auto;
  }}
  .panel-title {{
    font: 600 var(--fs) var(--font);
    color: var(--fg);
    margin: 2px 0 2px;
    line-height: 1.2;
  }}
  .meta {{
    font: 400 var(--fs2) var(--font);
    color: var(--muted);
    margin-bottom: 8px;
  }}
  svg text {{
    font-family: var(--font);
    font-size: var(--fs2);
    fill: var(--fg);
  }}
</style>

<div class="layout">
  <div class="map-panel">
    <div id="map"></div>
  </div>

  <div class="charts-panel">
    <div class="panel-title chart-title" id="cmp"></div>
    <div class="panel-title chart-title" id="cmp2" style="margin-top: 0px;"></div>
    <div class="meta" id="sel" style="display: none;"></div>

    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="bar_title"></div>
    <div id="bar_month"></div>

    <div style="height:10px"></div>
    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="ts_title"></div>
    <div class="panel-title chart-title" id="ts_title2" style="margin-top: 0px;"></div>
    <div class="panel-title chart-title" id="ts_title3" style="margin-top: 0px;"></div>
    <div id="scatter"></div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
(async function() {{
  const W = {width}, H = {height};
  const metricKey = "{metric_key}";
  const points = {data_json};
  const bundle = {prof_json};
  const keys = bundle.keys || [];
  const profiles = bundle.profiles || {{}};

  const PCT = {percentile};
  const P = Math.max(0.95, Math.min(1.0, (PCT / 100.0)));
  const statLabel = (metricKey === "dTmax") ? "ΔTmax" : "ΔTavg";
  const metricLabel = (metricKey === "dTmax") ? "Tmax" : "Tavg";

  function baselineLabel() {{
    const b = "{baseline_variant}";
    if (b.startsWith("tmyx_")) return "TMYx " + b.slice(5).replace("_", " ");
    if (b.startsWith("tmyx")) return "TMYx";
    return b.toUpperCase();
  }}
  function compareLabel() {{
    const c = "{compare_variant}";
    const parts = c.split("_");
    if (parts.length >= 2 && parts[0].startsWith("rcp")) {{
      const rcp = parts[0].slice(3);
      const year = parts[1];
      return "RCP " + rcp + " Year " + year;
    }}
    return c.toUpperCase();
  }}


  const mapDiv = d3.select("#map");
  mapDiv.selectAll("*").remove();
  const mapW = 748, mapH = 748;
  const svg = mapDiv.append("svg")
    .attr("viewBox", [0, 0, mapW, mapH])
    .style("width", "100%")
    .style("height", "auto");

  const topo = await d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json");
  const countries = topojson.feature(topo, topo.objects.countries);
  const italy = {{
    type: "FeatureCollection",
    features: countries.features.filter(d => +d.id === 380)
  }};

  const regionsUrl = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson";
  let regions = null;
  try {{
    regions = await d3.json(regionsUrl);
  }} catch (e) {{
    // ignore
  }}

  const pad = 16;
  const projection = d3.geoMercator().fitExtent([[pad, pad], [mapW - pad, mapH - pad]], italy);
  const path = d3.geoPath(projection);
  svg.append("path")
    .datum(italy.features[0])
    .attr("d", path)
    .attr("fill", "#f7f7f7")
    .attr("stroke", "#333")
    .attr("stroke-width", 1);

  // Region outlines (Italy) overlay
  if (regions && regions.features) {{
    svg.append("g")
      .attr("pointer-events", "none")
      .selectAll("path.region")
      .data(regions.features)
      .join("path")
        .attr("class", "region")
        .attr("d", path)
        .attr("fill", "none")
        .attr("stroke", "rgba(40,40,40,0.65)")
        .attr("stroke-width", 0.6);
  }}

  // Δ tab map colors: Viridis scale on |Δ|, clipped to 0..5°C
  const MINV = 0, MAXV = 5;
  const clamp = (x) => Math.max(MINV, Math.min(MAXV, x));
  // Reverse Viridis so 0 -> yellow, 5 -> blue/purple
  const color = d3.scaleSequential(d3.interpolateViridis).domain([MAXV, MINV]);
  const futureColor = d3.interpolateViridis(0.12);
  const baseColor = d3.interpolateViridis(0.33);
  const futureColorStrong = d3.color(futureColor).copy({{opacity: 0.75}}).formatRgb();
  const futureColorDim = d3.color(futureColor).copy({{opacity: 0.45}}).formatRgb();
  const baseColorStrong = d3.color(baseColor).copy({{opacity: 0.75}}).formatRgb();
  const baseColorDim = d3.color(baseColor).copy({{opacity: 0.45}}).formatRgb();

  function cleanVals(arr) {{
    return (arr || []).filter(v => v != null && Number.isFinite(+v)).map(v => +v);
  }}

  function quant(vals, p) {{
    const v = cleanVals(vals).sort(d3.ascending);
    if (!v.length) return NaN;
    return d3.quantileSorted(v, p);
  }}

  function mean(vals) {{
    const v = cleanVals(vals);
    if (!v.length) return NaN;
    return d3.mean(v);
  }}

  // Annual ΔT shown on the map:
  // - ΔTmax: qP(Comp) - qP(Base) over the whole year
  // - ΔTavg: mean(Comp) - mean(Base) over the whole year
  const pointById = new Map(points.map(d => [String(d.location_id), d]));
  function mapValueForLocation(locId) {{
    const p = pointById.get(String(locId));
    if (!p) return NaN;
    return (metricKey === "dTmax") ? (+p.dTmax) : (+p.dTavg);
  }}

  const tip = mapDiv.append("div")
    .style("position", "absolute")
    .style("pointer-events", "none")
    .style("background", "rgba(255,255,255,0.95)")
    .style("border", "1px solid #ddd")
    .style("border-radius", "8px")
    .style("padding", "10px 12px")
    .style("font", "12px/1.3 Inter, Helvetica, Arial, sans-serif")
    .style("display", "none");

  function format1(x) {{
    return (Math.round(x * 10) / 10).toFixed(1);
  }}

  function getPairs(locId) {{
    const p = profiles[locId];
    if (!p) return [];
    const base = (p.base) || [];
    const comp = (p.comp) || [];
    const out = [];
    for (let i = 0; i < keys.length; i++) {{
      const b = base[i];
      const c = comp[i];
      if (b == null || c == null) continue;
      const month = keys[i][0];
      const day = keys[i][1];
      out.push({{ month, day, base: +b, comp: +c, delta: (+c) - (+b), doy: i + 1 }});
    }}
    return out;
  }}

  function monthDelta(locId, month) {{
    const p = profiles[String(locId)];
    if (!p) return NaN;
    const base = [];
    const comp = [];
    for (let i = 0; i < keys.length; i++) {{
      if (keys[i][0] !== month) continue;
      const b = p.base[i];
      const c = p.comp[i];
      if (b == null || c == null) continue;
      base.push(+b);
      comp.push(+c);
    }}
    if (metricKey === "dTmax") {{
      return quant(comp, P) - quant(base, P);
    }}
    return mean(comp) - mean(base);
  }}

  function renderCharts(locId) {{
    const meta = profiles[locId];
    if (!meta) return;
    document.getElementById("sel").textContent = "";

    const cmp = document.getElementById("cmp");
    if (cmp) {{
      cmp.innerHTML = (`Station: <b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    }}
    const cmp2 = document.getElementById("cmp2");
    if (cmp2) {{
      cmp2.innerHTML = (`Scenario: <b>${{compareLabel()}}</b> vs <b>${{baselineLabel()}}</b>`);
    }}
    const barTitle = document.getElementById("bar_title");
    if (barTitle) {{
      if (metricKey === "dTmax") {{
        barTitle.textContent = ("Monthly ΔTmax - " + PCT.toFixed(1) + "th perc. (difference)");
      }} else {{
        barTitle.textContent = ("Monthly ΔTavg (mean)");
      }}
    }}
    const tsTitle = document.getElementById("ts_title");
    if (tsTitle) {{
      tsTitle.innerHTML = (`Station: <b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    }}
    const tsTitle2 = document.getElementById("ts_title2");
    if (tsTitle2) {{
      tsTitle2.innerHTML = (`Scenario: <b>${{compareLabel()}}</b> vs <b>${{baselineLabel()}}</b>`);
    }}
    const tsTitle3 = document.getElementById("ts_title3");
    if (tsTitle3) {{
      if (metricKey === "dTmax") {{
        tsTitle3.textContent = ("Daily ΔTmax - " + PCT.toFixed(1) + "th perc. - Hottest month");
      }} else {{
        tsTitle3.textContent = ("Daily ΔTavg (mean) - Hottest month");
      }}
    }}

    const pairs = getPairs(locId);
    const monthScores = d3.rollup(
      pairs,
      v => d3.max(v, d => d.comp),
      d => d.month
    );
    const warmMonth = monthScores.size
      ? Array.from(monthScores.entries()).sort((a, b) => b[1] - a[1])[0][0]
      : null;

    // --- Time series (warmest month only) ---
    const scatterPairs = warmMonth ? pairs.filter(d => d.month === warmMonth) : [];
    const sw = 240, sh = 200;
    const sm = {{t: 26, r: 14, b: 44, l: 50}};
    d3.select("#scatter").selectAll("*").remove();
    const ssvg = d3.select("#scatter").append("svg").attr("viewBox", [0, 0, sw, sh]);

    function labelFromDoy(doy) {{
      const k = keys[(doy|0) - 1];
      if (!k) return String(doy);
      return `${{k[0]}}/${{k[1]}}`;
    }}

    if (!scatterPairs.length) {{
      ssvg.append("text")
        .attr("x", sm.l)
        .attr("y", sm.t + 16)
        .style("font", "12px Inter, Helvetica, Arial, sans-serif")
        .attr("fill", "rgba(0,0,0,0.65)")
        .text("No July/August daily data available for this location.");
    }} else {{
      const x = d3.scaleLinear()
        .domain(d3.extent(scatterPairs, d => d.doy) || [182, 243])
        .range([sm.l, sw - sm.r]);

      const y = d3.scaleLinear()
        .domain([0, 40])
        .range([sh - sm.b, sm.t]);

      const tickVals = scatterPairs
        .map(d => d.doy)
        .filter((d, i, a) => a.indexOf(d) === i)
        .filter((d, i) => i % 7 === 0);

      ssvg.append("g")
        .attr("transform", `translate(0,${{sh - sm.b}})`)
        .call(d3.axisBottom(x).tickValues(tickVals).tickFormat(labelFromDoy))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("g")
        .attr("transform", `translate(${{sm.l}},0)`)
        .call(
          d3.axisLeft(y)
            .tickValues([0, 10, 20, 30, 40])
            .tickFormat(d => `${{d}} ºC`)
        )
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("text")
        .attr("x", (sw / 2))
        .attr("y", sh - 11)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Days");

      ssvg.append("text")
        .attr("x", 6)
        .attr("y", 10)
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("");

      const lineBase = d3.line().x(d => x(d.doy)).y(d => y(d.base));
      const lineComp = d3.line().x(d => x(d.doy)).y(d => y(d.comp));
      const sorted = scatterPairs.slice().sort((a,b)=>a.doy-b.doy);

      ssvg.append("path")
        .datum(sorted)
        .attr("fill", "none")
        .attr("stroke", baseColorStrong)
        .attr("stroke-width", 1.5)
        .attr("d", lineBase);

      ssvg.append("path")
        .datum(sorted)
        .attr("fill", "none")
        .attr("stroke", futureColorStrong)
        .attr("stroke-width", 1.5)
        .attr("d", lineComp);

      ssvg.append("g")
        .selectAll("circle.base")
        .data(sorted)
        .join("circle")
          .attr("class", "base")
          .attr("cx", d => x(d.doy))
          .attr("cy", d => y(d.base))
          .attr("r", 2.2)
          .attr("fill", baseColorDim)
          .attr("stroke", "rgba(20,20,20,0.15)");

      ssvg.append("g")
        .selectAll("circle.comp")
        .data(sorted)
        .join("circle")
          .attr("class", "comp")
          .attr("cx", d => x(d.doy))
          .attr("cy", d => y(d.comp))
          .attr("r", 2.2)
          .attr("fill", futureColorDim)
          .attr("stroke", "rgba(20,20,20,0.15)");

      const baseLegend = baselineLabel();
      const compLegend = compareLabel();
      const lx = sw - sm.r - 160, ly = 8;
      const lg = ssvg.append("g").attr("transform", `translate(${{lx}},${{ly}})`);
      lg.append("circle").attr("r", 4).attr("cx", 0).attr("cy", 0).attr("fill", baseColorStrong);
      lg.append("text").attr("x", 10).attr("y", 4).style("font-size", "7.43px").text(baseLegend);
      lg.append("circle").attr("r", 4).attr("cx", 0).attr("cy", 18).attr("fill", futureColorStrong);
      lg.append("text").attr("x", 10).attr("y", 22).style("font-size", "7.43px").text(compLegend);
    }}

    // --- Monthly bars ---
    function renderBar(selector, data) {{
      const bw = 230, bh = 127;
      const bm = {{t: 10, r: 10, b: 42, l: 50}};
      d3.select(selector).selectAll("*").remove();
      const bsvg = d3.select(selector).append("svg").attr("viewBox", [0, 0, bw, bh]);

      const bx = d3.scaleBand().domain(data.map(d => d.k)).range([bm.l, bw - bm.r]).padding(0.15);
      const by = d3.scaleLinear().domain([0, 5]).range([bh - bm.b, bm.t]);

      bsvg.append("g")
        .attr("transform", `translate(0,${{bh - bm.b}})`)
        .call(d3.axisBottom(bx).tickValues(bx.domain().filter((d,i)=> (data.length>20? i%4===0:true))))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));
      bsvg.append("g")
        .attr("transform", `translate(${{bm.l}},0)`)
        .call(d3.axisLeft(by).ticks(5).tickFormat(d => `${{d}} ºC`))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      bsvg.append("g")
        .attr("transform", `translate(${{bm.l}},0)`)
        .call(d3.axisLeft(by).ticks(5).tickSize(-(bw - bm.l - bm.r)).tickFormat(""))
        .call(g => g.selectAll("line").attr("stroke", "rgba(255,255,255,0.9)"))
        .call(g => g.select(".domain").remove());
      bsvg.append("line")
        .attr("x1", bm.l).attr("x2", bw - bm.r)
        .attr("y1", by(0)).attr("y2", by(0))
        .attr("stroke", "rgba(0,0,0,0.55)").attr("stroke-width", 0.8);

      bsvg.append("g").selectAll("rect")
        .data(data)
        .join("rect")
          .attr("x", d => bx(d.k))
          .attr("width", bx.bandwidth())
          .attr("y", d => Math.min(by(0), by(d.v)))
          .attr("height", d => Math.abs(by(d.v) - by(0)))
          .attr("fill", d =>
            d.v >= 0
              ? d3.color(futureColor).copy({{opacity: 0.60}}).formatRgb()
              : d3.color(baseColor).copy({{opacity: 0.60}}).formatRgb()
          );

      bsvg.append("text")
        .attr("x", (bw / 2))
        .attr("y", bh - 7)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Months");

      bsvg.append("text")
        .attr("x", 6)
        .attr("y", 10)
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("");
    }}

    const monthData = d3.range(1, 13).map(m => {{
      const v = monthDelta(locId, m);
      return {{k: String(m), v: Math.max(0, Math.min(5, v))}};
    }});
    renderBar("#bar_month", monthData);
  }}

  // points + interaction
  svg.append("g")
    .selectAll("circle")
    .data(points)
    .join("circle")
      .attr("cx", d => projection([+d.longitude, +d.latitude])[0])
      .attr("cy", d => projection([+d.longitude, +d.latitude])[1])
      .attr("r", 5.0)
      .attr("fill", d => {{
        const v = mapValueForLocation(d.location_id);
        if (!Number.isFinite(v)) return "#999";
        return color(clamp(Math.abs(v)));
      }})
      .attr("stroke", "#111")
      .attr("stroke-width", 0.7)
      .attr("opacity", 0.95)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {{
        const v = mapValueForLocation(d.location_id);
        tip
          .style("display", "block")
          .style("left", (event.pageX + 12) + "px")
          .style("top", (event.pageY + 12) + "px")
          .html(`
            <div style="font-weight:600; margin-bottom:4px;">${{d.location_name || d.location_id}}</div>
            <div>Annual Δ: <b>${{Number.isFinite(v) ? format1(v) : "n/a"}}</b> °C</div>
          `);
      }})
      .on("mouseleave", () => tip.style("display", "none"))
      .on("click", (_, d) => {{
        renderCharts(String(d.location_id));
      }});

  // Legend (match Plotly: 0..5 for |Δ|)
  const legendW = 180, legendH = 10;
  const legendX = 16, legendY = mapH - 34; // a tiny bit up
  const defs = svg.append("defs");
  const lg = defs.append("linearGradient").attr("id", "lg");
  const stops = d3.range(0, 1.00001, 0.1);
  lg.selectAll("stop")
    .data(stops)
    .join("stop")
      .attr("offset", d => (d * 100) + "%")
      .attr("stop-color", d => color(MINV + d * (MAXV - MINV)));

  svg.append("rect")
    .attr("x", legendX)
    .attr("y", legendY)
    .attr("width", legendW)
    .attr("height", legendH)
    .attr("fill", "url(#lg)")
    .attr("stroke", "#333")
    .attr("stroke-width", 0.6);

  const legendFont = 18.17; // 16.52px +10%
  svg.append("text")
    .attr("x", legendX)
    .attr("y", legendY - 10) // move font a bit further from bar
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("|ΔT| (°C)");

  svg.append("text")
    .attr("x", legendX)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text(MINV.toFixed(0));

  svg.append("text")
    .attr("x", legendX + legendW / 2)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("text-anchor", "middle")
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text("2.5");

  svg.append("text")
    .attr("x", legendX + legendW)
    .attr("y", legendY + 26) // a bit further from bar
    .attr("text-anchor", "end")
    .attr("fill", "#222")
    .style("font", `${{legendFont}}px Inter, Helvetica, Arial, sans-serif`)
    .text(MAXV.toFixed(0));

  if (points.length) {{
    renderCharts(String(points[0].location_id));
  }}
}})();
</script>
"""


def f206__d3_region_dashboard_html(
    *,
    points: List[Dict[str, Any]],
    profiles_bundle: Dict[str, Any],
    metric_key: str,
    width: int = 1100,
    height: int = 640,
    baseline_variant: str,
    compare_variant: str,
    percentile: float = 99.0,
    region_options: List[Dict[str, Any]],
) -> str:
    data_json = json.dumps(points, ensure_ascii=False)
    prof_json = json.dumps(profiles_bundle, ensure_ascii=False)
    region_json = json.dumps(region_options, ensure_ascii=False)

    return f"""
<!doctype html>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:wght@400;600&display=swap">
<style>
  :root {{
    --font: Inter, Helvetica, Arial, sans-serif;
    --serif-font: 'Lora', 'Source Serif Pro', 'Source Serif 4', Georgia, 'Times New Roman', serif;
    --fs: 13px;
    --fs2: 12px;
    --fg: #222;
    --muted: rgba(0,0,0,0.65);
  }}
  body {{ font-family: var(--font); }}
  .controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 0 0 10px;
  }}
  .controls label {{
    font-size: 12px;
    color: var(--muted);
  }}
  .controls select {{
    font-size: 12px;
    padding: 4px 6px;
  }}
  .layout {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }}
  .map-panel {{
    flex: 0 0 62%;
  }}
  .map-row {{
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }}
  .map-overview {{
    width: 230px;
    flex: 0 0 auto;
  }}
  .map-region {{
    width: 256px;
    flex: 0 0 auto;
  }}
  .map-overview svg,
  .map-region svg {{
    width: 100%;
    height: auto;
  }}
  .charts-panel {{
    flex: 1 1 auto;
  }}
  .panel-title {{
    font: 600 var(--fs) var(--font);
    color: var(--fg);
    margin: 2px 0 2px;
    line-height: 1.2;
  }}
  .chart-title {{
    padding-left: 50px;
  }}
  .meta {{
    font: 400 var(--fs2) var(--font);
    color: var(--muted);
    margin-bottom: 8px;
  }}
  svg text {{
    font-family: var(--font);
    font-size: var(--fs2);
    fill: var(--fg);
  }}
</style>

<div class="controls">
  <label for="region_select">Region</label>
  <select id="region_select"></select>
</div>

<div class="layout">
  <div class="map-panel">
    <div class="map-row">
      <div id="map_overview" class="map-overview"></div>
      <div id="map_region" class="map-region"></div>
    </div>
  </div>

  <div class="charts-panel">
    <div class="panel-title chart-title" id="cmp"></div>
    <div class="panel-title chart-title" id="cmp2" style="margin-top: 0px;"></div>
    <div class="meta" id="sel" style="display: none;"></div>

    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="bar_title"></div>
    <div id="bar_month"></div>

    <div style="height:10px"></div>
    <div class="panel-title chart-title" style="margin: 4px 0 2px;" id="ts_title"></div>
    <div class="panel-title chart-title" id="ts_title2" style="margin-top: 0px;"></div>
    <div class="panel-title chart-title" id="ts_title3" style="margin-top: 0px;"></div>
    <div id="scatter"></div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
(async function() {{
  const metricKey = "{metric_key}";
  const points = {data_json};
  const bundle = {prof_json};
  const keys = bundle.keys || [];
  const profiles = bundle.profiles || {{}};
  const regionOptions = {region_json};

  const PCT = {percentile};
  const P = Math.max(0.95, Math.min(1.0, (PCT / 100.0)));
  const statLabel = (metricKey === "dTmax") ? "ΔTmax" : "ΔTavg";
  const metricLabel = (metricKey === "dTmax") ? "Tmax" : "Tavg";

  function baselineLabel() {{
    const b = "{baseline_variant}";
    if (b.startsWith("tmyx_")) return "TMYx " + b.slice(5).replace("_", " ");
    if (b.startsWith("tmyx")) return "TMYx";
    return b.toUpperCase();
  }}
  function compareLabel() {{
    const c = "{compare_variant}";
    const parts = c.split("_");
    if (parts.length >= 2 && parts[0].startsWith("rcp")) {{
      const rcp = parts[0].slice(3);
      const year = parts[1];
      return "RCP " + rcp + " Year " + year;
    }}
    return c.toUpperCase();
  }}

  const regionSelect = d3.select("#region_select");
  regionSelect.selectAll("option")
    .data(regionOptions)
    .join("option")
      .attr("value", d => d.code)
      .text(d => `${{d.name}} (${{d.code}})`);

  let currentRegion = regionOptions.length ? regionOptions[0].code : null;
  regionSelect.property("value", currentRegion || "");
  regionSelect.on("change", () => {{
    currentRegion = regionSelect.property("value");
    renderRegion();
  }});

  const mapOverview = d3.select("#map_overview");
  const mapRegion = d3.select("#map_region");

  const topo = await d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json");
  const countries = topojson.feature(topo, topo.objects.countries);
  const italy = {{
    type: "FeatureCollection",
    features: countries.features.filter(d => +d.id === 380)
  }};
  const regionsUrl = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson";
  let regions = null;
  try {{
    regions = await d3.json(regionsUrl);
  }} catch (e) {{
    // ignore
  }}

  // Δ tab map colors: Viridis scale on |Δ|, clipped to 0..5°C
  const MINV = 0, MAXV = 5;
  const clamp = (x) => Math.max(MINV, Math.min(MAXV, x));
  const color = d3.scaleSequential(d3.interpolateViridis).domain([MAXV, MINV]);
  const futureColor = d3.interpolateViridis(0.12);
  const baseColor = d3.interpolateViridis(0.33);
  const futureColorStrong = d3.color(futureColor).copy({{opacity: 0.75}}).formatRgb();
  const futureColorDim = d3.color(futureColor).copy({{opacity: 0.45}}).formatRgb();
  const baseColorStrong = d3.color(baseColor).copy({{opacity: 0.75}}).formatRgb();
  const baseColorDim = d3.color(baseColor).copy({{opacity: 0.45}}).formatRgb();

  function cleanVals(arr) {{
    return (arr || []).filter(v => v != null && Number.isFinite(+v)).map(v => +v);
  }}
  function quant(vals, p) {{
    const v = cleanVals(vals).sort(d3.ascending);
    if (!v.length) return NaN;
    return d3.quantileSorted(v, p);
  }}
  function mean(vals) {{
    const v = cleanVals(vals);
    if (!v.length) return NaN;
    return d3.mean(v);
  }}

  const pointById = new Map(points.map(d => [String(d.location_id), d]));
  function mapValueForLocation(locId) {{
    const p = pointById.get(String(locId));
    if (!p) return NaN;
    return (metricKey === "dTmax") ? (+p.dTmax) : (+p.dTavg);
  }}

  const tip = mapRegion.append("div")
    .style("position", "absolute")
    .style("pointer-events", "none")
    .style("background", "rgba(255,255,255,0.95)")
    .style("border", "1px solid #ddd")
    .style("border-radius", "8px")
    .style("padding", "10px 12px")
    .style("font", "12px/1.3 Inter, Helvetica, Arial, sans-serif")
    .style("display", "none");

  function format1(x) {{
    return (Math.round(x * 10) / 10).toFixed(1);
  }}

  function getPairs(locId) {{
    const p = profiles[locId];
    if (!p) return [];
    const base = (p.base) || [];
    const comp = (p.comp) || [];
    const out = [];
    for (let i = 0; i < keys.length; i++) {{
      const b = base[i];
      const c = comp[i];
      if (b == null || c == null) continue;
      const month = keys[i][0];
      const day = keys[i][1];
      out.push({{ month, day, base: +b, comp: +c, delta: (+c) - (+b), doy: i + 1 }});
    }}
    return out;
  }}

  function monthDelta(locId, month) {{
    const p = profiles[String(locId)];
    if (!p) return NaN;
    const base = [];
    const comp = [];
    for (let i = 0; i < keys.length; i++) {{
      if (keys[i][0] !== month) continue;
      const b = p.base[i];
      const c = p.comp[i];
      if (b == null || c == null) continue;
      base.push(+b);
      comp.push(+c);
    }}
    if (metricKey === "dTmax") {{
      return quant(comp, P) - quant(base, P);
    }}
    return mean(comp) - mean(base);
  }}

  function renderCharts(locId) {{
    const meta = profiles[locId];
    if (!meta) return;
    document.getElementById("sel").textContent = "";

    const cmp = document.getElementById("cmp");
    if (cmp) {{
      cmp.innerHTML = (`Station: <b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    }}
    const cmp2 = document.getElementById("cmp2");
    if (cmp2) {{
      cmp2.innerHTML = (`Scenario: <b>${{compareLabel()}}</b> vs <b>${{baselineLabel()}}</b>`);
    }}
    const barTitle = document.getElementById("bar_title");
    if (barTitle) {{
      if (metricKey === "dTmax") {{
        barTitle.textContent = ("Monthly ΔTmax - " + PCT.toFixed(1) + "th perc. (difference)");
      }} else {{
        barTitle.textContent = ("Monthly ΔTavg (mean)");
      }}
    }}
    const tsTitle = document.getElementById("ts_title");
    if (tsTitle) {{
      tsTitle.innerHTML = (`Station: <b>${{locId}}</b> - <b>${{meta.name}}</b>`);
    }}
    const tsTitle2 = document.getElementById("ts_title2");
    if (tsTitle2) {{
      tsTitle2.innerHTML = (`Scenario: <b>${{compareLabel()}}</b> vs <b>${{baselineLabel()}}</b>`);
    }}
    const tsTitle3 = document.getElementById("ts_title3");
    if (tsTitle3) {{
      if (metricKey === "dTmax") {{
        tsTitle3.textContent = ("Daily ΔTmax - " + PCT.toFixed(1) + "th perc. - Hottest month");
      }} else {{
        tsTitle3.textContent = ("Daily ΔTavg (mean) - Hottest month");
      }}
    }}

    const pairs = getPairs(locId);
    const monthScores = d3.rollup(
      pairs,
      v => d3.max(v, d => d.comp),
      d => d.month
    );
    const warmMonth = monthScores.size
      ? Array.from(monthScores.entries()).sort((a, b) => b[1] - a[1])[0][0]
      : null;

    // --- Time series (warmest month only) ---
    const scatterPairs = warmMonth ? pairs.filter(d => d.month === warmMonth) : [];
    const sw = 240, sh = 200;
    const sm = {{t: 26, r: 14, b: 44, l: 50}};
    d3.select("#scatter").selectAll("*").remove();
    const ssvg = d3.select("#scatter").append("svg").attr("viewBox", [0, 0, sw, sh]);

    function labelFromDoy(doy) {{
      const k = keys[(doy|0) - 1];
      if (!k) return String(doy);
      return `${{k[0]}}/${{k[1]}}`;
    }}

    if (!scatterPairs.length) {{
      ssvg.append("text")
        .attr("x", sm.l)
        .attr("y", sm.t + 16)
        .style("font", "12px Inter, Helvetica, Arial, sans-serif")
        .attr("fill", "rgba(0,0,0,0.65)")
        .text("No July/August daily data available for this location.");
    }} else {{
      const x = d3.scaleLinear()
        .domain(d3.extent(scatterPairs, d => d.doy) || [182, 243])
        .range([sm.l, sw - sm.r]);

      const y = d3.scaleLinear()
        .domain([0, 40])
        .range([sh - sm.b, sm.t]);

      const tickVals = scatterPairs
        .map(d => d.doy)
        .filter((d, i, a) => a.indexOf(d) === i)
        .filter((d, i) => i % 7 === 0);

      ssvg.append("g")
        .attr("transform", `translate(0,${{sh - sm.b}})`)
        .call(d3.axisBottom(x).tickValues(tickVals).tickFormat(labelFromDoy))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("g")
        .attr("transform", `translate(${{sm.l}},0)`)
        .call(
          d3.axisLeft(y)
            .tickValues([0, 10, 20, 30, 40])
            .tickFormat(d => `${{d}} ºC`)
        )
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      ssvg.append("text")
        .attr("x", (sw / 2))
        .attr("y", sh - 11)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Days");

      const lineBase = d3.line().x(d => x(d.doy)).y(d => y(d.base));
      const lineComp = d3.line().x(d => x(d.doy)).y(d => y(d.comp));
      const sorted = scatterPairs.slice().sort((a,b)=>a.doy-b.doy);

      ssvg.append("path")
        .datum(sorted)
        .attr("fill", "none")
        .attr("stroke", baseColorStrong)
        .attr("stroke-width", 1.5)
        .attr("d", lineBase);

      ssvg.append("path")
        .datum(sorted)
        .attr("fill", "none")
        .attr("stroke", futureColorStrong)
        .attr("stroke-width", 1.5)
        .attr("d", lineComp);

      ssvg.append("g")
        .selectAll("circle.base")
        .data(sorted)
        .join("circle")
          .attr("class", "base")
          .attr("cx", d => x(d.doy))
          .attr("cy", d => y(d.base))
          .attr("r", 2.2)
          .attr("fill", baseColorDim)
          .attr("stroke", "rgba(20,20,20,0.15)");

      ssvg.append("g")
        .selectAll("circle.comp")
        .data(sorted)
        .join("circle")
          .attr("class", "comp")
          .attr("cx", d => x(d.doy))
          .attr("cy", d => y(d.comp))
          .attr("r", 2.2)
          .attr("fill", futureColorDim)
          .attr("stroke", "rgba(20,20,20,0.15)");

      const baseLegend = baselineLabel();
      const compLegend = compareLabel();
      const lx = sw - sm.r - 160, ly = 8;
      const lg = ssvg.append("g").attr("transform", `translate(${{lx}},${{ly}})`);
      lg.append("circle").attr("r", 4).attr("cx", 0).attr("cy", 0).attr("fill", baseColorStrong);
      lg.append("text").attr("x", 10).attr("y", 4).style("font-size", "7.43px").text(baseLegend);
      lg.append("circle").attr("r", 4).attr("cx", 0).attr("cy", 18).attr("fill", futureColorStrong);
      lg.append("text").attr("x", 10).attr("y", 22).style("font-size", "7.43px").text(compLegend);
    }}

    // --- Monthly bars (delta) ---
    function renderBar(selector, data) {{
      const bw = 230, bh = 127;
      const bm = {{t: 10, r: 10, b: 42, l: 50}};
      d3.select(selector).selectAll("*").remove();
      const bsvg = d3.select(selector).append("svg").attr("viewBox", [0, 0, bw, bh]);

      const bx = d3.scaleBand().domain(data.map(d => d.k)).range([bm.l, bw - bm.r]).padding(0.15);
      const extent = d3.extent(data, d => d.v) || [0, 1];
      const by = d3.scaleLinear().domain(extent).nice().range([bh - bm.b, bm.t]);

      bsvg.append("g")
        .attr("transform", `translate(0,${{bh - bm.b}})`)
        .call(d3.axisBottom(bx))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));
      bsvg.append("g")
        .attr("transform", `translate(${{bm.l}},0)`)
        .call(d3.axisLeft(by).ticks(5).tickFormat(d => `${{d}} ºC`))
        .call(g => g.selectAll("text").style("font-size", "8.26px"));

      bsvg.append("g").selectAll("rect")
        .data(data)
        .join("rect")
          .attr("x", d => bx(d.k))
          .attr("width", bx.bandwidth())
          .attr("y", d => Math.min(by(0), by(d.v)))
          .attr("height", d => Math.abs(by(d.v) - by(0)))
          .attr("fill", d =>
            d.v >= 0
              ? d3.color(futureColor).copy({{opacity: 0.60}}).formatRgb()
              : d3.color(baseColor).copy({{opacity: 0.60}}).formatRgb()
          );

      bsvg.append("text")
        .attr("x", (bw / 2))
        .attr("y", bh - 7)
        .attr("text-anchor", "middle")
        .style("font", "8.44px Inter, Helvetica, Arial, sans-serif")
        .text("Months");
    }}

    const monthData = d3.range(1, 13).map(m => {{
      const v = monthDelta(locId, m);
      return {{k: String(m), v: Math.max(0, Math.min(5, v))}};
    }});
    renderBar("#bar_month", monthData);
  }}

  function renderRegion() {{
    if (!currentRegion || !regions || !regions.features) {{
      mapOverview.selectAll("*").remove();
      mapRegion.selectAll("*").remove();
      return;
    }}

    const regionMeta = regionOptions.find(r => r.code === currentRegion);
    const regionIstat = regionMeta ? regionMeta.istat : null;
    const regionFeature = regions.features.find(
      f => +f.properties.reg_istat_code_num === +regionIstat
    );

    const regionPoints = points.filter(p => String(p.region) === String(currentRegion));

    // Small overview Italy map
    mapOverview.selectAll("*").remove();
    const ovW = 230, ovH = 190;
    const ovSvg = mapOverview.append("svg").attr("viewBox", [0, 0, ovW, ovH]);
    const ovPad = 6;
    const ovProj = d3.geoMercator().fitExtent([[ovPad, ovPad], [ovW - ovPad, ovH - ovPad]], italy);
    const ovPath = d3.geoPath(ovProj);
    ovSvg.append("path")
      .datum(italy.features[0])
      .attr("d", ovPath)
      .attr("fill", "#f7f7f7")
      .attr("stroke", "#333")
      .attr("stroke-width", 0.8);
    if (regionFeature) {{
      const hl = d3.color(d3.interpolateViridis(0.33)).copy({{opacity: 0.45}}).formatRgb();
      ovSvg.append("path")
        .datum(regionFeature)
        .attr("d", ovPath)
        .attr("fill", hl)
        .attr("stroke", "rgba(40,40,40,0.65)")
        .attr("stroke-width", 0.6);
    }}

    // Main region map (50% smaller than previous)
    mapRegion.selectAll("*").remove();
    const mapW = 256, mapH = 256;
    const svg = mapRegion.append("svg").attr("viewBox", [0, 0, mapW, mapH]);
    if (regionFeature) {{
      const pad = 10;
      const projection = d3.geoMercator().fitExtent([[pad, pad], [mapW - pad, mapH - pad]], regionFeature);
      const path = d3.geoPath(projection);
      svg.append("path")
        .datum(regionFeature)
        .attr("d", path)
        .attr("fill", "#f7f7f7")
        .attr("stroke", "#333")
        .attr("stroke-width", 1);

      svg.append("g")
        .selectAll("circle")
        .data(regionPoints)
        .join("circle")
          .attr("cx", d => projection([+d.longitude, +d.latitude])[0])
          .attr("cy", d => projection([+d.longitude, +d.latitude])[1])
          .attr("r", 200)
          .attr("fill", d => {{
            const v = mapValueForLocation(d.location_id);
            if (!Number.isFinite(v)) return "#999";
            return color(clamp(Math.abs(v)));
          }})
          .attr("stroke", "#111")
          .attr("stroke-width", 0.6)
          .attr("opacity", 0.95)
          .style("cursor", "pointer")
          .on("mousemove", (event, d) => {{
            const v = mapValueForLocation(d.location_id);
            tip
              .style("display", "block")
              .style("left", (event.pageX + 12) + "px")
              .style("top", (event.pageY + 12) + "px")
              .html(`
                <div style="font-weight:600; margin-bottom:4px;">${{d.location_name || d.location_id}}</div>
                <div>Annual Δ: <b>${{Number.isFinite(v) ? format1(v) : "n/a"}}</b> °C</div>
              `);
          }})
          .on("mouseleave", () => tip.style("display", "none"))
          .on("click", (_, d) => {{
            renderCharts(String(d.location_id), variant);
          }});

      svg.append("g")
        .selectAll("text.label")
        .data(regionPoints)
        .join("text")
          .attr("class", "label")
          .attr("x", d => projection([+d.longitude, +d.latitude])[0] + 6)
          .attr("y", d => projection([+d.longitude, +d.latitude])[1] + 4)
          .style("font-size", "8px")
          .style("fill", "#111")
          .style("paint-order", "stroke")
          .style("stroke", "rgba(255,255,255,0.85)")
          .style("stroke-width", "2px")
          .text(d => d.location_name || d.location_id);
    }}

    if (regionPoints.length) {{
      renderCharts(String(regionPoints[0].location_id));
    }}
  }}

  renderRegion();
}})();
</script>
"""


def f207__d3_region_maps_html(
    *,
    baseline_variants: List[str],
    abs_points_by_variant: Dict[str, List[Dict[str, Any]]],
    delta_points_by_variant: Dict[str, List[Dict[str, Any]]],
    metric_key: str,
    compare_variant: str,
    percentile: float,
    region_options: List[Dict[str, Any]],
    future_abs_points_by_variant: Dict[str, List[Dict[str, Any]]] | None = None,
    initial_region: str | None = None,
    force_region_code: str | None = None,
    hide_region_selector: bool = False,
    layout_mode: str = "default",
    cti_on_top: bool = False,
    hide_row2: bool = False,
    side_text_html: str | None = None,
    dashboard_cols: str = "5fr 3fr 2fr",
    maps_row_gap_px: int = 12,
    maps_row3_gap_px: int = 12,
    click_callback_key: str | None = None,
    profiles_bundle_by_variant: Dict[str, Dict[str, Any]] | None = None,
) -> str:
    variants_json = json.dumps(baseline_variants, ensure_ascii=False)
    abs_json = json.dumps(abs_points_by_variant, ensure_ascii=False)
    delta_json = json.dumps(delta_points_by_variant, ensure_ascii=False)
    region_json = json.dumps(region_options, ensure_ascii=False)
    future_abs_json = json.dumps(future_abs_points_by_variant or dict(), ensure_ascii=False)
    initial_region_str = json.dumps(initial_region) if initial_region else "null"
    force_region_str = json.dumps(force_region_code) if force_region_code else "null"
    callback_key_json = json.dumps(click_callback_key) if click_callback_key else "null"
    profiles_bundle_json = json.dumps(profiles_bundle_by_variant or dict(), ensure_ascii=False)
    layout_mode_str = str(layout_mode or "default")
    cti_on_top_str = "true" if cti_on_top else "false"
    hide_row2_str = "true" if hide_row2 else "false"
    side_text_html_str = side_text_html or ""
    dashboard_cols_str = str(dashboard_cols or "5fr 3fr 2fr")
    maps_row_gap_str = str(int(maps_row_gap_px))
    maps_row3_gap_str = str(int(maps_row3_gap_px))
    if layout_mode_str == "columns":
        layout_block = """
<div id="cti_top_container"></div>
<div class="dashboard-grid">
  <div>
    <div class="maps-container" id="maps_container"></div>
    <div style="margin-top: 0; border-top: 0px solid rgba(0,0,0,0.1); padding-top: 12px;">
      <div id="charts_location_info" style="margin-bottom: 12px; font-size: 13px; color: rgba(0,0,0,0.7);">
        Click on a map marker above to view charts
      </div>
      <div style="margin-bottom: 30px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div style="font-weight: 600; font-size: 12px;">Yearly Daily Temperature - All Scenarios</div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <label for="time_period_select" style="font-size: 11px;">Time Period:</label>
            <select id="time_period_select" style="font-size: 11px; padding: 4px 8px;">
              <option value="entire_year">Entire Year</option>
              <option value="summer">Summer</option>
              <option value="winter">Winter</option>
              <option value="january">January</option>
              <option value="february">February</option>
              <option value="march">March</option>
              <option value="april">April</option>
              <option value="may">May</option>
              <option value="june">June</option>
              <option value="july">July</option>
              <option value="august">August</option>
              <option value="september">September</option>
              <option value="october">October</option>
              <option value="november">November</option>
              <option value="december">December</option>
            </select>
          </div>
        </div>
        <div id="scatter_combined"></div>
      </div>
    </div>
  </div>
  <div>
    <div class="thermo-wrapper" id="thermo_column">
      <div id="thermo_location_caption" class="thermo-location-caption">Click on a map marker to view temperature</div>
      <div class="thermo-row">
        <div class="thermo-cell" id="thermo_cell_tmyx">
          <div id="thermo_container_tmyx" class="thermo-container"></div>
          <div id="thermo_label_tmyx" class="thermo-scenario-label">TMYx</div>
          <div id="thermo_value_tmyx" class="thermo-value"></div>
        </div>
        <div class="thermo-cell" id="thermo_cell_rcp45_2050">
          <div id="thermo_container_rcp45_2050" class="thermo-container"></div>
          <div id="thermo_label_rcp45_2050" class="thermo-scenario-label">RCP 4.5 2050</div>
          <div id="thermo_value_rcp45_2050" class="thermo-value"></div>
        </div>
        <div class="thermo-cell" id="thermo_cell_rcp85_2080">
          <div id="thermo_container_rcp85_2080" class="thermo-container"></div>
          <div id="thermo_label_rcp85_2080" class="thermo-scenario-label">RCP 8.5 2080</div>
          <div id="thermo_value_rcp85_2080" class="thermo-value"></div>
        </div>
      </div>
    </div>
  </div>
</div>
"""
    else:
        layout_block = """
<div class="maps-container" id="maps_container"></div>

<!-- Charts section below maps -->
<div style="margin-top: 30px; border-top: 1px solid rgba(0,0,0,0.1); padding-top: 20px;">
  <div class="section-title" style="margin-bottom: 16px;">Charts</div>
  <div id="charts_location_info" style="margin-bottom: 12px; font-size: 13px; color: rgba(0,0,0,0.7);">
    Click on a map marker above to view charts
  </div>
  
  <!-- Combined scatter chart for all three scenarios -->
  <div style="margin-bottom: 30px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <div style="font-weight: 600; font-size: 12px;">Yearly Daily Temperature - All Scenarios</div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <label for="time_period_select" style="font-size: 11px;">Time Period:</label>
        <select id="time_period_select" style="font-size: 11px; padding: 4px 8px;">
          <option value="entire_year">Entire Year</option>
          <option value="summer">Summer</option>
          <option value="winter">Winter</option>
          <option value="january">January</option>
          <option value="february">February</option>
          <option value="march">March</option>
          <option value="april">April</option>
          <option value="may">May</option>
          <option value="june">June</option>
          <option value="july">July</option>
          <option value="august">August</option>
          <option value="september">September</option>
          <option value="october">October</option>
          <option value="november">November</option>
          <option value="december">December</option>
        </select>
      </div>
    </div>
    <div id="scatter_combined"></div>
  </div>
</div>
"""

    _css_root = (
        "  :root {\n"
        "    --font: Inter, Helvetica, Arial, sans-serif;\n"
        "    --fs: 13px;\n"
        "    --fs2: 12px;\n"
        "    --fg: #222;\n"
        "    --muted: rgba(0,0,0,0.65);\n"
        "  }\n"
        "  body {\n"
        "    font-family: var(--font);\n"
        "    width: 100%;\n"
        "    max-width: 100%;\n"
        "    margin: 0;\n"
        "    padding: 0;\n"
        "    box-sizing: border-box;\n"
        "  }\n"
        "  * { box-sizing: border-box; }\n"
        "  .controls { display: flex; gap: 8px; align-items: center; margin: 0 0 50px; }\n"
        "  .controls label { font-size: 12px; color: var(--muted); }\n"
        "  .controls select { font-size: 12px; padding: 4px 6px; }\n"
        "  .section-title { font: 600 var(--fs) var(--font); color: var(--fg); margin: 0 0 6px; }\n"
        "  .maps-container { display: flex; justify-content: flex-start; flex-direction: column; gap: 10px; align-items: flex-start; width: 100%; max-width: 100%; }\n"
        "  .maps-row { display: flex; flex-direction: row; gap: "
    )
    return f"""
<!doctype html>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap">
<style>
{_css_root}{maps_row_gap_str}px;
    align-items: flex-start;
    width: 100%;
    flex-wrap: nowrap;
  }}
  .maps-row-3 {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: {maps_row3_gap_str}px;
    width: 100%;
  }}
  .map-card {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    flex: 1 1 auto;
    min-width: 0;
    max-width: 100%;
  }}
  .map-title {{
    font-size: 11.7px;
    color: var(--muted);
    margin: 0;
    line-height: 1.15;
    height: 16px; /* consistent baseline alignment */
    display: flex;
    align-items: flex-end;
    justify-content: center;
    font-weight: 600;
    background: #fff;
    padding: 2px 4px;
    border-radius: 3px;
  }}
  .map-value-label {{
    font-size: 9px;
    fill: #111;
    paint-order: stroke;
    stroke: rgba(255,255,255,0.85);
    stroke-width: 1.7px;
    font-weight: 500;
  }}
  .map-card svg {{
    width: 100%;
    max-width: 100%;
    height: auto;
    min-width: 200px;
    background: #fff;
  }}
  .thermo-wrapper {{
    width: 100%;
  }}
  .thermo-location-caption {{
    margin-bottom: 10px;
    font-size: 13px;
    color: rgba(0,0,0,0.75);
  }}
  .thermo-row {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: {maps_row3_gap_str}px;
    align-items: end;
    width: 100%;
  }}
  .thermo-cell {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }}
  .thermo-scenario-label {{
    font-size: 10px;
    color: var(--muted);
    font-weight: 600;
    text-align: center;
  }}
  .thermo-value {{
    font-size: 12px;
    font-weight: 700;
    color: #222;
    text-align: center;
  }}
  .thermo-container svg {{
    width: 100%;
    max-width: 160px;
  }}
  .dashboard-grid {{
    display: grid;
    grid-template-columns: {dashboard_cols_str};
    gap: {maps_row_gap_str}px;
    width: 100%;
    align-items: start;
  }}
  .dashboard-tabs {{
    margin-top: 10px;
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
  }}
  .tab-buttons {{
    display: flex;
    gap: 6px;
    padding: 6px 8px;
    border-bottom: 1px solid rgba(0,0,0,0.12);
    background: #fafafa;
  }}
  .tab-btn {{
    border: 1px solid rgba(0,0,0,0.15);
    background: #fff;
    color: #222;
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 4px;
    cursor: pointer;
  }}
  .tab-btn.active {{
    background: #f0f0f0;
    font-weight: 600;
  }}
  .tab-panel {{
    display: none;
    padding: 10px 12px 12px 12px;
  }}
  .tab-panel.active {{
    display: block;
  }}
  .thermo-caption {{
    margin-top: 6px;
    font-size: 13px;
    color: rgba(0,0,0,0.75);
  }}
</style>

<div class="controls" id="region_controls" style="display: {('none' if hide_region_selector else 'flex')};">
  <label for="region_select" style="font-weight: 600;">Region</label>
  <select id="region_select"></select>
</div>

<div class="section-title" id="abs_title"></div>
<div id="scenarios_subtitle" style="display: {('none' if hide_region_selector else 'block')}; margin: 8px 0 12px; font-weight: 600; font-size: 0.95rem;">Current vs Future Climate Scenarios</div>
{layout_block}

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
(async function() {{
  const baselineVariants = {variants_json};
  const absByVar = {abs_json};
  const deltaByVar = {delta_json};
  const callbackKey = {callback_key_json};
  const futureAbsByVar = {future_abs_json} || {{}};
  const regionOptions = {region_json} || [];
  const profilesByVariant = {profiles_bundle_json} || {{}};
  const metricKey = "{metric_key}";
  const compareVariant = "{compare_variant}";
  const forceRegionCode = {force_region_str};
  const ctiOnTop = {cti_on_top_str};
  const hideRow2 = {hide_row2_str};
  const PCT = {percentile};
  
  // Extract keys and profiles from bundles
  const variantKeys = {{}};
  const variantProfiles = {{}};
  for (const [variant, bundle] of Object.entries(profilesByVariant)) {{
    if (bundle && bundle.keys && bundle.profiles) {{
      variantKeys[variant] = bundle.keys;
      variantProfiles[variant] = bundle.profiles;
    }}
  }}

  const absTitle = document.getElementById("abs_title");
  if (absTitle) {{
    absTitle.textContent = (metricKey === "dTmax")
      ? `Tmax (${{PCT.toFixed(1)}}th perc.)`
      : "Tavg (mean)";
  }}

  let currentRegion = null;
  const regionSelect = d3.select("#region_select");
  const STORAGE_KEY = "fwg_single_regions_region";
  const hideRegionSelector = {str(hide_region_selector).lower()};
  
  // Create a single tooltip element to be reused by all maps
  const tip = d3.select("body").append("div")
    .attr("class", "map-tooltip")
    .style("position", "absolute")
    .style("padding", "6px 10px")
    .style("background", "rgba(0, 0, 0, 0.85)")
    .style("color", "#fff")
    .style("border-radius", "4px")
    .style("font-size", "11px")
    .style("pointer-events", "none")
    .style("opacity", 0)
    .style("z-index", 1000);

  function safeGetStoredRegion() {{
    try {{
      return localStorage.getItem(STORAGE_KEY);
    }} catch (e) {{
      return null;
    }}
  }}
  function safeSetStoredRegion(code) {{
    try {{
      localStorage.setItem(STORAGE_KEY, String(code || ""));
    }} catch (e) {{
      // ignore
    }}
  }}

  if (regionOptions && Array.isArray(regionOptions) && regionOptions.length > 0) {{
    regionSelect.selectAll("option")
      .data(regionOptions)
      .join("option")
        .attr("value", d => d.code)
        .text(d => `${{d.name}} (${{d.code}})`);

    const stored = safeGetStoredRegion();
    const storedValid = stored && regionOptions.find(r => r.code === stored);

    if (storedValid) {{
      currentRegion = stored;
    }} else {{
      // Use initial_region if provided, otherwise use first option
      const initialRegionCode = {initial_region_str};
      if (initialRegionCode && initialRegionCode !== null && initialRegionCode !== "None" && initialRegionCode !== "") {{
        const foundRegion = regionOptions.find(r => r.code === initialRegionCode);
        currentRegion = foundRegion ? initialRegionCode : regionOptions[0].code;
      }} else {{
        currentRegion = regionOptions[0].code;
      }}
      // Only set storage if nothing valid was already stored (avoid overriding the "main" selector)
      safeSetStoredRegion(currentRegion);
    }}

    if (forceRegionCode && regionOptions.find(r => r.code === forceRegionCode)) {{
      currentRegion = forceRegionCode;
      safeSetStoredRegion(currentRegion);
    }}

    regionSelect.property("value", currentRegion || "");
    regionSelect.on("change", () => {{
      currentRegion = regionSelect.property("value");
      safeSetStoredRegion(currentRegion);
      renderAll();
    }});

    // Keep in sync with the other iframe (top block) via localStorage
    window.addEventListener("storage", (e) => {{
      if (!e) return;
      if (e.key !== STORAGE_KEY) return;
      const nv = e.newValue;
      if (!nv) return;
      if (nv === currentRegion) return;
      const ok = regionOptions.find(r => r.code === nv);
      if (!ok) return;
      currentRegion = nv;
      regionSelect.property("value", currentRegion || "");
      renderAll();
    }});

    // If selector is hidden, still reflect stored region immediately (in case storage event didn't fire yet)
    if (hideRegionSelector) {{
      const s2 = safeGetStoredRegion();
      if (s2 && s2 !== currentRegion && regionOptions.find(r => r.code === s2)) {{
        currentRegion = s2;
        regionSelect.property("value", currentRegion || "");
      }}
      // Ensure currentRegion is set even if no stored value
      if (!currentRegion && regionOptions.length > 0) {{
        currentRegion = regionOptions[0].code;
      }}
    }}
  }} else {{
    console.warn("No region options available");
    // Even if no region options, try to set a default if we have stored value
    if (!currentRegion) {{
      const s3 = safeGetStoredRegion();
      if (s3) currentRegion = s3;
    }}
  }}

  const topo = await d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json");
  const countries = topojson.feature(topo, topo.objects.countries);
  const italy = {{
    type: "FeatureCollection",
    features: countries.features.filter(d => +d.id === 380)
  }};
  const regionsUrl = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson";
  let regions = null;
  try {{
    regions = await d3.json(regionsUrl);
  }} catch (e) {{
    // ignore
  }}
  const provincesUrl = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson";
  let provinces = null;
  try {{
    provinces = await d3.json(provincesUrl);
  }} catch (e) {{
    // ignore
  }}

  const tempScale = d3.scaleSequential(t => d3.interpolateRdBu(1 - t)).domain([0, 40]);
  const deltaScale = d3.scaleSequential(t => d3.interpolateRdBu(1 - t)).domain([0, 5]);
  let combinedAbsByVar = {{}};

  function variantLabel(v) {{
    if (v.startsWith("tmyx_")) return "TMYx " + v.slice(5).replace("_", "-");
    if (v.startsWith("tmyx")) return "TMYx";
    // Format RCP scenarios: rcp45_2050 -> "RCP 4.5 Year 2050"
    if (v.startsWith("rcp")) {{
      const parts = v.split("_");
      if (parts.length === 2) {{
        const rcpPart = parts[0]; // e.g., "rcp45"
        const rcpNum = rcpPart.replace("rcp", ""); // e.g., "45"
        const year = parts[1];
        if (rcpNum.length === 2) {{
          return `RCP ${{rcpNum[0]}}.${{rcpNum[1]}} Year ${{year}}`;
        }}
        return `RCP ${{rcpNum}} Year ${{year}}`;
      }}
    }}
    return v.toUpperCase();
  }}

  function getRegionFeature() {{
    if (!regions || !regions.features || !currentRegion) return null;
    const meta = regionOptions.find(r => r.code === currentRegion);
    const istat = meta ? meta.istat : null;
    if (!Number.isFinite(istat)) return null;
    return regions.features.find(f => +f.properties.reg_istat_code_num === +istat) || null;
  }}

  function getProvincesInRegion(regionIstat) {{
    if (!provinces || !provinces.features || !Number.isFinite(regionIstat)) return [];
    return provinces.features.filter(f => +f.properties.reg_istat_code_num === +regionIstat);
  }}

  function renderMap(card, variant, dataByVariant, scale, clampMax) {{
    const allPoints = dataByVariant[variant] || [];
    let points = allPoints.filter(p => String(p.region) === String(currentRegion));
    // CTI: if region filter leaves no points but we have data, show all CTI points so markers are visible
    if (variant === "cti" && points.length === 0 && allPoints.length > 0) points = allPoints;
    const regionFeature = getRegionFeature();
    
    // Skip only if no points AND no region feature
    if (!points.length && !regionFeature) {{
      return;
    }}

    card.append("div").attr("class", "map-title").text(variantLabel(variant));
    // Use responsive sizing - base size but will scale with container
    const baseSize = 240;
    const mapW = baseSize, mapH = baseSize;
    const svg = card.append("svg")
      .attr("viewBox", [0, 0, mapW, mapH])
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("width", "100%")
      .style("height", "auto");

    const pad = 8;
    let projection = null;
    let path = null;

    if (regionFeature) {{
      projection = d3.geoMercator().fitExtent([[pad, pad], [mapW - pad, mapH - pad]], regionFeature);
      path = d3.geoPath(projection);
      svg.append("path")
        .datum(regionFeature)
        .attr("d", path)
        .attr("fill", "#f7f7f7")
        .attr("stroke", "#333")
        .attr("stroke-width", 1.8);
      // Province outlines within the region (thinner than region)
      const meta = regionOptions.find(r => r.code === currentRegion);
      const regionIstat = meta && Number.isFinite(meta.istat) ? meta.istat : null;
      const provinceFeatures = getProvincesInRegion(regionIstat);
      if (provinceFeatures.length && path) {{
        svg.selectAll("path.province")
          .data(provinceFeatures)
          .join("path")
          .attr("class", "province")
          .attr("d", path)
          .attr("fill", "none")
          .attr("stroke", "#555")
          .attr("stroke-width", 0.4)
          .attr("stroke-opacity", 0.9);
      }}
    }}
    
    // If no regionFeature but we have points, create projection from points
    if (!projection && points.length) {{
      const mp = {{
        type: "Feature",
        geometry: {{
          type: "MultiPoint",
          coordinates: points.map(p => [+p.longitude, +p.latitude]),
        }},
      }};
      projection = d3.geoMercator().fitExtent([[pad, pad], [mapW - pad, mapH - pad]], mp);
    }}

    // Render markers if we have points and a projection
    if (points.length && projection) {{
      svg.append("g")
        .selectAll("circle")
        .data(points)
        .join("circle")
          .attr("cx", d => projection([+d.longitude, +d.latitude])[0])
          .attr("cy", d => projection([+d.longitude, +d.latitude])[1])
          .attr("r", 4.6)
          .attr("fill", d => {{
            const val = Number.isFinite(+d.value) ? +d.value : NaN;
            if (!Number.isFinite(val)) return "#999";
            const v = clampMax ? Math.max(0, Math.min(clampMax, val)) : val;
            return scale(v);
          }})
          .attr("stroke", "#111")
          .attr("stroke-width", 0.6)
          .attr("opacity", 0.95)
          .style("cursor", "pointer")
          .on("mousemove", (event, d) => {{
            const val = Number.isFinite(+d.value) ? +d.value : NaN;
            const valStr = Number.isFinite(val) ? val.toFixed(1) : "n/a";
            const metricLabel = (metricKey === "dTmax") ? "Tmax" : "Tavg";
            tip
              .style("opacity", 1)
              .style("left", (event.pageX + 12) + "px")
              .style("top", (event.pageY + 12) + "px")
              .html(`
                <div style="font-weight:600; margin-bottom:2px;">${{d.location_name || d.location_id}}</div>
                <div>${{metricLabel}}: <b>${{valStr}}</b> °C</div>
              `);
          }})
          .on("mouseleave", () => {{
            tip.style("opacity", 0);
          }})
          .on("click", (_, d) => {{
            // Render charts for this location
            renderCharts(String(d.location_id), variant);
            
            // Store selection for Streamlit to read
            if (callbackKey) {{
              // Store in a data attribute on the container that Streamlit can access
              const container = d3.select("body").node();
              if (container) {{
                container.setAttribute("data-selected-location", JSON.stringify({{
                  location_id: String(d.location_id),
                  location_name: d.location_name || String(d.location_id),
                  variant: variant,
                  callback_key: callbackKey
                }}));
                // Trigger a custom event
                const event = new CustomEvent("streamlit:locationSelected", {{
                  detail: {{
                    location_id: String(d.location_id),
                    location_name: d.location_name || String(d.location_id),
                    variant: variant,
                    callback_key: callbackKey
                  }}
                }});
                window.dispatchEvent(event);
                // Also try postMessage for Streamlit
                if (window.parent && window.parent.postMessage) {{
                  window.parent.postMessage({{
                    type: "streamlit:setComponentValue",
                    value: {{
                      location_id: String(d.location_id),
                      location_name: d.location_name || String(d.location_id),
                      variant: variant
                    }}
                  }}, "*");
                }}
              }}
            }}
          }});

      // Label positioning with collision detection
      const labelGroup = svg.append("g").attr("class", "label-group");
      const labelData = points.map(d => {{
        const val = Number.isFinite(+d.value) ? +d.value : NaN;
        const text = Number.isFinite(val) ? val.toFixed(1) : "";
        const [x, y] = projection([+d.longitude, +d.latitude]);
        return {{ ...d, text, x, y }};
      }}).filter(d => d.text !== "");
      
      if (labelData.length === 0) return;
      
      // Collision detection helper
      function checkCollision(box1, box2) {{
        return !(box1.right < box2.left || box1.left > box2.right || 
                 box1.bottom < box2.top || box1.top > box2.bottom);
      }}
      
      // Get bounding box for a label at a given position
      function getLabelBox(x, y, offsetX, offsetY, text) {{
        // Approximate text dimensions (10px font)
        const width = text.length * 6 + 4; // Add padding
        const height = 14;
        // Calculate anchor point based on offset direction
        let anchorX = width / 2; // Default center
        if (offsetX < -4) anchorX = width; // Left side
        else if (offsetX > 4) anchorX = 0; // Right side
        
        return {{
          left: x + offsetX - anchorX,
          right: x + offsetX - anchorX + width,
          top: y + offsetY - height / 2,
          bottom: y + offsetY + height / 2
        }};
      }}
      
      // Calculate distance from label box center to circle center
      function distanceToCircle(box, circleX, circleY) {{
        const centerX = (box.left + box.right) / 2;
        const centerY = (box.top + box.bottom) / 2;
        return Math.sqrt(Math.pow(centerX - circleX, 2) + Math.pow(centerY - circleY, 2));
      }}
      
      // Try different positions for each label (priority order with more options)
      // Expanded set of positions to increase chances of finding non-overlapping spots
      const labelPositions = [
        {{ offsetX: 8, offsetY: 3, name: "right" }},
        {{ offsetX: -8, offsetY: 3, name: "left" }},
        {{ offsetX: 0, offsetY: -10, name: "top" }},
        {{ offsetX: 0, offsetY: 16, name: "bottom" }},
        {{ offsetX: 12, offsetY: 0, name: "far-right" }},
        {{ offsetX: -12, offsetY: 0, name: "far-left" }},
        {{ offsetX: 8, offsetY: -8, name: "top-right" }},
        {{ offsetX: -8, offsetY: -8, name: "top-left" }},
        {{ offsetX: 8, offsetY: 14, name: "bottom-right" }},
        {{ offsetX: -8, offsetY: 14, name: "bottom-left" }},
        {{ offsetX: 15, offsetY: 3, name: "very-far-right" }},
        {{ offsetX: -15, offsetY: 3, name: "very-far-left" }},
        {{ offsetX: 0, offsetY: -15, name: "very-top" }},
        {{ offsetX: 0, offsetY: 20, name: "very-bottom" }}
      ];
      
      // Position labels to avoid overlaps (process in order)
      const positionedLabels = [];
      const circleRadius = 4.6;
      const minDistanceFromCircle = 4; // Minimum distance from marker circle
      
      for (let i = 0; i < labelData.length; i++) {{
        const d = labelData[i];
        let bestPosition = labelPositions[0];
        let bestScore = Infinity;
        
        // Score each position - lower is better
        for (const pos of labelPositions) {{
          const box = getLabelBox(d.x, d.y, pos.offsetX, pos.offsetY, d.text);
          let score = 0;
          
          // Check against own marker circle (ensure minimum distance)
          const distToOwnCircle = distanceToCircle(box, d.x, d.y);
          if (distToOwnCircle < circleRadius + minDistanceFromCircle) {{
            score += 1000; // Very heavy penalty for overlapping own marker
          }} else {{
            // Prefer positions closer to marker (but not too close)
            score += Math.max(0, (circleRadius + minDistanceFromCircle) - distToOwnCircle) * 10;
          }}
          
          // Check against other markers (all points)
          for (const otherPoint of points) {{
            if (otherPoint === d) continue;
            const [otherX, otherY] = projection([+otherPoint.longitude, +otherPoint.latitude]);
            const distToOtherCircle = distanceToCircle(box, otherX, otherY);
            if (distToOtherCircle < circleRadius + minDistanceFromCircle) {{
              score += 500; // Heavy penalty for overlapping other markers
            }}
          }}
          
          // Check against already positioned labels
          for (const otherLabel of positionedLabels) {{
            const otherBox = getLabelBox(
              otherLabel.x, otherLabel.y, 
              otherLabel.offsetX, otherLabel.offsetY, 
              otherLabel.text
            );
            if (checkCollision(box, otherBox)) {{
              // Calculate overlap area
              const overlapX = Math.max(0, Math.min(box.right, otherBox.right) - Math.max(box.left, otherBox.left));
              const overlapY = Math.max(0, Math.min(box.bottom, otherBox.bottom) - Math.max(box.top, otherBox.top));
              const overlapArea = overlapX * overlapY;
              const boxArea = (box.right - box.left) * (box.bottom - box.top);
              // Penalty proportional to overlap percentage
              score += (overlapArea / boxArea) * 200;
            }}
          }}
          
          // Prefer positions that are closer (right, left, top, bottom) over far positions
          const distance = Math.sqrt(pos.offsetX * pos.offsetX + pos.offsetY * pos.offsetY);
          score += distance * 0.5; // Small penalty for distance
          
          // If this position has no collisions, use it immediately
          if (score === 0 || (score < 50 && bestScore > 50)) {{
            bestPosition = pos;
            bestScore = score;
            if (score === 0) break; // Perfect position found, stop searching
          }} else if (score < bestScore) {{
            bestPosition = pos;
            bestScore = score;
          }}
        }}
        
        // Always use the best position found - labels are always moved, never faded
        positionedLabels.push({{
          ...d, 
          offsetX: bestPosition.offsetX, 
          offsetY: bestPosition.offsetY
        }});
      }}
      
      // Render labels - always at full opacity since we're moving them to avoid overlaps
      labelGroup.selectAll("text.map-value-label")
        .data(positionedLabels)
        .join("text")
          .attr("class", "map-value-label")
          .attr("x", d => d.x + d.offsetX)
          .attr("y", d => d.y + d.offsetY)
          .attr("text-anchor", d => {{
            if (d.offsetX < -4) return "end";
            if (d.offsetX > 4) return "start";
            return "middle";
          }})
          .attr("dominant-baseline", d => {{
            if (d.offsetY < -4) return "auto";
            if (d.offsetY > 4) return "hanging";
            return "middle";
          }})
          .style("opacity", 1.0) // Always full opacity - labels are moved, not faded
          .text(d => d.text);
    }}
  }}

  // Helper function to get series pairs for a location and variant
  function seriesPairs(locId, variant) {{
    const keys = variantKeys[variant] || [];
    const profiles = variantProfiles[variant] || {{}};
    const locStr = String(locId);
    const p = profiles[locStr] || profiles[locId];
    if (!p) return [];
    const s = (p.series) || [];
    const out = [];
    const baseYear = 2001; // Base year for date calculations
    for (let i = 0; i < keys.length; i++) {{
      const v = s[i];
      if (v == null) continue;
      const k = keys[i];
      if (!k || !Array.isArray(k)) continue;
      const month = Number(k[0]) || 1;
      const day = Number(k[1]) || 1;
      const date = new Date(baseYear, month - 1, day);
      out.push({{ month, day, v: +v, doy: i + 1, date: date }});
    }}
    return out;
  }}
  
  // Helper function to filter pairs by time period
  function filterByTimePeriod(pairs, timePeriod) {{
    if (!timePeriod || timePeriod === "" || timePeriod === "entire_year") {{
      return pairs;
    }}
    if (timePeriod === "summer") {{
      // June, July, August
      return pairs.filter(d => d.month >= 6 && d.month <= 8);
    }} else if (timePeriod === "winter") {{
      // December, January, February
      return pairs.filter(d => d.month === 12 || d.month === 1 || d.month === 2);
    }} else {{
      // Single month
      const monthMap = {{
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
      }};
      const targetMonth = monthMap[timePeriod];
      if (targetMonth) {{
        return pairs.filter(d => d.month === targetMonth);
      }}
      return pairs;
    }}
  }}
  
  // Function to render scatter chart
  function renderScatterChart(locId, timePeriod) {{
    const scatterSelector = "#scatter_combined";
    d3.select(scatterSelector).selectAll("*").remove();
    
    // Scenarios to render: use baselineVariants that have profile data
    const scenarioColors = {{
      "tmyx": "#1f77b4",
      "rcp45_2050": "#ff7f0e",
      "rcp85_2080": "#d62728",
      "tmyx_2004-2018": "#2ca02c",
      "tmyx_2007-2021": "#9467bd",
      "tmyx_2009-2023": "#8c564b",
      "tmyx_2004_2018": "#2ca02c",
      "tmyx_2007_2021": "#9467bd",
      "tmyx_2009_2023": "#8c564b"
    }};
    function variantLabel(v) {{
      if (v === "tmyx") return "TMY (Current)";
      if (v === "rcp45_2050") return "RCP 4.5 2050";
      if (v === "rcp85_2080") return "RCP 8.5 2080";
      if (v && v.startsWith("tmyx_")) return "TMYx " + v.replace("tmyx_", "").replace(/_/g, "-");
      return v || "";
    }}
    function variantColor(v) {{
      return scenarioColors[v] || "#17becf";
    }}
    let scenarios = baselineVariants.filter(v => variantKeys[v] && variantProfiles[v]);
    if (scenarios.length === 0 && Object.keys(variantProfiles).length > 0) {{
      scenarios = Object.keys(variantProfiles).filter(v => variantKeys[v]);
    }}
    
    // Normalize time period (default to entire year if missing)
    const period = (timePeriod && timePeriod !== "") ? timePeriod : "entire_year";
    
    // Collect all data pairs for all scenarios
    const allPairs = [];
    scenarios.forEach(variant => {{
      const profiles = variantProfiles[variant] || {{}};
      const p = profiles[String(locId)];
      if (p && p.series) {{
        const pairs = seriesPairs(locId, variant);
        if (pairs.length > 0) {{
          const filteredPairs = filterByTimePeriod(pairs, period);
          if (filteredPairs.length > 0) {{
            allPairs.push({{
              variant: variant,
              pairs: filteredPairs.slice().sort((a, b) => a.date - b.date),
              color: variantColor(variant),
              label: variantLabel(variant)
            }});
          }}
        }}
      }}
    }});
    
    if (allPairs.length === 0) {{
      const msg = scenarios.length === 0
        ? "No profile data loaded. Click a map marker above to view charts."
        : "No data for this location or time period.";
      d3.select(scatterSelector).append("div")
        .style("padding", "20px")
        .style("color", "rgba(0,0,0,0.5)")
        .style("font-size", "12px")
        .text(msg);
      return;
    }}
    
    // Scatter chart dimensions (height reduced 15%)
    const sw = 800, sh = 340;
    const sm = {{t: 30, r: 80, b: 45, l: 55}};
    const ssvg = d3.select(scatterSelector).append("svg").attr("viewBox", [0, 0, sw, sh]);
    
    // Find global domain for x-axis (dates); y-axis fixed 0 to 45
    let dateMin = null, dateMax = null;
    const yMin = 0, yMax = 45;
    
    allPairs.forEach(item => {{
      item.pairs.forEach(d => {{
        if (d.date) {{
          if (dateMin === null || d.date < dateMin) dateMin = d.date;
          if (dateMax === null || d.date > dateMax) dateMax = d.date;
        }}
      }});
    }});
    
    // Add padding to date range
    if (dateMin && dateMax) {{
      const dateRange = dateMax - dateMin;
      dateMin = new Date(dateMin.getTime() - dateRange * 0.02);
      dateMax = new Date(dateMax.getTime() + dateRange * 0.02);
    }}
    
    // Scales
    const x = d3.scaleTime()
      .domain(dateMin && dateMax ? [dateMin, dateMax] : [new Date(2001, 0, 1), new Date(2001, 11, 31)])
      .range([sm.l, sw - sm.r]);
    
    const y = d3.scaleLinear()
      .domain([yMin, yMax])
      .range([sh - sm.b, sm.t]);
    
    // Axes
    ssvg.append("g")
      .attr("transform", `translate(0,${{sh - sm.b}})`)
      .call(d3.axisBottom(x).ticks(d3.timeMonth.every(1)))
      .call(g => g.selectAll("text").style("font-size", "10px"));
    
    ssvg.append("g")
      .attr("transform", `translate(${{sm.l}},0)`)
      .call(d3.axisLeft(y).ticks(8).tickFormat(d => `${{d}}°C`))
      .call(g => g.selectAll("text").style("font-size", "10px"));
    
    // Labels
    ssvg.append("text")
      .attr("x", sw / 2)
      .attr("y", sh - 10)
      .attr("text-anchor", "middle")
      .style("font-size", "11px")
      .text("Date");
    
    ssvg.append("text")
      .attr("x", 8)
      .attr("y", 15)
      .style("font-size", "11px")
      .text("Temperature (°C)");
    
    // Draw lines for each scenario
    const line = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.v))
      .curve(d3.curveMonotoneX);
    
    allPairs.forEach(item => {{
      // Draw line
      ssvg.append("path")
        .datum(item.pairs)
        .attr("fill", "none")
        .attr("stroke", item.color)
        .attr("stroke-width", 2)
        .attr("d", line);
    }});
    
    // Legend (50px left of previous position)
    const legendX = sw - sm.r + 10 - 50;
    const legendY = sm.t;
    const legendSpacing = 20;
    
    allPairs.forEach((item, i) => {{
      const yPos = legendY + i * legendSpacing;
      
      // Legend line
      ssvg.append("line")
        .attr("x1", legendX)
        .attr("x2", legendX + 20)
        .attr("y1", yPos)
        .attr("y2", yPos)
        .attr("stroke", item.color)
        .attr("stroke-width", 2);
      
      // Legend text
      ssvg.append("text")
        .attr("x", legendX + 25)
        .attr("y", yPos + 4)
        .style("font-size", "11px")
        .text(item.label);
    }});
  }}
  
  // Helper function to get monthly stat
  function monthStat(locId, month, variant) {{
    const P = Math.max(0.95, Math.min(1.0, (PCT / 100.0)));
    const pairs = seriesPairs(locId, variant).filter(d => d.month === month).map(d => d.v);
    if (pairs.length === 0) return NaN;
    if (metricKey === "dTmax" || metricKey === "Tmax") {{
      // For Tmax, use percentile
      const sorted = pairs.slice().sort((a, b) => a - b);
      const idx = Math.floor((sorted.length - 1) * P);
      return sorted[idx] || 0;
    }}
    // For Tavg, use mean
    const sum = pairs.reduce((a, b) => a + b, 0);
    return sum / pairs.length;
  }}
  
  // Store current location ID for time period filtering
  let currentLocationId = null;
  let thermoVariants = {{ tmyx: "tmyx", rcp45_2050: "rcp45_2050", rcp85_2080: "rcp85_2080" }};

  function getLocationName(locId) {{
    let name = locId;
    for (const variant of baselineVariants) {{
      const profiles = variantProfiles[variant] || {{}};
      const p = profiles[String(locId)];
      if (p && p.name) {{
        name = p.name;
        break;
      }}
    }}
    return name;
  }}

  function getVariantValue(locId, variant) {{
    const vKey = (combinedAbsByVar[variant] ? variant : (combinedAbsByVar["tmyx"] ? "tmyx" : null));
    if (!vKey) return {{ value: NaN, variantKey: variant }};
    const pts = combinedAbsByVar[vKey] || [];
    const row = pts.find(p => String(p.location_id) === String(locId));
    return {{
      value: row ? Number(row.value) : NaN,
      variantKey: vKey
    }};
  }}

  function renderThermometer(containerSelector, labelId, valueId, locId, variant) {{
    const container = d3.select(containerSelector);
    if (container.empty()) return;
    container.selectAll("*").remove();
    const labelEl = document.getElementById(labelId);
    if (labelEl) labelEl.textContent = variantLabel(variant || "tmyx");
    const valueEl = document.getElementById(valueId);
    if (valueEl) valueEl.textContent = "";
    if (!locId) return;
    const vKey = variant || "tmyx";
    const {{ value, variantKey }} = getVariantValue(locId, vKey);
    if (valueEl) valueEl.textContent = Number.isFinite(value) ? `${{value.toFixed(1)}} °C` : "";
    if (!Number.isFinite(value)) return;
    const maxTemp = 45;
    const v = Math.max(0, Math.min(maxTemp, value));
    const colorScale = d3.scaleSequential(t => d3.interpolateRdBu(1 - t)).domain([0, maxTemp]);
    const color = colorScale(v);

    const w = 140, h = 240;
    const tubeW = 22, tubeX = (w - tubeW) / 2, tubeY = 16;
    const bulbR = 24 * 0.9;
    const tubeH = h - tubeY - bulbR * 2 + 4;
    const bulbCy = tubeY + tubeH + bulbR - 4;
    const innerPad = 3.5;
    const innerH = tubeH - innerPad * 2;
    const fillH = (v / maxTemp) * innerH;
    const fillY = tubeY + innerPad + (innerH - fillH);

    const svg = container.append("svg")
      .attr("viewBox", [0, 0, w, h])
      .style("width", "100%")
      .style("max-width", `${{w}}px`)
      .style("height", `${{h}}px`);

    // Outline
    svg.append("rect")
      .attr("x", tubeX)
      .attr("y", tubeY)
      .attr("width", tubeW)
      .attr("height", tubeH)
      .attr("rx", 12)
      .attr("ry", 12)
      .attr("fill", "#fff")
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    svg.append("circle")
      .attr("cx", w / 2)
      .attr("cy", bulbCy)
      .attr("r", bulbR)
      .attr("fill", "#fff")
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // Mercury fill (flat top, no fillet)
    svg.append("rect")
      .attr("x", tubeX + innerPad)
      .attr("y", fillY)
      .attr("width", tubeW - innerPad * 2)
      .attr("height", fillH)
      .attr("rx", 0)
      .attr("ry", 0)
      .attr("fill", color);

    svg.append("circle")
      .attr("cx", w / 2)
      .attr("cy", bulbCy)
      .attr("r", bulbR - innerPad)
      .attr("fill", color);

    // White horizontal tick lines inside the tube (drawn on top of mercury)
    const tickVals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
    const tubeInnerLeft = tubeX + innerPad;
    const tubeInnerRight = tubeX + tubeW - innerPad;
    tickVals.forEach(t => {{
      const y = tubeY + tubeH - (t / maxTemp) * tubeH;
      svg.append("line")
        .attr("x1", tubeInnerLeft)
        .attr("x2", tubeInnerRight)
        .attr("y1", y)
        .attr("y2", y)
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.7);
    }});

    // Scale ticks (right side)
    const tickX1 = tubeX + tubeW + 8;
    const tickX2 = tickX1 + 6;
    tickVals.forEach(t => {{
      const y = tubeY + tubeH - (t / maxTemp) * tubeH;
      svg.append("line")
        .attr("x1", tickX1)
        .attr("x2", tickX2)
        .attr("y1", y)
        .attr("y2", y)
        .attr("stroke", "#666")
        .attr("stroke-width", 1);
      svg.append("text")
        .attr("x", tickX2 + 4)
        .attr("y", y + 3)
        .style("font-size", "10px")
        .style("fill", "#444")
        .text(`${{t}}`);
    }});
  }}

  function updateThermoVariants() {{
    const tmyxKey = baselineVariants.find(v => v === "tmyx") || "tmyx";
    const rcp45Key = Object.keys(combinedAbsByVar || {{}}).find(k => k.endsWith("__rcp45_2050")) || "rcp45_2050";
    const rcp85Key = Object.keys(combinedAbsByVar || {{}}).find(k => k.endsWith("__rcp85_2080")) || "rcp85_2080";
    thermoVariants = {{
      tmyx: tmyxKey,
      rcp45_2050: rcp45Key,
      rcp85_2080: rcp85Key
    }};
  }}

  function renderThermometers(locId) {{
    const locationCaption = document.getElementById("thermo_location_caption");
    if (locationCaption) {{
      if (locId) {{
        const locName = getLocationName(locId);
        locationCaption.innerHTML = `Location: <b>${{locName}} (${{locId}})</b>`;
      }} else {{
        locationCaption.textContent = "Click on a map marker to view temperature";
      }}
    }}
    const targets = [
      {{ slot: "tmyx", containerId: "#thermo_container_tmyx", labelId: "thermo_label_tmyx", valueId: "thermo_value_tmyx", cellId: "thermo_cell_tmyx" }},
      {{ slot: "rcp45_2050", containerId: "#thermo_container_rcp45_2050", labelId: "thermo_label_rcp45_2050", valueId: "thermo_value_rcp45_2050", cellId: "thermo_cell_rcp45_2050" }},
      {{ slot: "rcp85_2080", containerId: "#thermo_container_rcp85_2080", labelId: "thermo_label_rcp85_2080", valueId: "thermo_value_rcp85_2080", cellId: "thermo_cell_rcp85_2080" }}
    ];
    targets.forEach(t => {{
      const variantKey = thermoVariants[t.slot];
      const cell = document.getElementById(t.cellId);
      const hasVariant = Boolean(combinedAbsByVar && combinedAbsByVar[variantKey]);
      if (cell) cell.style.display = hasVariant ? "flex" : "none";
      if (hasVariant) {{
        renderThermometer(t.containerId, t.labelId, t.valueId, locId, variantKey);
      }}
    }});
  }}
  
  // Render charts for a location (all three scenarios)
  function renderCharts(locId, variant) {{
    currentLocationId = String(locId);
    
    const locationInfo = document.getElementById("charts_location_info");
    if (locationInfo) {{
      // Find location name from any variant
      const locName = getLocationName(locId);
      locationInfo.innerHTML = `Location: <b>${{locName}} (${{locId}})</b>`;
    }}
    
    // Get current time period selection
    const timePeriodSelect = document.getElementById("time_period_select");
    const timePeriod = timePeriodSelect ? timePeriodSelect.value : "entire_year";
    
    // Render scatter chart
    renderScatterChart(locId, timePeriod);
    renderThermometers(locId);
  }}
  
  // Add event listener for time period selector (set up after DOM is ready)
  setTimeout(function() {{
    const timePeriodSelect = document.getElementById("time_period_select");
    if (timePeriodSelect) {{
      timePeriodSelect.addEventListener("change", function() {{
        if (currentLocationId) {{
          renderScatterChart(currentLocationId, this.value);
        }}
      }});
    }}

    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {{
      btn.addEventListener("click", () => {{
        tabButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab-panel").forEach(panel => {{
          panel.classList.toggle("active", panel.id === `tab_${{tabId}}`);
        }});
      }});
    }});
  }}, 100);

  function renderAll() {{
    if (!currentRegion) {{
      console.warn("No region selected, cannot render maps");
      return;
    }}
    
    const container = d3.select("#maps_container");
    container.selectAll("*").remove();
    const ctiContainer = d3.select("#cti_top_container");
    if (ctiContainer && !ctiContainer.empty()) {{
      ctiContainer.selectAll("*").remove();
    }}
    
    // Combine baseline and future data
    combinedAbsByVar = {{...absByVar}};
    if (futureAbsByVar && Object.keys(futureAbsByVar).length > 0) {{
      Object.assign(combinedAbsByVar, futureAbsByVar);
    }}
    updateThermoVariants();
    
    try {{
      // Row 0: CTI map (if available)
      const ctiVariant = "cti";
      if (combinedAbsByVar[ctiVariant]) {{
        if (ctiOnTop && ctiContainer && !ctiContainer.empty()) {{
          const card0 = ctiContainer.append("div").attr("class", "map-card");
          renderMap(card0, ctiVariant, combinedAbsByVar, tempScale, 40);
        }} else {{
          const row0 = container.append("div").attr("class", "maps-row");
          const card0 = row0.append("div").attr("class", "map-card");
          renderMap(card0, ctiVariant, combinedAbsByVar, tempScale, 40);
        }}
      }}

      // Row 1: TMYx, RCP 4.5 Year 2050, RCP 8.5 Year 2080
      // Only show RCP scenarios if they are in baselineVariants (to avoid repetition in bottom container)
      const row1 = container.append("div").attr("class", "maps-row maps-row-3");
      
      // TMYx (exact match "tmyx")
      const tmyxVariant = baselineVariants.find(v => v === "tmyx");
      if (tmyxVariant) {{
        const card1 = row1.append("div").attr("class", "map-card");
        renderMap(card1, tmyxVariant, combinedAbsByVar, tempScale, 40);
      }}
      
      // RCP 4.5 Year 2050 - handle baseline-prefixed keys (e.g., "tmyx__rcp45_2050")
      const rcp45_2050 = Object.keys(combinedAbsByVar || {{}}).find(k => k.endsWith("__rcp45_2050")) || "rcp45_2050";
      if (combinedAbsByVar[rcp45_2050]) {{
        const card2 = row1.append("div").attr("class", "map-card");
        renderMap(card2, rcp45_2050, combinedAbsByVar, tempScale, 40);
      }}
      
      // RCP 8.5 Year 2080 - handle baseline-prefixed keys (e.g., "tmyx__rcp85_2080")
      const rcp85_2080 = Object.keys(combinedAbsByVar || {{}}).find(k => k.endsWith("__rcp85_2080")) || "rcp85_2080";
      if (combinedAbsByVar[rcp85_2080]) {{
        const card3 = row1.append("div").attr("class", "map-card");
        renderMap(card3, rcp85_2080, combinedAbsByVar, tempScale, 40);
      }}
      
      // Row 2: TMYx 2004-2018, TMYx 2007-2021, TMYx 2009-2023
      // Filter out RCP scenarios from row 2
      if (!hideRow2) {{
        const row2 = container.append("div").attr("class", "maps-row");
        const otherBaselineVariants = baselineVariants.filter(v =>
          v !== "tmyx" &&
          v !== "cti" &&
          !v.startsWith("rcp") &&
          !v.includes("__rcp")
        );
        otherBaselineVariants.forEach(v => {{
          const card = row2.append("div").attr("class", "map-card");
          renderMap(card, v, combinedAbsByVar, tempScale, 40);
        }});
      }}

      renderThermometers(currentLocationId);
    }} catch (e) {{
      console.error("Error rendering maps:", e);
    }}
  }}

  // Initial render after async data loads
  // Ensure currentRegion is set before rendering
  if (!currentRegion && regionOptions && regionOptions.length > 0) {{
    currentRegion = regionOptions[0].code;
    if (!hideRegionSelector) {{
      regionSelect.property("value", currentRegion || "");
    }}
    safeSetStoredRegion(currentRegion);
  }}
  
  // Only render if we have a region selected
  if (currentRegion) {{
    renderAll();
  }} else {{
    console.warn("Cannot render maps: no region selected and no region options available");
  }}
}})();
</script>
"""



def f208__plotly_tmyx_heatmap(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variant: str,
    colorscale: str = "RdBu_r",
    zmin: float | None = 0,
    zmax: float | None = 40,
    height: int = 320,
    margin: dict | None = None,
    base_year: int = 2000,
) -> go.Figure | None:
    """
    Build a month x hour heatmap from raw hourly DBT for a specific location + variant.

    Adapted from the chart helpers in the referenced app (heatmap-style aggregation).
    """
    margin = margin or dict(l=42, r=8, t=32, b=32)
    rel_paths = idx_df[
        (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
    ]["rel_path"].dropna().unique().tolist()
    if not rel_paths:
        return None

    ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
    if ts.empty:
        return None

    ts["dayofyear"] = ts["datetime"].dt.dayofyear
    ts["hour"] = ts["datetime"].dt.hour
    heat = ts.groupby(["dayofyear", "hour"], as_index=False).agg(DBT=("DBT", "mean"))
    pivot = (
        heat.pivot(index="hour", columns="dayofyear", values="DBT")
        .reindex(index=list(range(24)), columns=list(range(1, 367)))
    )
    z = pivot.values
    x_dates = pd.to_datetime(f"{base_year}-01-01") + pd.to_timedelta(
        pivot.columns - 1, unit="D"
    )

    # Use nuanced colorscale if not explicitly overridden
    if colorscale == "RdBu_r":
        plotly_colorscale = _get_nuanced_colorscale()
    else:
        plotly_colorscale = colorscale

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_dates,
            y=[str(h) for h in pivot.index],
            colorscale=plotly_colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="°C", len=0.7, thickness=10),
        )
    )
    fig.update_layout(
        title=variant,
        height=height,
        margin=margin,
    )
    fig.update_xaxes(title="Date", tickformat="%b %d")
    # Bring Y-axis title closer to tick labels
    fig.update_yaxes(title="Hour", autorange="reversed", title_standoff=5, automargin=True)
    return fig


def f209__plotly_tmyx_daily_range_scatter(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variant: str,
    height: int = 220,
    margin: dict | None = None,
    y_range: tuple[float, float] = (0, 40),
    base_year: int = 2000,
) -> go.Figure | None:
    """
    Daily min/max/mean scatter (lines + range band) from hourly DBT.

    Inspired by the daily range chart helpers in the referenced app.
    """
    margin = margin or dict(l=42, r=10, t=26, b=30)
    rel_paths = idx_df[
        (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
    ]["rel_path"].dropna().unique().tolist()
    if not rel_paths:
        return None

    ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
    if ts.empty:
        return None
    ts = ts.set_index("datetime").sort_index()

    daily = ts.resample("D").agg(
        DBT_min=("DBT", "min"),
        DBT_max=("DBT", "max"),
        DBT_mean=("DBT", "mean"),
    ).reset_index()
    daily["dayofyear"] = daily["datetime"].dt.dayofyear
    daily["date_index"] = pd.to_datetime(f"{base_year}-01-01") + pd.to_timedelta(
        daily["dayofyear"] - 1, unit="D"
    )
    daily = daily.sort_values("date_index")
    # Drop any rows with NaN in critical columns
    daily = daily.dropna(subset=["DBT_min", "DBT_max", "DBT_mean", "date_index"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date_index"],
            y=daily["DBT_min"],
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date_index"],
            y=daily["DBT_max"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(112, 115, 155, 0.18)",
            name="Daily Range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date_index"],
            y=daily["DBT_mean"],
            mode="lines",
            line=dict(color="#1f77b4", width=2.5),
            name="Daily Mean",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"{variant} — Daily DBT",
        height=height,
        margin=margin,
        showlegend=True,
    )
    fig.update_xaxes(title="Date", tickformat="%b %d")
    # Bring Y-axis title closer to tick labels
    fig.update_yaxes(title="Temp (°C)", range=list(y_range), title_standoff=5, automargin=True)
    return fig


def f210__plotly_tmyx_stacked_column(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variant: str,
    height: int = 320,
    margin: dict | None = None,
    temp_min: float = 0,
    temp_max: float = 40,
    bin_step: float = 5,
    normalize: bool = True,
    colorscale: str = "RdBu_r",
) -> go.Figure | None:
    """
    Create a stacked column chart showing temperature distribution by month.
    
    Bins hourly DBT into temperature ranges and stacks them by month.
    """
    margin = margin or dict(l=42, r=8, t=32, b=32)
    rel_paths = idx_df[
        (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
    ]["rel_path"].dropna().unique().tolist()
    if not rel_paths:
        return None

    ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
    if ts.empty:
        return None
    
    ts["month"] = ts["datetime"].dt.month
    ts = ts.dropna(subset=["DBT", "month"])
    
    # Create temperature bins
    bins = np.arange(temp_min, temp_max + bin_step, bin_step)
    bin_labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins) - 1)]
    ts["binned"] = pd.cut(ts["DBT"], bins=bins, labels=bin_labels, right=False, include_lowest=True)
    
    # Group by month and bin, count hours
    binned_data = ts.groupby(["month", "binned"], observed=True).size().reset_index(name="count")
    
    if normalize:
        # Calculate percentage per month
        month_totals = binned_data.groupby("month")["count"].sum().reset_index(name="total")
        binned_data = binned_data.merge(month_totals, on="month")
        binned_data["percentage"] = (binned_data["count"] / binned_data["total"]) * 100
        value_field = "percentage"
        yaxis_title = "% Hours"
    else:
        value_field = "count"
        yaxis_title = "Total Hours"
    
    # Map month numbers to abbreviations and ensure proper order
    month_order = list(range(1, 13))
    month_abbrs = [calendar.month_abbr[m] for m in month_order]
    binned_data["month_abbr"] = binned_data["month"].apply(lambda x: calendar.month_abbr[int(x)])
    
    # Ensure all months are present (fill missing months with 0)
    all_months = pd.DataFrame({"month": month_order, "month_abbr": month_abbrs})
    binned_data = all_months.merge(binned_data, on=["month", "month_abbr"], how="left")
    binned_data[value_field] = binned_data[value_field].fillna(0)
    binned_data = binned_data.sort_values("month")
    
    # Get unique bins and sort them
    unique_bins = sorted(binned_data["binned"].dropna().unique(), key=lambda x: float(str(x).split("-")[0]))
    
    # Use nuanced colorset
    nuanced_colors = _get_nuanced_colorset()
    # Map bins to colors based on temperature (cold to hot)
    bin_mins = [float(str(b).split("-")[0]) for b in unique_bins]
    if bin_mins:
        norm = np.array([(bm - temp_min) / (temp_max - temp_min) for bm in bin_mins])
        norm = np.clip(norm, 0, 1)
        colors = [nuanced_colors[int(n * (len(nuanced_colors) - 1))] for n in norm]
    else:
        colors = [nuanced_colors[0]] * len(unique_bins)
    
    fig = go.Figure()
    
    # Add a trace for each bin
    for i, bin_val in enumerate(unique_bins):
        bin_data = binned_data[binned_data["binned"] == bin_val].copy()
        if bin_data.empty:
            # Create empty data for this bin with all months
            bin_data = all_months.copy()
            bin_data[value_field] = 0
            bin_data["binned"] = bin_val
        
        # Ensure months are in correct order
        bin_data = bin_data.sort_values("month")
        
        fig.add_trace(
            go.Bar(
                x=bin_data["month_abbr"],
                y=bin_data[value_field],
                name=str(bin_val),
                marker_color=colors[i],
            )
        )
    
    # Adjust margin to accommodate right-side legend
    adjusted_margin = margin.copy() if margin else dict(l=42, r=8, t=32, b=32)
    adjusted_margin["r"] = max(adjusted_margin.get("r", 8), 120)  # Ensure space for legend
    
    fig.update_layout(
        barmode="stack",
        title=f"{variant} — Temperature Distribution",
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        height=height,
        margin=adjusted_margin,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    # Set category order to ensure months appear in chronological order
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=month_abbrs)
    # Bring Y-axis title closer to tick labels
    fig.update_yaxes(title_standoff=5, automargin=True)
    
    return fig


# -----------------------------
# Combined subplot functions (all variants in one figure)
# -----------------------------
def f211__plotly_tmyx_heatmap_subplots(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variants: list[str],
    colorscale: str = "RdBu_r",
    zmin: float | None = 0,
    zmax: float | None = 40,
    subplot_height: int = 260,
    vertical_spacing: float = 0.08,
    show_subplot_titles: bool = True,
    subplot_title_font_size: int = 11,
    subplot_title_yshift: int = -12,
    subplot_title_bold: bool = True,
    margin: dict | None = None,
    base_year: int = 2000,
) -> go.Figure | None:
    """
    Create a vertical subplot figure with all heatmaps stacked.
    """
    margin = margin or dict(l=42, r=8, t=32, b=32)
    n_variants = len(variants)
    if n_variants == 0:
        return None
    
    # Use nuanced colorscale if not explicitly overridden
    if colorscale == "RdBu_r":
        plotly_colorscale = _get_nuanced_colorscale()
    else:
        plotly_colorscale = colorscale
    
    fig = make_subplots(
        rows=n_variants,
        cols=1,
        subplot_titles=(variants if show_subplot_titles else None),
        vertical_spacing=float(vertical_spacing),
        shared_xaxes=True,
        shared_yaxes=False,
    )
    # Subplot titles: keep them, but pull them closer to plots
    if show_subplot_titles and getattr(fig.layout, "annotations", None):
        for i in range(min(len(variants), len(fig.layout.annotations))):
            ann = fig.layout.annotations[i]
            ann.font.size = int(subplot_title_font_size)
            ann.yshift = int(subplot_title_yshift)
            if bool(subplot_title_bold):
                t = str(getattr(ann, "text", "") or "")
                if t and not t.lstrip().startswith("<b>"):
                    ann.text = f"<b>{t}</b>"
    
    for i, variant in enumerate(variants, 1):
        rel_paths = idx_df[
            (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
        ]["rel_path"].dropna().unique().tolist()
        if not rel_paths:
            continue
        
        ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
        if ts.empty:
            continue
        
        ts["dayofyear"] = ts["datetime"].dt.dayofyear
        ts["hour"] = ts["datetime"].dt.hour
        heat = ts.groupby(["dayofyear", "hour"], as_index=False).agg(DBT=("DBT", "mean"))
        day_min = int(heat["dayofyear"].min())
        day_max = int(heat["dayofyear"].max())
        unique_days = heat["dayofyear"].nunique()
        full_year = day_min == 1 and day_max >= 365 and unique_days >= 360
        day_cols = list(range(1, 367)) if full_year else list(range(day_min, day_max + 1))
        pivot = (
            heat.pivot(index="hour", columns="dayofyear", values="DBT")
            .reindex(index=list(range(24)), columns=day_cols)
        )
        z = pivot.values
        x_dates = pd.to_datetime(f"{base_year}-01-01") + pd.to_timedelta(
            pivot.columns - 1, unit="D"
        )
        
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_dates,
                y=[str(h) for h in pivot.index],
                colorscale=plotly_colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title="°C",
                    len=0.25,
                    thickness=14,
                    x=0.0,
                    y=1.02,
                    yanchor="bottom",
                    orientation="h",
                    xanchor="left",
                    tickangle=0,
                ) if i == 1 else None,
                showscale=(i == 1),
                hovertemplate="Date: %{x|%b %d}<br>Hour: %{y}<br>T (C): %{z:.1f}<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title="Hour", autorange="reversed", title_standoff=5, automargin=True, row=i, col=1)
    
    fig.update_xaxes(title="Date", tickformat="%b %d", row=n_variants, col=1)
    # Adjust top margin to accommodate colorbar
    adjusted_margin = margin.copy() if margin else dict(l=42, r=8, t=32, b=32)
    # In the Streamlit TMYx tab we render legends outside the figures, so keep top margin tight.
    adjusted_margin["t"] = max(adjusted_margin.get("t", 32), 10)
    fig.update_layout(
        height=subplot_height * n_variants,
        margin=adjusted_margin,
        hovermode="x unified",
    )
    return fig


def f212__plotly_tmyx_scatter_subplots(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variants: list[str],
    subplot_height: int = 260,
    vertical_spacing: float = 0.08,
    show_subplot_titles: bool = True,
    subplot_title_font_size: int = 11,
    subplot_title_yshift: int = -12,
    subplot_title_bold: bool = True,
    margin: dict | None = None,
    y_range: tuple[float, float] = (0, 40),
    base_year: int = 2000,
) -> go.Figure | None:
    """
    Create a vertical subplot figure with all daily scatter charts stacked.
    """
    margin = margin or dict(l=42, r=10, t=26, b=30)
    n_variants = len(variants)
    if n_variants == 0:
        return None
    
    fig = make_subplots(
        rows=n_variants,
        cols=1,
        subplot_titles=(variants if show_subplot_titles else None),
        vertical_spacing=float(vertical_spacing),
        shared_xaxes=True,
        shared_yaxes=True,
    )
    # Subplot titles: keep them, but pull them closer to plots
    if show_subplot_titles and getattr(fig.layout, "annotations", None):
        for i in range(min(len(variants), len(fig.layout.annotations))):
            ann = fig.layout.annotations[i]
            ann.font.size = int(subplot_title_font_size)
            ann.yshift = int(subplot_title_yshift)
            if bool(subplot_title_bold):
                t = str(getattr(ann, "text", "") or "")
                if t and not t.lstrip().startswith("<b>"):
                    ann.text = f"<b>{t}</b>"
    
    for i, variant in enumerate(variants, 1):
        rel_paths = idx_df[
            (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
        ]["rel_path"].dropna().unique().tolist()
        if not rel_paths:
            continue
        
        ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
        if ts.empty:
            continue
        ts = ts.set_index("datetime").sort_index()
        
        daily = ts.resample("D").agg(
            DBT_min=("DBT", "min"),
            DBT_max=("DBT", "max"),
            DBT_mean=("DBT", "mean"),
        ).reset_index()
        daily["dayofyear"] = daily["datetime"].dt.dayofyear
        daily["date_index"] = pd.to_datetime(f"{base_year}-01-01") + pd.to_timedelta(
            daily["dayofyear"] - 1, unit="D"
        )
        daily = daily.sort_values("date_index")
        daily = daily.dropna(subset=["DBT_min", "DBT_max", "DBT_mean", "date_index"])
        
        # Min trace (invisible, for fill)
        fig.add_trace(
            go.Scatter(
                x=daily["date_index"],
                y=daily["DBT_min"],
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=i,
            col=1,
        )
        # Max trace (fill)
        fig.add_trace(
            go.Scatter(
                x=daily["date_index"],
                y=daily["DBT_max"],
                fill="tonexty",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(112, 115, 155, 0.18)",
                name="Daily Range",
                showlegend=(i == 1),
                hovertemplate="Date: %{x|%b %d}<br>Max: %{y:.1f}°C<extra></extra>",
            ),
            row=i,
            col=1,
        )
        # Mean trace
        fig.add_trace(
            go.Scatter(
                x=daily["date_index"],
                y=daily["DBT_mean"],
                mode="lines",
                line=dict(color="#1f77b4", width=2.5),
                name="Daily Mean",
                showlegend=(i == 1),
                hovertemplate="Date: %{x|%b %d}<br>Mean: %{y:.1f}°C<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title="Temp (°C)", range=list(y_range), title_standoff=5, automargin=True, row=i, col=1)
    
    fig.update_xaxes(title="Date", tickformat="%b %d", row=n_variants, col=1)
    # Adjust top margin to accommodate legend
    adjusted_margin = margin.copy() if margin else dict(l=42, r=10, t=26, b=30)
    # In the Streamlit TMYx tab we render legends outside the figures, so keep top margin tight.
    adjusted_margin["t"] = max(adjusted_margin.get("t", 26), 10)
    fig.update_layout(
        height=subplot_height * n_variants,
        margin=adjusted_margin,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    return fig


def f213__plotly_tmyx_stacked_subplots(
    hourly_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    *,
    location_id: str,
    variants: list[str],
    subplot_height: int = 260,
    vertical_spacing: float = 0.08,
    show_subplot_titles: bool = True,
    subplot_title_font_size: int = 11,
    subplot_title_yshift: int = -12,
    subplot_title_bold: bool = True,
    margin: dict | None = None,
    temp_min: float = 0,
    temp_max: float = 40,
    bin_step: float = 5,
    normalize: bool = True,
    colorscale: str = "RdBu_r",
) -> go.Figure | None:
    """
    Create a vertical subplot figure with all stacked column charts stacked.
    """
    margin = margin or dict(l=42, r=8, t=32, b=32)
    n_variants = len(variants)
    if n_variants == 0:
        return None
    
    # Legend is horizontal (top), so no need for extra right margin.
    adjusted_margin = margin.copy() if margin else dict(l=42, r=8, t=32, b=32)
    adjusted_margin["r"] = max(adjusted_margin.get("r", 8), 20)
    
    # Prepare month order
    month_order = list(range(1, 13))
    month_abbrs = [calendar.month_abbr[m] for m in month_order]
    
    # Get unique bins (same for all variants)
    bins = np.arange(temp_min, temp_max + bin_step, bin_step)
    bin_labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins) - 1)]
    unique_bins = sorted(bin_labels, key=lambda x: float(x.split("-")[0]))
    
    # Use nuanced colorset
    nuanced_colors = _get_nuanced_colorset()
    bin_mins = [float(str(b).split("-")[0]) for b in unique_bins]
    if bin_mins:
        norm = np.array([(bm - temp_min) / (temp_max - temp_min) for bm in bin_mins])
        norm = np.clip(norm, 0, 1)
        colors = [nuanced_colors[int(n * (len(nuanced_colors) - 1))] for n in norm]
    else:
        colors = [nuanced_colors[0]] * len(unique_bins)
    
    fig = make_subplots(
        rows=n_variants,
        cols=1,
        subplot_titles=(variants if show_subplot_titles else None),
        vertical_spacing=float(vertical_spacing),
        shared_xaxes=True,
        shared_yaxes=True,
    )
    # Subplot titles: keep them, but pull them closer to plots
    if show_subplot_titles and getattr(fig.layout, "annotations", None):
        for i in range(min(len(variants), len(fig.layout.annotations))):
            ann = fig.layout.annotations[i]
            ann.font.size = int(subplot_title_font_size)
            ann.yshift = int(subplot_title_yshift)
            if bool(subplot_title_bold):
                t = str(getattr(ann, "text", "") or "")
                if t and not t.lstrip().startswith("<b>"):
                    ann.text = f"<b>{t}</b>"
    
    for i, variant in enumerate(variants, 1):
        rel_paths = idx_df[
            (idx_df["location_id"] == location_id) & (idx_df["variant"] == variant)
        ]["rel_path"].dropna().unique().tolist()
        if not rel_paths:
            continue
        
        ts = hourly_df[hourly_df["rel_path"].isin(rel_paths)][["datetime", "DBT"]].copy()
        if ts.empty:
            continue
        
        ts["month"] = ts["datetime"].dt.month
        ts = ts.dropna(subset=["DBT", "month"])
        ts["binned"] = pd.cut(ts["DBT"], bins=bins, labels=bin_labels, right=False, include_lowest=True)
        
        binned_data = ts.groupby(["month", "binned"], observed=True).size().reset_index(name="count")
        
        if normalize:
            month_totals = binned_data.groupby("month")["count"].sum().reset_index(name="total")
            binned_data = binned_data.merge(month_totals, on="month")
            binned_data["percentage"] = (binned_data["count"] / binned_data["total"]) * 100
            value_field = "percentage"
            yaxis_title = "% Hours"
        else:
            value_field = "count"
            yaxis_title = "Total Hours"
        
        binned_data["month_abbr"] = binned_data["month"].apply(lambda x: calendar.month_abbr[int(x)])
        all_months = pd.DataFrame({"month": month_order, "month_abbr": month_abbrs})
        binned_data = all_months.merge(binned_data, on=["month", "month_abbr"], how="left")
        binned_data[value_field] = binned_data[value_field].fillna(0)
        binned_data = binned_data.sort_values("month")
        
        for j, bin_val in enumerate(unique_bins):
            bin_data = binned_data[binned_data["binned"] == bin_val].copy()
            if bin_data.empty:
                bin_data = all_months.copy()
                bin_data[value_field] = 0
                bin_data["binned"] = bin_val
            bin_data = bin_data.sort_values("month")
            
            fig.add_trace(
                go.Bar(
                    x=bin_data["month_abbr"],
                    y=bin_data[value_field],
                    name=str(bin_val),
                    marker_color=colors[j],
                    showlegend=(i == 1),  # Only show legend in first subplot
                    hovertemplate="Month: %{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>" if normalize else "Month: %{x}<br>%{fullData.name}: %{y:.0f}<extra></extra>",
                ),
                row=i,
                col=1,
            )
        fig.update_yaxes(title=yaxis_title, title_standoff=5, automargin=True, row=i, col=1)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=month_abbrs, row=i, col=1)
    
    fig.update_xaxes(title="Month", row=n_variants, col=1)
    # Adjust top margin to accommodate legend
    # In the Streamlit TMYx tab we render legends outside the figures, so keep top margin tight.
    adjusted_margin["t"] = max(adjusted_margin.get("t", 32), 10)
    fig.update_layout(
        barmode="stack",
        height=subplot_height * n_variants,
        margin=adjusted_margin,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    return fig



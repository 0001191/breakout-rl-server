const statusUrl = "../artifacts/live/status.json";
const historyUrl = "../artifacts/live/history.json";
const previewUrl = "../artifacts/live/preview.png";
const previewStreamUrl = "../api/preview-stream";

const phaseBadge = document.getElementById("phaseBadge");
const timesteps = document.getElementById("timesteps");
const progress = document.getElementById("progress");
const meanReward = document.getElementById("meanReward");
const epsilon = document.getElementById("epsilon");
const loss = document.getElementById("loss");
const fps = document.getElementById("fps");
const previewImage = document.getElementById("previewImage");
const recentEpisodes = document.getElementById("recentEpisodes");
const rewardChart = document.getElementById("rewardChart");
const refreshButton = document.getElementById("refreshButton");
const logPanel = document.getElementById("logPanel");

let streamStarted = false;

function safeNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

async function loadJson(url) {
  const response = await fetch(`${url}?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`加载失败 ${response.status}`);
  return response.json();
}

function drawChart(points) {
  rewardChart.innerHTML = "";
  if (!points.length) return;

  const width = 640;
  const height = 260;
  const padding = 28;
  const rewards = points.map((point) => point.mean_reward).filter((value) => value !== null && value !== undefined);
  const epsilons = points.map((point) => point.exploration_rate).filter((value) => value !== null && value !== undefined);
  if (!rewards.length || !epsilons.length) return;

  const minReward = Math.min(...rewards);
  const maxReward = Math.max(...rewards);
  const rewardSpan = Math.max(maxReward - minReward, 1);
  const rewardPath = [];
  const epsilonPath = [];

  points.forEach((point, index) => {
    const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
    const rewardY = height - padding - (((point.mean_reward ?? minReward) - minReward) / rewardSpan) * (height - padding * 2);
    const epsilonY = height - padding - (point.exploration_rate ?? 0) * (height - padding * 2);
    rewardPath.push(`${index === 0 ? "M" : "L"} ${x} ${rewardY}`);
    epsilonPath.push(`${index === 0 ? "M" : "L"} ${x} ${epsilonY}`);
  });

  [
    { y: padding, label: `奖励高 ${safeNumber(maxReward)}` },
    { y: height - padding, label: `奖励低 ${safeNumber(minReward)}` },
  ].forEach((line) => {
    const guide = document.createElementNS("http://www.w3.org/2000/svg", "line");
    guide.setAttribute("x1", String(padding));
    guide.setAttribute("x2", String(width - padding));
    guide.setAttribute("y1", String(line.y));
    guide.setAttribute("y2", String(line.y));
    guide.setAttribute("stroke", "rgba(255,255,255,0.08)");
    guide.setAttribute("stroke-dasharray", "4 6");
    rewardChart.appendChild(guide);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(padding));
    text.setAttribute("y", String(line.y - 6));
    text.setAttribute("fill", "#8fa7c7");
    text.setAttribute("font-size", "12");
    text.textContent = line.label;
    rewardChart.appendChild(text);
  });

  const rewardLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
  rewardLine.setAttribute("d", rewardPath.join(" "));
  rewardLine.setAttribute("fill", "none");
  rewardLine.setAttribute("stroke", "#5fd1ff");
  rewardLine.setAttribute("stroke-width", "3");
  rewardChart.appendChild(rewardLine);

  const epsilonLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
  epsilonLine.setAttribute("d", epsilonPath.join(" "));
  epsilonLine.setAttribute("fill", "none");
  epsilonLine.setAttribute("stroke", "#f7b955");
  epsilonLine.setAttribute("stroke-width", "2.5");
  epsilonLine.setAttribute("stroke-dasharray", "8 6");
  rewardChart.appendChild(epsilonLine);
}

function renderEpisodes(episodes) {
  recentEpisodes.innerHTML = "";
  if (!episodes.length) {
    recentEpisodes.innerHTML = "<p>还没有足够的 episode 数据。</p>";
    return;
  }
  episodes.slice().reverse().forEach((episode, index) => {
    const item = document.createElement("div");
    item.className = "episode-item";
    item.innerHTML = `<span>最近局 ${index + 1}</span><span>奖励 ${safeNumber(episode.reward, 1)} / 步数 ${safeNumber(episode.length, 0)}</span>`;
    recentEpisodes.appendChild(item);
  });
}

function ensurePreviewStream(status) {
  if (!streamStarted && status.stream_path) {
    previewImage.src = `${previewStreamUrl}?t=${Date.now()}`;
    streamStarted = true;
    return;
  }

  if (!streamStarted && status.last_preview_path) {
    previewImage.src = `${previewUrl}?v=${status.preview_updated_at || Date.now()}`;
  }
}

async function refresh() {
  try {
    const [status, history] = await Promise.all([loadJson(statusUrl), loadJson(historyUrl)]);
    phaseBadge.textContent = status.phase || "unknown";
    timesteps.textContent = `${status.timesteps ?? 0}`;
    progress.textContent = `${safeNumber(status.progress_percent, 1)}%`;
    meanReward.textContent = safeNumber(status.mean_reward_100, 2);
    epsilon.textContent = safeNumber(status.exploration_rate, 4);
    loss.textContent = safeNumber(status.loss, 4);
    fps.textContent = `${safeNumber(status.fps, 1)} step/s`;
    ensurePreviewStream(status);
    drawChart(history);
    renderEpisodes(status.recent_episodes || []);
  } catch (error) {
    phaseBadge.textContent = "未开始";
    recentEpisodes.innerHTML = `<p>还没读到训练状态：${error.message}</p>`;
  }

  try {
    const logPayload = await loadJson("../api/logs");
    logPanel.textContent = (logPayload.lines || []).join("\n") || "日志暂时为空。";
  } catch (error) {
    logPanel.textContent = `日志读取失败：${error.message}`;
  }
}

previewImage.addEventListener("error", () => {
  streamStarted = false;
});

refreshButton.addEventListener("click", refresh);
refresh();
setInterval(refresh, 2500);

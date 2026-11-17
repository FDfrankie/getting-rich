// Truthful trading snapshot renderer (single in-flight-guarded scheduler, zero innerHTML XSS, binds both canonical & legacy DOM IDs).
async function fetchJsonStrict(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const text = await response.text();
  let payload = null;
  if (text.trim()) {
    try { payload = JSON.parse(text); }
    catch (_) {
      throw new Error(`NON_JSON_RESPONSE http=${response.status} endpoint=${url}`);
    }
  }
  if (!payload) throw new Error(`EMPTY_RESPONSE http=${response.status} endpoint=${url}`);
  if (!response.ok) {
    throw new Error(payload.safe_error || payload.error || `HTTP_${response.status}`);
  }
  return payload;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? 'N/A';
}

function renderTradingSnapshot(s) {
  const account = s.account || {};
  const market = s.market || {};
  const positions = Array.isArray(s.positions) ? s.positions : [];
  const curr = account.currency || market.settlement_currency || 'USDT';

  const eqText = account.equity == null ? 'N/A' : `${account.equity} ${curr}`;
  const walText = account.wallet_balance == null ? 'N/A' : `${account.wallet_balance} ${curr}`;
  const pnlText = account.unrealized_pnl == null ? 'N/A' : `${account.unrealized_pnl} ${curr}`;
  const posText = positions.length > 0 ? `${positions.length} 筆合約持倉` : 'FLAT (0)';
  const priceText = market.mark_price == null ? 'N/A' : `${market.mark_price} ${curr}`;
  const modeText = `${s.context?.requested_mode || 'UNKNOWN'} -> ${s.context?.effective_mode || 'UNKNOWN'}`;

  // Update canonical IDs
  setText('equityValue', eqText);
  setText('walletBalanceValue', walText);
  setText('unrealizedPnlValue', pnlText);
  setText('positionCountValue', String(positions.length));
  setText('markPriceValue', priceText);
  setText('tradingModeValue', modeText);
  setText('snapshotStatusValue', s.status || 'UNAVAILABLE');
  setText('snapshotAgeValue', market.stale_seconds == null ? 'N/A' : `${Number(market.stale_seconds).toFixed(1)}s`);

  // Also update existing index.html card IDs so legacy cards never show 15,000 / +112.50
  setText('cryptoEquity', eqText);
  setText('cryptoPrice', priceText);
  setText('cryptoPosition', posText);
  setText('cryptoPnL', pnlText);
  setText('cryptoModeBadge', s.context?.degraded ? `DEGRADED (${modeText})` : modeText);

  const safeForNewPositions = s.status === 'HEALTHY' && s.risk?.trading_enabled === true;
  document.querySelectorAll('[data-opens-position="true"]').forEach((button) => {
    button.disabled = !safeForNewPositions;
    button.title = safeForNewPositions ? '' : `Trading blocked: ${(s.errors || s.risk?.reasons || []).join(', ')}`;
  });
}

function updateSubsystemStatusBar(s, brainConn, sysHealth) {
  const headerEl = document.getElementById('headerStatus');
  const status = s?.status || 'UNAVAILABLE';
  const degraded = Boolean(s?.context?.degraded);
  if (headerEl) {
    if (status === 'HEALTHY' && !degraded) {
      headerEl.textContent = '系統健康 (HEALTHY)';
    } else if (degraded) {
      headerEl.textContent = `降級本地 (${s?.context?.degrade_reason || 'DEGRADED'})`;
    } else {
      headerEl.textContent = `警戒 (${status})`;
    }
  }
  let bar = document.getElementById('subsystemStatusBar');
  if (!bar) {
    const header = document.querySelector('header');
    if (header && header.parentNode) {
      bar = document.createElement('div');
      bar.id = 'subsystemStatusBar';
      bar.className = 'max-w-7xl mx-auto w-full px-4 pt-2 flex flex-wrap gap-2 text-xs';
      header.parentNode.insertBefore(bar, header.nextSibling);
    }
  }
  if (bar) {
    bar.replaceChildren();
    const pills = [
      ['Bot', sysHealth?.watchdog?.trading_bot || 'UNKNOWN', sysHealth?.watchdog?.trading_bot === 'OK'],
      ['Mode', `${s?.context?.requested_mode || '?'} -> ${s?.context?.effective_mode || '?'}`, !degraded],
      ['Market', s?.market?.data_source || 'N/A', status === 'HEALTHY'],
      ['Account', s?.connection?.authenticated ? 'AUTH' : (s?.connection?.error_code || 'LOCAL'), Boolean(s?.connection?.authenticated)],
      ['Reconcile', s?.reconciliation?.status || 'UNKNOWN', s?.reconciliation?.status === 'OK'],
      ['BRAIN Auth', brainConn?.authenticated ? 'OK' : (brainConn?.safe_error_code || 'OFFLINE'), Boolean(brainConn?.authenticated)],
      ['BRAIN Queue', String(brainConn?.queue_depth ?? 0), Boolean(brainConn?.brain_worker_alive)],
    ];
    pills.forEach(([k, v, ok]) => {
      const span = document.createElement('span');
      span.className = ok
        ? 'px-2 py-0.5 rounded border bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
        : 'px-2 py-0.5 rounded border bg-amber-500/10 border-amber-500/30 text-amber-300';
      const strong = document.createElement('strong');
      strong.textContent = `${k}: `;
      span.appendChild(strong);
      span.appendChild(document.createTextNode(String(v)));
      bar.appendChild(span);
    });
  }
}

let truthRefreshInFlight = false;
async function refreshAllTruthStatus() {
  if (truthRefreshInFlight) return;
  truthRefreshInFlight = true;
  let snap = null;
  let brainConn = null;
  let sysHealth = null;
  try {
    snap = await fetchJsonStrict('/api/trading/snapshot');
    renderTradingSnapshot(snap);
  } catch (error) {
    ['equityValue', 'walletBalanceValue', 'unrealizedPnlValue', 'positionCountValue', 'markPriceValue', 'cryptoEquity', 'cryptoPrice', 'cryptoPosition', 'cryptoPnL'].forEach((id) => setText(id, 'N/A'));
    setText('snapshotStatusValue', 'UNAVAILABLE');
    document.querySelectorAll('[data-opens-position="true"]').forEach((button) => { button.disabled = true; });
    console.error('[TRADING_SNAPSHOT]', error);
  }
  try { brainConn = await fetchJsonStrict('/api/brain/connectivity'); } catch (_) {}
  try { sysHealth = await fetchJsonStrict('/api/status'); } catch (_) {}
  updateSubsystemStatusBar(snap, brainConn, sysHealth);
  truthRefreshInFlight = false;
}

setInterval(refreshAllTruthStatus, 3000);
refreshAllTruthStatus();

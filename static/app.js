/**
 * LLM Manager – shared JS utilities
 * Loaded via <script src="/static/app.js"> in base.html (optional; pages may inline their own JS).
 */

'use strict';

/**
 * Escape a string for safe insertion as HTML text content.
 * @param {string} s
 * @returns {string}
 */
function escapeHtml(s) {
  var d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

/**
 * Show a flash message at the top of .container.
 * @param {string} message
 * @param {'success'|'error'} type
 * @param {number} [ttlMs=4000]  Auto-dismiss after this many ms (0 = never)
 */
function showFlash(message, type, ttlMs) {
  var container = document.querySelector('.container');
  if (!container) return;
  var el = document.createElement('div');
  el.className = 'flash ' + (type || 'success');
  el.textContent = message;
  container.insertBefore(el, container.firstChild);
  var ms = ttlMs === undefined ? 4000 : ttlMs;
  if (ms > 0) {
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, ms);
  }
}

/**
 * Simple fetch wrapper that returns parsed JSON or throws with a readable message.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
function apiFetch(url, options) {
  return fetch(url, options).then(function (r) {
    if (!r.ok) {
      return r.text().then(function (t) {
        var msg;
        try { msg = JSON.parse(t).detail || t; } catch (_) { msg = t; }
        throw new Error('HTTP ' + r.status + ': ' + msg);
      });
    }
    return r.json();
  });
}

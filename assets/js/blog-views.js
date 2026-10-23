(function () {
  'use strict';

  var loader = document.currentScript;
  if (!loader) return;

  var endpoint = loader.dataset.counterEndpoint || '';
  var badges = Array.prototype.slice.call(document.querySelectorAll('[data-blog-views]'));
  if (!badges.length) return;

  var counterOrigin;
  try {
    // Development servers and other preview origins must not record visits.
    if (new URL(loader.dataset.siteUrl).origin !== window.location.origin) return;
    var parsedEndpoint = new URL(endpoint);
    if (parsedEndpoint.protocol !== 'https:' || !/^\/count\/?$/.test(parsedEndpoint.pathname) ||
        parsedEndpoint.username || parsedEndpoint.password || parsedEndpoint.search || parsedEndpoint.hash) return;
    counterOrigin = parsedEndpoint.origin;
  } catch (error) {
    return;
  }

  // The list only reads totals; opening an article records a pageview.
  if (document.querySelector('.blog-article')) {
    var tracker = document.createElement('script');
    tracker.async = true;
    tracker.src = 'https://gc.zgo.at/count.js';
    tracker.dataset.goatcounter = endpoint;
    document.head.appendChild(tracker);
  }

  badges.forEach(function (badge) {
    var path = badge.dataset.blogPath;
    var count = badge.querySelector('[data-blog-views-count]');
    if (!path || path.charAt(0) !== '/' || !count) return;

    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, 8000);
    fetch(counterOrigin + '/counter/' + encodeURIComponent(path) + '.json', { mode: 'cors', credentials: 'omit', signal: controller.signal })
      .then(function (response) {
        if (!response.ok) throw new Error('Counter request failed');
        return response.json();
      })
      .then(function (data) {
        if (!data || typeof data.count !== 'string' || !/^\d+(?:[,\s]\d{3})*$/.test(data.count)) {
          throw new Error('Invalid counter response');
        }
        count.textContent = data.count;
        badge.hidden = false;
      })
      .catch(function () {
        badge.hidden = true;
      })
      .finally(function () { window.clearTimeout(timeout); });
  });
})();

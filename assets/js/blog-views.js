(function () {
  'use strict';

  var loader = document.currentScript;
  if (!loader) return;

  var endpoint = loader.dataset.counterEndpoint || '';
  if (!document.querySelector('.blog-article')) return;
  try {
    // Development servers and other preview origins must not record visits.
    if (new URL(loader.dataset.siteUrl).origin !== window.location.origin) return;
    var parsedEndpoint = new URL(endpoint);
    if (parsedEndpoint.protocol !== 'https:' || !/^\/count\/?$/.test(parsedEndpoint.pathname) ||
        parsedEndpoint.username || parsedEndpoint.password || parsedEndpoint.search || parsedEndpoint.hash) return;
  } catch (error) {
    return;
  }

  // Keep private visit analytics without requesting or displaying public totals.
  var tracker = document.createElement('script');
  tracker.async = true;
  tracker.src = 'https://gc.zgo.at/count.js';
  tracker.dataset.goatcounter = endpoint;
  document.head.appendChild(tracker);
})();

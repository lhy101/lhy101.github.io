(function () {
  'use strict';
  var map = document.querySelector('[data-visitor-map]');
  if (!map || !map.dataset.siteId) return;
  // Local previews must not count as public visits.
  if (window.location.hostname !== map.dataset.siteHost) {
    map.hidden = true;
    return;
  }
  var canvas = map.querySelector('[data-visitor-map-canvas]');
  var script = document.createElement('script');
  script.id = 'clustrmaps';
  script.async = true;
  script.src = 'https://cdn.clustrmaps.com/map_v2.js?d=' + encodeURIComponent(map.dataset.siteId) + '&cl=426951&w=a&t=tt&co=ffffff&cmo=426951&cmn=a6b99c&ct=657169';
  script.onerror = function () { map.hidden = true; };
  canvas.appendChild(script);
})();

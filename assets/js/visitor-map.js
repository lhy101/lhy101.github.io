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
  var mapQuery = '?d=' + encodeURIComponent(map.dataset.siteId) + '&cl=ffffff';
  // The official image embed scales with the column and links to full statistics.
  var link = document.createElement('a');
  link.href = map.dataset.statsUrl || 'https://mapmyvisitors.com/';
  link.title = 'Visitor statistics';
  var image = document.createElement('img');
  image.alt = 'World map of visitors to this website';
  image.decoding = 'async';
  image.onload = function () { canvas.replaceChildren(link); };
  image.onerror = function () { map.hidden = true; };
  link.appendChild(image);
  image.src = 'https://mapmyvisitors.com/map.png' + mapQuery;
})();

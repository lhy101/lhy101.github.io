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
  var status = map.querySelector('[data-visitor-map-status]');
  var retry = map.querySelector('[data-visitor-map-retry]');
  var imageUrl = 'https://mapmyvisitors.com/map.png?d=' + encodeURIComponent(map.dataset.siteId) + '&cl=ffffff';
  var attempts = 0;
  var activeImage;
  var timer;

  function loadMap() {
    attempts += 1;
    var image = document.createElement('img');
    activeImage = image;
    status.textContent = 'Loading visitor map…';
    status.hidden = false;
    retry.hidden = true;
    image.alt = 'World map of visitors to this website';
    image.decoding = 'async';

    function failed() {
      if (activeImage !== image) return;
      activeImage = null;
      window.clearTimeout(timer);
      image.onload = image.onerror = null;
      image.removeAttribute('src');
      if (attempts < 2) {
        timer = window.setTimeout(loadMap, 1000);
      } else {
        status.textContent = 'Visitor map temporarily unavailable.';
        retry.hidden = false;
      }
    }

    image.onload = function () {
      if (activeImage !== image) return;
      activeImage = null;
      window.clearTimeout(timer);
      var link = document.createElement('a');
      link.href = map.dataset.statsUrl;
      link.title = 'Visitor statistics';
      link.appendChild(image);
      canvas.replaceChildren(link);
      canvas.hidden = false;
      status.hidden = true;
      retry.hidden = true;
    };
    image.onerror = failed;
    // A stalled third-party request must not leave a permanent loading state.
    timer = window.setTimeout(failed, 8000);
    image.src = imageUrl + (attempts > 1 ? '&retry=' + Date.now() : '');
  }

  retry.addEventListener('click', loadMap);
  loadMap();
})();

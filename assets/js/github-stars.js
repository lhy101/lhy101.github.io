(function () {
  'use strict';

  document.querySelectorAll('[data-github-repo]').forEach(function (badge) {
    var repo = badge.dataset.githubRepo;
    var cacheKey = 'github-stars:' + repo;
    var countLabel = badge.querySelector('[data-star-count]');
    if (!countLabel) return;

    function display(count) {
      countLabel.textContent = count.toLocaleString();
      badge.setAttribute('aria-label', repo + ' on GitHub, ' + count.toLocaleString() + ' stars');
    }

    try {
      var cached = JSON.parse(localStorage.getItem(cacheKey));
      if (cached && Number.isInteger(cached.count) && cached.count >= 0) {
        display(cached.count);
      }
    } catch (error) {}

    fetch('https://api.github.com/repos/' + repo, {
      headers: { Accept: 'application/vnd.github+json' }
    })
      .then(function (response) {
        if (!response.ok) throw new Error('GitHub request failed');
        return response.json();
      })
      .then(function (data) {
        if (!Number.isInteger(data.stargazers_count) || data.stargazers_count < 0) {
          throw new Error('Invalid GitHub star count');
        }
        display(data.stargazers_count);
        try {
          localStorage.setItem(cacheKey, JSON.stringify({ count: data.stargazers_count }));
        } catch (error) {}
      })
      .catch(function () {
        badge.title = 'Last available count. Open GitHub for the latest stars.';
      });
  });
})();

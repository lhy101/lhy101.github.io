---
layout: portfolio
title: "Publications"
excerpt: "The complete publication list of Haoyang Li, a Ph.D. student at Peking University (PKU), on deep learning training and reinforcement learning post-training systems."
permalink: /publications/
author_profile: false
---

<div class="publications-intro">
  <p class="eyebrow">PUBLICATIONS · 2024—2026</p>
  <h2>Systems for AI</h2>
  <p>My work explores the infrastructure behind deep learning (DL) training and reinforcement learning (RL) post-training: how to make large-scale workloads more efficient, balanced, and elastic.</p>
</div>

<div class="publication-list">
  {% for paper in site.data.publications %}
  <article class="publication-card" id="{{ paper.id }}">
    <figure class="publication-card__visual">
      <a class="publication-card__figure-link" href="{{ paper.figure_pdf | prepend: site.baseurl }}" target="_blank" rel="noopener" aria-label="Open figure PDF for {{ paper.title | escape }} (new tab)">
        <img src="{{ paper.image | prepend: site.baseurl }}" alt="{{ paper.image_alt }}" width="{{ paper.image_width }}" height="{{ paper.image_height }}" loading="lazy" decoding="async">
      </a>
      <figcaption><span class="publication-card__summary">{{ paper.figure_summary }}</span><span class="publication-card__year">{{ paper.year }}</span></figcaption>
    </figure>
    <div class="publication-card__body">
      <div class="publication-card__meta">
        <div class="publication-card__venues">
          <span class="publication-venue">{{ paper.venue }}</span>
          {% if paper.venue_note %}<span class="publication-venue__note">{{ paper.venue_note }}</span>{% endif %}
        </div>
        <span class="publication-card__index">{{ forloop.index }}</span>
      </div>
      <h3><a href="{{ paper.url }}">{{ paper.title }}</a></h3>
      <p class="publication-card__authors">{{ paper.authors | markdownify | remove: '<p>' | remove: '</p>' }}</p>
      <details class="publication-details publication-details--contributions" data-closed-label="Key contributions" data-open-label="Hide contributions">
        <summary><span class="publication-details__label">Key contributions</span><span class="publication-details__icon" aria-hidden="true">+</span></summary>
        <div class="publication-details__contributions"><p>{{ paper.contributions | escape }}</p></div>
      </details>
      <div class="publication-card__footer">
        <a class="publication-card__paper-link" href="{{ paper.url }}">View paper <span aria-hidden="true">↗</span></a>
        <details class="publication-details publication-details--abstract">
          <summary><span class="publication-details__label">Read abstract</span><span class="publication-details__icon" aria-hidden="true">+</span></summary>
          <div class="publication-details__abstract">{{ paper.abstract | markdownify }}</div>
        </details>
      </div>
    </div>
  </article>
  {% endfor %}
</div>

<p class="publication-note"><strong>*</strong> Equal contribution.</p>

<script src="{{ '/assets/js/publication-details.js?v=20261017c' | relative_url }}" defer></script>

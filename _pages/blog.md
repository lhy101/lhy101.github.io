---
layout: portfolio
title: "Blog"
excerpt: "Technical notes by Haoyang Li on AI systems, deep learning training, and reinforcement learning post-training."
permalink: /blog/
author_profile: false
---

<div class="blog-intro">
  <p class="eyebrow">NOTES ON AI SYSTEMS</p>
  <p>Ideas, implementation details, and lessons from building AI systems.</p>
</div>

<div class="blog-list">
  {% assign articles = site.blog | sort: 'date' | reverse %}
  {% for post in articles %}
  <article class="blog-card" lang="{{ post.lang | default: 'en' }}">
    <a class="blog-card__visual" href="{{ post.url | relative_url }}" aria-label="{{ post.title | escape }}">
      <img src="{{ post.cover | relative_url }}" alt="{{ post.cover_alt | escape }}" width="{{ post.cover_width }}" height="{{ post.cover_height }}" loading="lazy" decoding="async">
    </a>
    <div class="blog-card__body">
      <p class="blog-card__meta">{% if post.series %}<span>{{ post.series }}</span>{% endif %}<time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%Y.%m.%d' }}</time></p>
      <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <p class="blog-card__excerpt">{{ post.description }}</p>
      <a class="blog-card__read" href="{{ post.url | relative_url }}">阅读全文 <span aria-hidden="true">→</span></a>
    </div>
  </article>
  {% endfor %}
</div>

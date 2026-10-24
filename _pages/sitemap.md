---
layout: portfolio
title: "Sitemap"
permalink: /sitemap/
author_profile: false
sitemap: false
---

<nav aria-label="Site pages">
  <ul>
    <li><a href="{{ '/' | relative_url }}">Haoyang Li — Home</a></li>
    <li><a href="{{ '/publications/' | relative_url }}">Publications</a></li>
    <li><a href="{{ '/blog/' | relative_url }}">Blog</a></li>
  </ul>
</nav>

<h2>Blog articles</h2>
<ul>
{% assign posts = site.blog | sort: 'date' | reverse %}
{% for post in posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>

(() => {
  const article = document.querySelector('.blog-article__content');
  const toc = document.querySelector('.blog-toc');
  if (!article || !toc) return;

  const headings = [...article.querySelectorAll('h2, h3, h4, details.framework-detail > summary')];
  const uniqueId = base => {
    let id = base;
    let suffix = 2;
    while (document.getElementById(id)) id = `${base}-${suffix++}`;
    return id;
  };
  const disclosures = [...article.querySelectorAll('details.framework-detail')];
  disclosures.forEach((detail, index) => {
    detail.id ||= uniqueId(`framework-${index + 1}`);
  });
  const root = toc.querySelector('ol');
  root.replaceChildren();
  const outline = [];
  headings.forEach((heading, index) => {
    const isDisclosure = heading.tagName === 'SUMMARY';
    const target = isDisclosure ? heading.parentElement : heading;
    const level = isDisclosure ? 4 : Number(heading.tagName.slice(1));
    target.id ||= uniqueId(`section-${index + 1}`);
    while (outline.length && outline[outline.length - 1].level >= level) outline.pop();
    const parent = outline[outline.length - 1];
    let list = root;
    if (parent) {
      if (!parent.children) {
        parent.children = document.createElement('ol');
        parent.item.append(parent.children);
      }
      list = parent.children;
    }
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${encodeURIComponent(target.id)}`;
    link.textContent = heading.textContent.trim();
    item.append(link);
    list.append(item);
    outline.push({ level, item });
  });
  // Keep the title first while giving long articles a compact, optional outline.
  article.querySelector('h1')?.after(toc);
  toc.hidden = headings.length === 0;

  article.querySelectorAll('img').forEach(image => {
    if (image.closest('a')) return;
    const link = document.createElement('a');
    link.href = image.src;
    link.className = 'blog-figure';
    link.target = '_blank';
    link.rel = 'noopener';
    link.title = '点击查看原图';
    link.setAttribute('aria-label', `${image.alt}（查看原图）`);
    image.before(link);
    link.append(image);
  });

  // Keep table borders on the cells and scroll within the article's margins.
  article.querySelectorAll('table').forEach(table => {
    if (table.closest('.blog-table-scroll')) return;
    const scroll = document.createElement('div');
    scroll.className = 'blog-table-scroll';
    scroll.tabIndex = 0;
    scroll.setAttribute('role', 'region');
    scroll.setAttribute('aria-label', '表格（可横向滚动）');
    table.before(scroll);
    scroll.append(table);
  });

  const revealTarget = () => {
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target || !article.contains(target)) return;
    let parent = target;
    while (parent && parent !== article) {
      if (parent.tagName === 'DETAILS') parent.open = true;
      parent = parent.parentElement;
    }
    target.scrollIntoView();
  };
  window.addEventListener('hashchange', revealTarget);
  // Also reopen a disclosure when its existing hash is clicked a second time.
  toc.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (link && link.hash === window.location.hash) revealTarget();
  });
  if (window.location.hash) revealTarget();
})();

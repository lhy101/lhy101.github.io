/* Enhance native details without taking away the no-JavaScript disclosure. */
(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  document.querySelectorAll('.publication-details--abstract').forEach(details => {
    if (!details.animate) return;
    const summary = details.querySelector('summary');
    const content = details.querySelector('.publication-details__abstract');
    const label = summary?.querySelector('.publication-details__label');
    if (!summary || !content || !label) return;
    let expanded = details.open;
    let heightAnimation;
    let contentAnimation;

    const updateControl = () => {
      details.dataset.expanded = String(expanded);
      summary.setAttribute('aria-expanded', String(expanded));
      label.textContent = expanded ? 'Hide abstract' : 'Read abstract';
    };

    const finish = () => {
      details.open = expanded;
      details.style.height = '';
      details.style.overflow = '';
      heightAnimation = null;
      contentAnimation?.cancel();
      contentAnimation = null;
    };

    updateControl();

    summary.addEventListener('click', event => {
      event.preventDefault();

      const startHeight = details.getBoundingClientRect().height;
      const startOpacity = details.open ? getComputedStyle(content).opacity : 0;
      heightAnimation?.cancel();
      contentAnimation?.cancel();
      expanded = !expanded;
      updateControl();

      if (reducedMotion.matches) {
        finish();
        return;
      }

      // Keep the native content visible until the closing motion has finished.
      details.open = true;
      details.style.height = '';
      const endHeight = expanded
        ? details.getBoundingClientRect().height
        : summary.getBoundingClientRect().height;
      details.style.height = `${startHeight}px`;
      details.style.overflow = 'hidden';

      const options = { duration: 320, easing: 'cubic-bezier(.22, 1, .36, 1)' };
      heightAnimation = details.animate(
        { height: [`${startHeight}px`, `${endHeight}px`] }, options
      );
      contentAnimation = content.animate(
        { opacity: [startOpacity, expanded ? 1 : 0] },
        { ...options, fill: 'both' }
      );
      heightAnimation.onfinish = finish;
    });
  });

  document.querySelectorAll('.publication-details--contributions').forEach(details => {
    const summary = details.querySelector('summary');
    const label = summary?.querySelector('.publication-details__label');
    const content = details.querySelector('.publication-details__contributions p');
    if (!summary || !label || !content) return;

    const fullText = content.textContent.trim().replace(/\s+/g, ' ');
    const characters = Array.from(fullText);
    const accessibleText = document.createElement('span');
    accessibleText.className = 'visually-hidden';
    accessibleText.textContent = fullText;
    const visualText = document.createElement('span');
    visualText.className = 'contribution-text';
    visualText.setAttribute('aria-hidden', 'true');
    visualText.textContent = fullText;
    content.replaceChildren(accessibleText, visualText);
    let frame;
    let expanded = details.open;

    const updateControl = () => {
      details.dataset.expanded = String(expanded);
      summary.setAttribute('aria-expanded', String(expanded));
      label.textContent = expanded ? 'Hide contributions' : 'Key contributions';
    };

    const stopTyping = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = null;
      details.dataset.typing = 'false';
    };

    const finish = () => {
      stopTyping();
      visualText.textContent = fullText;
    };

    const startTyping = () => {
      stopTyping();
      if (reducedMotion.matches) {
        finish();
        return;
      }

      details.dataset.typing = 'true';
      const textNode = document.createTextNode('');
      visualText.replaceChildren(textNode);
      const start = performance.now();
      const interval = Math.min(18, 4000 / characters.length);
      let shown = 0;
      const typeNext = now => {
        const count = Math.min(characters.length, Math.floor((now - start) / interval) + 1);
        if (count > shown) {
          textNode.appendData(characters.slice(shown, count).join(''));
          shown = count;
        }
        if (count < characters.length) frame = window.requestAnimationFrame(typeNext);
        else finish();
      };
      frame = window.requestAnimationFrame(typeNext);
    };

    updateControl();

    summary.addEventListener('click', event => {
      event.preventDefault();
      expanded = !expanded;
      updateControl();

      if (!expanded) {
        finish();
        details.open = false;
        return;
      }

      details.open = true;
      startTyping();
    });

    details.addEventListener('toggle', () => {
      if (!details.open && expanded) {
        expanded = false;
        updateControl();
        finish();
      }
    });

    reducedMotion.addEventListener('change', () => {
      if (reducedMotion.matches) finish();
    });
  });
})();

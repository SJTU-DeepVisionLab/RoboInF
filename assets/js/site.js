/* RoboInF wiki — shared behaviour for every page */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ---- theme (initial value is set inline in <head> to avoid a flash) ----
  window.toggleTheme = function () {
    const h = document.documentElement;
    h.dataset.theme = h.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('roboinf-theme', h.dataset.theme);
  };

  // ---- mobile sidebar ----
  const burger = $('.burger');
  if (burger) burger.onclick = () => $('.side').classList.toggle('open');

  // ---- copy BibTeX ----
  window.copyBib = function (b) {
    navigator.clipboard.writeText($('#bib').innerText);
    b.textContent = 'Copied';
    setTimeout(() => (b.textContent = 'Copy'), 1400);
  };

  // ---- image lightbox ----
  const lb = $('#lb');
  if (lb) $$('figure .frame img').forEach(im => (im.onclick = () => { lb.querySelector('img').src = im.src; lb.classList.add('on'); }));

  // ---- "On this page" scrollspy ----
  const toc = $$('.toc a[href^="#"]');
  if (toc.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return;
      const id = e.target.id;
      toc.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + id));
    }), { rootMargin: '-40% 0px -55% 0px' });
    toc.forEach(a => { const el = document.getElementById(a.getAttribute('href').slice(1)); if (el) io.observe(el); });
  }

  // ---- results tables (home leaderboard) ----
  const R = window.RESULTS, rt = $('#rtable');
  if (R && rt) {
    const render = k => {
      const g = R.groups.find(x => x.key === k);
      let h = `<tr><th>Task</th>${g.columns.map(c => `<th>${c.model} <span class="set">${c.setting}</span></th>`).join('')}</tr>`;
      R.tasks.forEach((t, i) => {
        const js = g.columns.map(c => parseFloat(c.values[i]));
        const mx = Math.max(...js);
        const name = window.TASK_LINKS && window.TASK_LINKS[i] ? `<a href="${window.TASK_LINKS[i]}">${t}</a>` : t;
        h += `<tr class="${i === R.tasks.length - 1 ? 'avg' : ''}"><td>${name}</td>${g.columns.map((c, j) => {
          const [a, b] = c.values[i].split('/');
          return `<td class="${js[j] === mx && mx > 0 ? 'best' : ''}"><span class="j">${a}</span><span class="s"> / </span><span class="t">${b}</span></td>`;
        }).join('')}</tr>`;
      });
      rt.innerHTML = h;
      $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.k === k));
    };
    $$('#tabs button').forEach(b => (b.onclick = () => render(b.dataset.k)));
    render('clear');
  }

  // ---- branch grid: hover / tap to play ----
  $$('.branch video').forEach(v => {
    const card = v.closest('.branch');
    card.addEventListener('mouseenter', () => { v.play().catch(() => {}); card.classList.add('playing'); });
    card.addEventListener('mouseleave', () => { v.pause(); card.classList.remove('playing'); });
    card.addEventListener('click', () => {
      if (v.paused) { v.play().catch(() => {}); card.classList.add('playing'); }
      else { v.pause(); card.classList.remove('playing'); }
    });
  });
  // play / pause all branches in a grid together
  $$('[data-playall]').forEach(btn => btn.addEventListener('click', () => {
    const vids = $$('video', $(btn.dataset.playall));
    const start = vids.some(v => v.paused);
    vids.forEach(v => { if (start) { v.currentTime = 0; v.play().catch(() => {}); } else v.pause(); v.closest('.branch').classList.toggle('playing', start); });
    btn.textContent = start ? '❚❚ Pause all' : '▶ Play all in sync';
  }));

  // ---- comparison pair: restart both in sync ----
  $$('[data-sync]').forEach(btn => btn.addEventListener('click', () => {
    const vids = $$('video', $(btn.dataset.sync));
    vids.forEach(v => { v.pause(); v.currentTime = 0; });
    Promise.all(vids.map(v => v.play().catch(() => {})));
  }));
})();

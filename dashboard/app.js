/* ============================================
   OP'26 Analytics — Dashboard Interactivity
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initLazyLoading();
  initModal();
  initScrollEffects();
  animateKPIs();
});

/* ---------- Tab Switching ---------- */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      // Deactivate all
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => {
        p.classList.remove('active');
        p.style.animation = 'none';
      });

      // Activate selected
      btn.classList.add('active');
      const panel = document.getElementById(target);
      // Trigger reflow for animation restart
      void panel.offsetWidth;
      panel.style.animation = '';
      panel.classList.add('active');

      // Lazy load images in this panel
      lazyLoadPanel(panel);

      // Scroll to top of content
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

/* ---------- Lazy Loading ---------- */
function initLazyLoading() {
  // Immediately load images in the active panel
  const activePanel = document.querySelector('.tab-panel.active');
  if (activePanel) {
    lazyLoadPanel(activePanel);
  }
}

function lazyLoadPanel(panel) {
  const images = panel.querySelectorAll('img[data-src]');
  images.forEach((img, index) => {
    // Stagger the loading slightly for visual effect
    setTimeout(() => {
      loadImage(img);
    }, index * 80);
  });
}

function loadImage(img) {
  if (img.dataset.loaded === 'true') return;

  const src = img.dataset.src;
  const placeholder = img.parentElement.querySelector('.img-placeholder');

  // Create a temporary image to preload
  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = src;
    img.classList.add('loaded');
    img.dataset.loaded = 'true';
    if (placeholder) {
      placeholder.style.opacity = '0';
      setTimeout(() => placeholder.remove(), 300);
    }
  };
  tempImg.onerror = () => {
    if (placeholder) {
      placeholder.innerHTML = `
        <span style="font-size:1.5rem;">📊</span>
        <span>Image not found</span>
        <span style="font-size:0.7rem;opacity:0.5;">${src.split('/').pop()}</span>
      `;
      placeholder.querySelector('.spinner')?.remove();
    }
    img.dataset.loaded = 'error';
  };
  tempImg.src = src;
}

/* ---------- Image Modal ---------- */
function initModal() {
  const overlay = document.getElementById('imageModal');
  const modalImg = overlay.querySelector('.modal-img');
  const modalCaption = overlay.querySelector('.modal-caption');
  const closeBtn = overlay.querySelector('.modal-close');

  // Click on image cards to open modal
  document.addEventListener('click', (e) => {
    const card = e.target.closest('.image-card');
    if (!card) return;

    const img = card.querySelector('img');
    const caption = card.querySelector('.image-caption h4');
    if (!img || !img.src || img.dataset.loaded === 'error') return;

    modalImg.src = img.src;
    modalCaption.textContent = caption ? caption.textContent : '';
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  });

  // Close modal
  function closeModal() {
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
  }

  closeBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

/* ---------- Scroll Effects ---------- */
function initScrollEffects() {
  const header = document.querySelector('.header');
  let lastScroll = 0;

  window.addEventListener('scroll', () => {
    const currentScroll = window.scrollY;

    if (currentScroll > 10) {
      header.style.boxShadow = '0 4px 30px rgba(0,0,0,0.4)';
    } else {
      header.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
  }, { passive: true });

  // Intersection observer for fade-in elements
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -30px 0px' });

  document.querySelectorAll('.image-card, .kpi-card, .method-card, .metric-card, .arch-stage').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}

/* ---------- KPI Number Animation ---------- */
function animateKPIs() {
  const kpiValues = document.querySelectorAll('.kpi-value[data-count]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.dataset.count);
        const prefix = el.dataset.prefix || '';
        const suffix = el.dataset.suffix || '';
        const duration = 1500;
        const start = performance.now();

        function update(now) {
          const elapsed = now - start;
          const progress = Math.min(elapsed / duration, 1);
          // Ease out cubic
          const eased = 1 - Math.pow(1 - progress, 3);
          const current = Math.round(eased * target);
          el.textContent = prefix + current.toLocaleString() + suffix;

          if (progress < 1) {
            requestAnimationFrame(update);
          }
        }

        requestAnimationFrame(update);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  kpiValues.forEach(el => observer.observe(el));
}

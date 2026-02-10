/**
 * GenJAX Project Website - Interactive Features
 * Dual-track navigation: Tutorial (pedagogical) + Theory (formal PL)
 * Inspired by: https://tomasp.net/coeffects/
 */

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    storageKey: 'genjax-track-preference',
    defaultTrack: 'tutorial',
    tracks: ['tutorial', 'theory', 'all'],
    animationDuration: 300
  };

  // State
  let currentTrack = CONFIG.defaultTrack;
  let isInitialized = false;

  /**
   * Initialize the website
   */
  function init() {
    if (isInitialized) return;
    
    // Mark JS as enabled for CSS styling
    document.body.classList.add('js-enabled');
    
    loadTrackPreference();
    setupTrackSwitcher();
    setupSmoothScroll();
    setupActiveSectionHighlight();
    setupKeyboardNavigation();
    setupBibTexCopy();
    setupSectionAnchors();
    
    // Apply initial track
    applyTrack(currentTrack, false);
    
    isInitialized = true;
    console.log('GenJAX website initialized');
  }

  /**
   * Load track preference from localStorage
   * Note: We do NOT use URL hash for track state to preserve section anchors
   */
  function loadTrackPreference() {
    try {
      const saved = localStorage.getItem(CONFIG.storageKey);
      if (saved && CONFIG.tracks.includes(saved)) {
        currentTrack = saved;
      }
      // Note: Hash is not used for track state - section anchors (#evaluation, etc.) 
      // remain functional for deep-linking to content sections
    } catch (e) {
      // localStorage may be disabled
      console.warn('Could not load track preference:', e);
    }
  }

  /**
   * Save track preference to localStorage
   */
  function saveTrackPreference(track) {
    try {
      localStorage.setItem(CONFIG.storageKey, track);
    } catch (e) {
      console.warn('Could not save track preference:', e);
    }
  }

  /**
   * Setup track switcher buttons
   */
  function setupTrackSwitcher() {
    const buttons = document.querySelectorAll('.track-btn');
    
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const track = btn.id.replace('btn-', '');
        if (track !== currentTrack) {
          switchTrack(track);
        }
      });
    });
  }

  /**
   * Switch to a different track
   * @param {string} track - 'tutorial', 'theory', or 'all'
   * @param {boolean} save - whether to save preference
   */
  function switchTrack(track, save = true) {
    if (!CONFIG.tracks.includes(track)) return;
    
    currentTrack = track;
    applyTrack(track, true);
    
    if (save) {
      saveTrackPreference(track);
    }
    
    // Note: We intentionally do NOT use URL hash for track state
    // This keeps section anchors (#evaluation, #artifact, etc.) shareable
    // Track preference is stored in localStorage only
    
    // Dispatch custom event
    document.dispatchEvent(new CustomEvent('trackchange', { 
      detail: { track: track } 
    }));
  }

  /**
   * Apply track state to UI
   * @param {string} track - current track
   * @param {boolean} animate - whether to animate
   */
  function applyTrack(track, animate) {
    const buttons = document.querySelectorAll('.track-btn');
    const contents = document.querySelectorAll('.track-content');
    const body = document.body;
    
    // Handle "all" mode
    if (track === 'all') {
      body.classList.add('all-mode');
      
      // Update all buttons
      buttons.forEach(btn => {
        const btnTrack = btn.id.replace('btn-', '');
        const isActive = btnTrack === 'all';
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
        btn.setAttribute('tabindex', isActive ? '0' : '-1');
      });
      
      // Re-render MathJax for all content
      if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.typesetPromise().catch(console.error);
      }
      
      document.title = 'GenJAX: Complete View — Vectorized Probabilistic Programming';
      return;
    }
    
    // Normal single-track mode
    body.classList.remove('all-mode');
    
    // Update buttons
    buttons.forEach(btn => {
      const btnTrack = btn.id.replace('btn-', '');
      const isActive = btnTrack === track;
      
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      btn.setAttribute('tabindex', isActive ? '0' : '-1');
    });
    
    // Update content with animation
    contents.forEach(content => {
      const contentTrack = content.id.replace('track-', '');
      const isActive = contentTrack === track;
      
      if (isActive) {
        content.classList.add('active');
        if (animate) {
          content.style.animation = 'none';
          content.offsetHeight; // Force reflow
          content.style.animation = 'fadeIn 0.3s ease';
        }
        // Re-render MathJax if present
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
          MathJax.typesetPromise([content]).catch(console.error);
        }
      } else {
        content.classList.remove('active');
      }
    });
    
    // Update document title
    const trackLabel = track === 'tutorial' ? 'Tutorial' : 'Theory';
    document.title = `GenJAX: ${trackLabel} Track — Vectorized Probabilistic Programming`;
    
    // Focus management for accessibility
    const activeBtn = document.getElementById('btn-' + track);
    if (activeBtn && document.activeElement?.classList.contains('track-btn')) {
      activeBtn.focus();
    }
  }

  /**
   * Setup smooth scroll for anchor links
   */
  function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        
        // Handle section links
        if (href !== '#') {
          e.preventDefault();
          const target = document.querySelector(href);
          if (target) {
            const trackSection = target.closest('.track-content');
            const targetTrack = trackSection ? trackSection.id.replace('track-', '') : null;
            
            if (targetTrack && targetTrack !== 'all' && currentTrack !== 'all' && targetTrack !== currentTrack) {
              switchTrack(targetTrack);
            }
            
            requestAnimationFrame(() => {
              const offset = 80; // Account for sticky nav
              const top = target.getBoundingClientRect().top + window.scrollY - offset;
              
              window.scrollTo({
                top: top,
                behavior: 'smooth'
              });
              
              // Update focus for accessibility
              target.setAttribute('tabindex', '-1');
              target.focus({ preventScroll: true });
              
              // Update URL
              history.pushState(null, null, href);
            });
          }
        }
      });
    });
  }

  /**
   * Setup active section highlighting in navigation
   */
  function setupActiveSectionHighlight() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    
    if (sections.length === 0 || navLinks.length === 0) return;
    
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -70% 0px',
      threshold: 0
    };
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const navSection = entry.target.closest('section[data-nav-section]');
          const navId = navSection ? navSection.id : entry.target.id;
          
          navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + navId) {
              link.classList.add('active');
            }
          });
          
          const hasNestedSections = entry.target.querySelector('section[id]');
          
          // Also update sidebar TOC (prefer nested sections over parent containers)
          if (!(entry.target.hasAttribute('data-nav-section') && hasNestedSections)) {
            document.querySelectorAll('.toc-nav a').forEach(link => {
              link.classList.remove('active');
              if (link.getAttribute('href') === '#' + entry.target.id) {
                link.classList.add('active');
              }
            });
          }
        }
      });
    }, observerOptions);
    
    sections.forEach(section => observer.observe(section));
  }

  /**
   * Setup keyboard navigation for track switcher
   */
  function setupKeyboardNavigation() {
    const trackButtons = document.querySelectorAll('.track-btn');
    
    trackButtons.forEach(btn => {
      btn.addEventListener('keydown', (e) => {
        const buttons = Array.from(trackButtons);
        const index = buttons.indexOf(btn);
        
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          e.preventDefault();
          const next = buttons[(index + 1) % buttons.length];
          next.focus();
          next.click();
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          e.preventDefault();
          const prev = buttons[(index - 1 + buttons.length) % buttons.length];
          prev.focus();
          prev.click();
        }
      });
    });
  }

  /**
   * Setup BibTeX copy button
   */
  function setupBibTexCopy() {
    const copyBtn = document.querySelector('.copy-btn');
    const bibtexBlock = document.getElementById('bibtex');
    
    if (!copyBtn || !bibtexBlock) return;
    
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(bibtexBlock.textContent);
        
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(() => {
          copyBtn.textContent = originalText;
          copyBtn.classList.remove('copied');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
        copyBtn.textContent = 'Failed';
        setTimeout(() => copyBtn.textContent = 'Copy', 2000);
      }
    });
  }

  /**
   * Setup section anchor link behavior
   */
  function setupSectionAnchors() {
    document.querySelectorAll('.section-anchor').forEach(anchor => {
      anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const href = anchor.getAttribute('href');
        
        // Copy link to clipboard
        const fullUrl = window.location.origin + window.location.pathname + href;
        navigator.clipboard.writeText(fullUrl).catch(() => {});
        
        // Navigate to section
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          history.pushState(null, null, href);
        }
      });
    });
  }

  /**
   * Expose public API
   */
  window.GenJAX = {
    switchTrack: switchTrack,
    getCurrentTrack: () => currentTrack,
    CONFIG: CONFIG
  };

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

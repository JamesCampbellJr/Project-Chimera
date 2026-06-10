/* ═══════════════════════════════════════════════════
   NOODLES MEDIA — Main JavaScript
   ═══════════════════════════════════════════════════ */

'use strict';

/* ─── Utility ─── */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ═══════════════════════════════════════════════════
   1. NAVBAR
   ═══════════════════════════════════════════════════ */
(function initNav() {
  const navbar = $('#navbar');
  const hamburger = $('#hamburger');
  const navLinks = $('#navLinks');

  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
    $('#backToTop').classList.toggle('visible', window.scrollY > 400);
  }, { passive: true });

  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    const open = navLinks.classList.contains('open');
    hamburger.setAttribute('aria-expanded', open);
  });

  // Close mobile menu on link click
  $$('.nav-links a').forEach(a => a.addEventListener('click', () => {
    navLinks.classList.remove('open');
  }));

  // Back to top
  $('#backToTop').addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
})();

/* ═══════════════════════════════════════════════════
   2. PARTICLE CANVAS
   ═══════════════════════════════════════════════════ */
(function initParticles() {
  const canvas = $('#particleCanvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [], animId;

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.r = Math.random() * 1.5 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.1;
      this.color = Math.random() > 0.5 ? '108,63,255' : '255,107,107';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color},${this.alpha})`;
      ctx.fill();
    }
  }

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(108,63,255,${0.12 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    animId = requestAnimationFrame(loop);
  }

  function init() {
    resize();
    const count = Math.min(80, Math.floor(W * H / 12000));
    particles = Array.from({ length: count }, () => new Particle());
    if (animId) cancelAnimationFrame(animId);
    loop();
  }

  window.addEventListener('resize', init, { passive: true });
  init();
})();

/* ═══════════════════════════════════════════════════
   3. TYPEWRITER
   ═══════════════════════════════════════════════════ */
(function initTypewriter() {
  const el = $('#typewriter');
  const phrases = [
    'Redefined.',
    'Accessible.',
    'For Everyone.',
    'That Works.',
    'Made Simple.',
  ];
  let phraseIdx = 0, charIdx = 0, deleting = false;

  async function tick() {
    const phrase = phrases[phraseIdx];
    if (!deleting) {
      el.textContent = phrase.slice(0, ++charIdx);
      if (charIdx === phrase.length) {
        deleting = true;
        await sleep(2200);
      }
      setTimeout(tick, deleting ? 0 : 80);
    } else {
      el.textContent = phrase.slice(0, --charIdx);
      if (charIdx === 0) {
        deleting = false;
        phraseIdx = (phraseIdx + 1) % phrases.length;
      }
      setTimeout(tick, 45);
    }
  }
  tick();
})();

/* ═══════════════════════════════════════════════════
   4. COUNTER ANIMATION
   ═══════════════════════════════════════════════════ */
(function initCounters() {
  const counters = $$('.stat-num');
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const target = +el.dataset.target;
      let current = 0;
      const step = target / 60;
      const update = () => {
        current = Math.min(current + step, target);
        el.textContent = Math.floor(current);
        if (current < target) requestAnimationFrame(update);
      };
      update();
      io.unobserve(el);
    });
  }, { threshold: 0.5 });
  counters.forEach(c => io.observe(c));
})();

/* ═══════════════════════════════════════════════════
   5. SCROLL ANIMATIONS
   ═══════════════════════════════════════════════════ */
(function initScrollAnim() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('animated');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });
  $$('[data-aos]').forEach(el => io.observe(el));

  // Service cards stagger
  const cards = $$('.service-card');
  const io2 = new IntersectionObserver(entries => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        setTimeout(() => {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
        }, i * 80);
        io2.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });
  cards.forEach(c => {
    c.style.opacity = '0';
    c.style.transform = 'translateY(24px)';
    c.style.transition = 'opacity 0.5s ease, transform 0.5s ease, border-color 0.3s, box-shadow 0.3s';
    io2.observe(c);
  });
})();

/* ═══════════════════════════════════════════════════
   6. QUIZ / AI FINDER
   ═══════════════════════════════════════════════════ */
(function initQuiz() {
  const answers = {};
  let currentStep = 1;
  const totalSteps = 5;

  // AI Solution recommendations
  const solutions = {
    chatbot: {
      icon: '💬',
      title: 'Conversational AI',
      desc: 'A smart chatbot that handles your customer interactions 24/7 with context-aware responses.',
      match: null,
    },
    generative: {
      icon: '✨',
      title: 'Generative AI Studio',
      desc: 'AI-powered content generation aligned with your brand voice — copy, emails, social, and more.',
      match: null,
    },
    analytics: {
      icon: '🔍',
      title: 'Predictive Analytics',
      desc: 'Data-driven forecasting that helps you anticipate demand, churn, and opportunities.',
      match: null,
    },
    automation: {
      icon: '⚙️',
      title: 'Intelligent Automation',
      desc: 'AI workflows that eliminate manual repetitive tasks and boost operational efficiency.',
      match: null,
    },
    vision: {
      icon: '👁️',
      title: 'Computer Vision',
      desc: 'Visual AI for quality control, retail analytics, document scanning, and more.',
      match: null,
    },
  };

  function getRecommendations() {
    const scores = {
      chatbot:    0,
      generative: 0,
      analytics:  0,
      automation: 0,
      vision:     0,
    };

    // Scoring logic based on answers
    if (answers[2] === 'support')     { scores.chatbot += 40; scores.automation += 20; }
    if (answers[2] === 'content')     { scores.generative += 40; scores.chatbot += 10; }
    if (answers[2] === 'data')        { scores.analytics += 40; scores.vision += 15; }
    if (answers[2] === 'automation')  { scores.automation += 40; scores.analytics += 15; }

    if (answers[5] === 'revenue')     { scores.generative += 20; scores.chatbot += 15; scores.analytics += 10; }
    if (answers[5] === 'efficiency')  { scores.automation += 25; scores.analytics += 15; }
    if (answers[5] === 'experience')  { scores.chatbot += 25; scores.generative += 10; }
    if (answers[5] === 'innovation')  { scores.vision += 20; scores.analytics += 15; scores.generative += 15; }

    if (answers[4] === 'none')        { scores.chatbot += 10; scores.automation += 10; }
    if (answers[4] === 'advanced')    { scores.analytics += 10; scores.vision += 10; }

    if (answers[1] === 'agency')      { scores.generative += 20; }
    if (answers[1] === 'enterprise')  { scores.automation += 15; scores.analytics += 10; }

    // Add base scores so all non-zero
    Object.keys(scores).forEach(k => scores[k] = Math.max(scores[k] + 10, 10));

    return Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([key, score]) => ({ ...solutions[key], key, match: Math.min(score, 99) }));
  }

  function goToStep(step) {
    $$('.quiz-step').forEach(s => s.classList.remove('active'));
    const target = $(`.quiz-step[data-step="${step}"]`);
    if (target) target.classList.add('active');

    if (step === 'results') {
      $('#quizProgressBar').style.width = '100%';
      $('#quizStepLabel').textContent = 'Your personalized recommendations';
      showResults();
    } else {
      const pct = ((step - 1) / totalSteps * 100) + 20;
      $('#quizProgressBar').style.width = `${Math.min(pct, 80)}%`;
      $('#quizStepLabel').textContent = `Step ${step} of ${totalSteps}`;
    }
    currentStep = step;
  }

  function showResults() {
    const recs = getRecommendations();
    const intro = {
      startup:    'Based on your startup profile, here\'s your recommended AI roadmap:',
      smb:        'For your SMB, we\'d suggest starting with these high-impact AI solutions:',
      enterprise: 'For enterprise scale, these AI solutions will drive the most value:',
      agency:     'As a creative agency, these AI tools will amplify your team\'s output:',
    };
    $('#resultsIntro').textContent = intro[answers[1]] || 'Here\'s your personalized AI roadmap:';

    const container = $('#resultsCards');
    container.innerHTML = recs.map((r, i) => `
      <div class="result-card" style="animation-delay:${i * 0.1}s">
        <div class="result-card-icon">${r.icon}</div>
        <h4>${r.title}</h4>
        <p>${r.desc}</p>
        <span class="match-badge">${r.match}% match</span>
      </div>
    `).join('');
  }

  // Delegate click on quiz options
  $('#quizSteps').addEventListener('click', e => {
    const btn = e.target.closest('.quiz-option');
    if (!btn) return;
    const step = btn.closest('.quiz-step');
    const stepNum = +step.dataset.step;
    const value = btn.dataset.value;

    answers[stepNum] = value;

    // Highlight selected
    $$('.quiz-option', step).forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');

    // Advance after short delay
    setTimeout(() => {
      if (stepNum < totalSteps) {
        goToStep(stepNum + 1);
      } else {
        goToStep('results');
      }
    }, 350);
  });

  $('#restartQuiz').addEventListener('click', () => {
    Object.keys(answers).forEach(k => delete answers[k]);
    goToStep(1);
  });
})();

/* ═══════════════════════════════════════════════════
   7. DEMO TABS
   ═══════════════════════════════════════════════════ */
(function initDemoTabs() {
  $$('.demo-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('.demo-tab').forEach(t => t.classList.remove('active'));
      $$('.demo-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      $(`#demo-${tab.dataset.demo}`).classList.add('active');
    });
  });
})();

/* ═══════════════════════════════════════════════════
   8. CHATBOT DEMO
   ═══════════════════════════════════════════════════ */
(function initChatbot() {
  const window_ = $('#chatWindow');
  const input   = $('#chatInput');
  const sendBtn = $('#chatSend');

  // Knowledge base for the demo bot
  const KB = [
    {
      keywords: ['marketing', 'content', 'copy', 'brand'],
      response: "Great question! For marketing, our **Generative AI Studio** can produce blog posts, ad copy, social content, and emails at scale — all tuned to your brand voice. We typically see clients 10× their content output while maintaining quality. Would you like a demo of our content generator? 👉",
    },
    {
      keywords: ['customer support', 'support', 'chatbot', 'helpdesk', 'ticket'],
      response: "Our **Conversational AI** solution handles support queries 24/7 with context-aware dialogue. Clients typically deflect 60–80% of tickets — your team focuses on the complex stuff while the AI handles the routine. It integrates with Zendesk, Intercom, Salesforce, and more. Want to see it in action?",
    },
    {
      keywords: ['data', 'analytics', 'forecast', 'predict', 'churn'],
      response: "**Predictive Analytics** is one of our strongest offerings. We build custom models that forecast demand, predict customer churn, identify upsell opportunities, and detect anomalies — all delivered through real-time dashboards. What kind of data are you working with?",
    },
    {
      keywords: ['automate', 'automation', 'workflow', 'process', 'repetitive', 'manual'],
      response: "**Intelligent Automation** is perfect if your team spends hours on repetitive tasks. We use AI to automate document processing, data entry, scheduling, email triage, and more. One client cut document processing time from 3 days to 4 hours! What processes are slowing your team down?",
    },
    {
      keywords: ['image', 'vision', 'photo', 'camera', 'detect', 'visual'],
      response: "Our **Computer Vision** platform can detect objects, read documents, analyze faces, and understand scenes in real time. Popular use cases include retail shelf analytics, quality control in manufacturing, and medical image screening. What are you trying to 'see'?",
    },
    {
      keywords: ['price', 'cost', 'pricing', 'expensive', 'cheap', 'budget'],
      response: "Our pricing is project-based and scales to your needs. Starter packages begin at $5,000 for a proof-of-concept, full production deployments typically range from $20K–$200K depending on complexity. We always start with a **free consultation** and a detailed proposal so there are no surprises. Want to book a call?",
    },
    {
      keywords: ['llm', 'gpt', 'claude', 'gemini', 'openai', 'model'],
      response: "We work with all major LLMs — GPT-4o, Claude 3, Gemini, Mistral, and open-source models. We help you choose the right one for your use case based on cost, speed, privacy, and capability requirements. Sometimes a fine-tuned smaller model outperforms a giant one for specific tasks!",
    },
    {
      keywords: ['security', 'privacy', 'gdpr', 'compliance', 'safe'],
      response: "Security and compliance are baked in from day one. We offer on-premise deployment, data anonymization, role-based access, audit logs, and full GDPR/HIPAA compliance frameworks. Our **AI Security** service also includes red-team testing and bias auditing.",
    },
    {
      keywords: ['start', 'begin', 'get started', 'first step', 'how'],
      response: "The best first step is a **free 30-minute discovery call** with our team. We'll learn about your business, identify the top 3 AI opportunities, and give you an honest assessment — no sales pitch. Want me to connect you with our team? 📅",
    },
    {
      keywords: ['noodles', 'company', 'about', 'who are you', 'who made'],
      response: "Noodles Media is an AI-first consultancy and product studio. We build custom AI solutions for startups to enterprises across industries. Founded by engineers and AI researchers, we've deployed 500+ AI models across 30+ countries. Our name? Like a perfect bowl of noodles, the best AI systems are deeply interconnected. 🍜",
    },
  ];

  const fallbacks = [
    "That's a fascinating question! The AI opportunity in that area is significant. Could you share a bit more about your specific use case so I can give you a more tailored answer?",
    "I love that you're thinking about AI here. Based on what you've described, I'd recommend scheduling a quick discovery call with our team — they can give you a precise recommendation. Want me to help you get started?",
    "Great topic! This is something we help clients with regularly. What industry are you in, and what's the main problem you're trying to solve?",
    "Interesting! The AI landscape moves fast. Tell me more about your current setup and I can suggest the most efficient path forward.",
  ];
  let fallbackIdx = 0;

  function getReply(text) {
    const lower = text.toLowerCase();
    for (const item of KB) {
      if (item.keywords.some(kw => lower.includes(kw))) {
        return item.response;
      }
    }
    return fallbacks[fallbackIdx++ % fallbacks.length];
  }

  function addMessage(text, role) {
    const msg = document.createElement('div');
    msg.className = `chat-message ${role}`;
    const avatar = document.createElement('div');
    avatar.className = `chat-avatar ${role === 'bot' ? 'bot' : 'user'}-avatar`;
    avatar.textContent = role === 'bot' ? '🤖' : '👤';
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    // Basic markdown-like bold
    bubble.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    msg.append(avatar, bubble);
    window_.appendChild(msg);
    window_.scrollTop = window_.scrollHeight;
    return msg;
  }

  function showTyping() {
    const msg = document.createElement('div');
    msg.className = 'chat-message bot typing-indicator';
    msg.innerHTML = `
      <div class="chat-avatar bot-avatar">🤖</div>
      <div class="chat-bubble">
        <span class="dot-blink"></span>
        <span class="dot-blink"></span>
        <span class="dot-blink"></span>
      </div>`;
    window_.appendChild(msg);
    window_.scrollTop = window_.scrollHeight;
    return msg;
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage(text, 'user');
    sendBtn.disabled = true;
    const typing = showTyping();
    await sleep(900 + Math.random() * 600);
    typing.remove();
    const reply = getReply(text);
    addMessage(reply, 'bot');
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

  $$('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.text;
      send();
    });
  });
})();

/* ═══════════════════════════════════════════════════
   9. SENTIMENT ANALYZER DEMO
   ═══════════════════════════════════════════════════ */
(function initSentiment() {
  const POSITIVE_WORDS = ['great','good','excellent','amazing','fantastic','wonderful','love','best','outstanding','awesome','happy','pleased','satisfied','perfect','brilliant','superb','delightful','enjoy','enjoyed','incredible','helpful','fast','friendly'];
  const NEGATIVE_WORDS = ['bad','terrible','awful','horrible','hate','worst','poor','disappointing','frustrated','annoying','slow','rude','broken','failed','useless','disgusting','angry','upset','furious','complaint','issue','problem','wrong','never','never again'];

  function analyze(text) {
    const words = text.toLowerCase().split(/\W+/);
    let pos = 0, neg = 0;
    const themes = [];

    words.forEach(w => {
      if (POSITIVE_WORDS.includes(w)) pos++;
      if (NEGATIVE_WORDS.includes(w)) neg++;
    });

    // Theme extraction (simple keyword categories)
    const categories = {
      'Delivery': ['delivery','shipping','arrived','late','fast','slow'],
      'Product Quality': ['product','quality','broken','excellent','durable','cheap'],
      'Customer Service': ['service','support','staff','rude','helpful','responsive'],
      'Value': ['price','expensive','cheap','value','worth','money'],
      'User Experience': ['easy','difficult','confusing','intuitive','interface','app'],
    };
    Object.entries(categories).forEach(([cat, kws]) => {
      if (kws.some(kw => text.toLowerCase().includes(kw))) themes.push(cat);
    });

    const total = pos + neg || 1;
    const posScore = Math.round((pos / total) * 100);
    const negScore = Math.round((neg / total) * 100);
    const neutral = Math.max(0, 100 - posScore - negScore);

    let overall, emoji, label;
    if (pos > neg * 1.5) { overall = 'Positive'; emoji = '😊'; label = 'positive'; }
    else if (neg > pos * 1.5) { overall = 'Negative'; emoji = '😟'; label = 'negative'; }
    else if (pos > 0 && neg > 0) { overall = 'Mixed'; emoji = '😐'; label = 'mixed'; }
    else { overall = 'Neutral'; emoji = '😑'; label = 'neutral'; }

    return { posScore, negScore, neutral, overall, emoji, label, themes };
  }

  $('#analyzeSentiment').addEventListener('click', () => {
    const text = $('#sentimentInput').value.trim();
    if (!text) {
      $('#sentimentResult').innerHTML = '<p style="color:var(--accent)">Please enter some text to analyze.</p>';
      return;
    }

    const btn = $('#analyzeSentiment');
    btn.innerHTML = '<span class="spinner"></span> Analyzing…';
    btn.disabled = true;

    setTimeout(() => {
      const r = analyze(text);
      const themesHTML = r.themes.length
        ? `<div class="key-themes">${r.themes.map(t => `<span class="theme-tag">${t}</span>`).join('')}</div>`
        : '';

      $('#sentimentResult').innerHTML = `
        <div class="sentiment-summary">
          <div class="sentiment-emoji">${r.emoji}</div>
          <strong style="color:#fff;font-size:1.1rem">Overall: ${r.overall}</strong>
          <p style="margin-top:8px;font-size:0.9rem">Detected ${r.posScore > r.negScore ? 'predominantly positive' : r.negScore > r.posScore ? 'predominantly negative' : 'mixed'} sentiment across ${text.split(/\s+/).length} words.</p>
          ${themesHTML}
        </div>
        <div class="sentiment-gauge">
          <div class="gauge-item">
            <div class="gauge-label"><span>😊 Positive</span><span id="posVal">0%</span></div>
            <div class="gauge-bar"><div class="gauge-fill positive" id="posBar"></div></div>
          </div>
          <div class="gauge-item">
            <div class="gauge-label"><span>😟 Negative</span><span id="negVal">0%</span></div>
            <div class="gauge-bar"><div class="gauge-fill negative" id="negBar"></div></div>
          </div>
          <div class="gauge-item">
            <div class="gauge-label"><span>😑 Neutral</span><span id="neuVal">0%</span></div>
            <div class="gauge-bar"><div class="gauge-fill neutral" id="neuBar"></div></div>
          </div>
        </div>`;

      // Animate bars
      requestAnimationFrame(() => {
        $('#posBar').style.width = `${r.posScore}%`;
        $('#negBar').style.width = `${r.negScore}%`;
        $('#neuBar').style.width = `${r.neutral}%`;
        $('#posVal').textContent = `${r.posScore}%`;
        $('#negVal').textContent = `${r.negScore}%`;
        $('#neuVal').textContent = `${r.neutral}%`;
      });

      btn.innerHTML = 'Analyze Sentiment';
      btn.disabled = false;
    }, 900);
  });
})();

/* ═══════════════════════════════════════════════════
   10. CONTENT GENERATOR DEMO
   ═══════════════════════════════════════════════════ */
(function initGenerator() {
  const templates = {
    tagline: (topic) => [
      `"Where ${topic} meets the future."`,
      `"${capitalize(topic)}: reimagined for a smarter world."`,
      `"Experience ${topic} like never before."`,
      `"The ${topic} revolution starts here."`,
      `"Bold ${topic}. Brilliant results."`,
    ],
    email: (topic) => [`Subject: You've never seen ${topic} like this before 🚀

Hi [First Name],

We know your inbox is crowded — so we'll get straight to the point.

We've just launched something that's going to completely change how you think about ${topic}.

After months of development and feedback from hundreds of customers like you, we're proud to introduce our latest innovation — designed specifically for people who demand more.

Here's what makes it different:
• Built with cutting-edge technology
• Proven results with real customers  
• Zero learning curve — start getting value on day one

The early access window closes in 48 hours.

[👉 Claim Your Spot]

Talk soon,
The Team

P.S. Reply to this email with any questions — we actually read every one.`],
    social: (topic) => [
      `🚀 Big news for everyone in the ${topic} space!\n\nWe've been working behind the scenes on something that's going to shake things up. And it's finally here.\n\n✅ Faster\n✅ Smarter\n✅ Built for you\n\nDrop a 🔥 in the comments if you want early access!\n\n#${topic.replace(/\s+/g, '')} #Innovation #AI`,
      `Hot take: Most ${topic} solutions are solving yesterday's problems.\n\nWe built ours for tomorrow. Here's why that matters 👇\n\n[Thread 1/7]\n\n#${topic.replace(/\s+/g, '')} #FutureOfWork`,
      `We asked 500 people what they hate most about ${topic}.\n\nThe #1 answer? It takes too long.\n\nWe fixed that. ⚡\n\n#${topic.replace(/\s+/g, '')} #ProductLaunch`,
    ],
    blog: (topic) => [`# The Complete Guide to ${capitalize(topic)} in 2026

The landscape of ${topic} has changed dramatically. What worked two years ago barely scratches the surface of what's possible today — and organizations that haven't adapted are already falling behind.

In this guide, we'll cover everything you need to know: from the core principles that have stood the test of time, to the cutting-edge approaches that are reshaping the industry in real time.

**What You'll Learn:**
- Why traditional approaches to ${topic} are no longer sufficient
- The 3 biggest trends redefining the space right now
- A practical framework you can implement starting today
- Real-world case studies from companies doing it right

Whether you're a seasoned professional or just starting to explore ${topic}, this guide will give you the clarity and confidence to take action.

Let's dive in.`],
    ad: (topic) => [
      `Stop settling for mediocre ${topic}.\n\nYou deserve better — and now you can have it.\n\n🎯 Results in days, not months\n💰 ROI you can measure\n🔒 Zero risk — try it free\n\n[Start Free Trial →]`,
      `If your ${topic} strategy isn't working, it's not your fault.\n\nMost tools weren't built for businesses like yours.\n\nWe were. Here's proof: our customers see 3× better results in 30 days.\n\n[See How It Works]`,
    ],
  };

  function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  let currentContent = '';
  let copyBtn = null;

  $('#generateContent').addEventListener('click', () => {
    const type = $('#contentType').value;
    const topic = $('#contentTopic').value.trim() || 'your product';
    const btn = $('#generateContent');

    btn.innerHTML = '<span class="spinner"></span> Generating…';
    btn.disabled = true;

    setTimeout(() => {
      const options = templates[type](topic);
      const picked = options[Math.floor(Math.random() * options.length)];
      currentContent = picked;

      const result = $('#generatorResult');
      result.innerHTML = `<div class="generated-content" id="genText"></div>`;

      // Typewriter effect for generated content
      const genEl = $('#genText');
      let i = 0;
      function typeit() {
        if (i <= picked.length) {
          genEl.textContent = picked.slice(0, i++);
          requestAnimationFrame(typeit);
        } else {
          // Add copy button
          copyBtn = document.createElement('button');
          copyBtn.className = 'copy-btn';
          copyBtn.textContent = '📋 Copy to Clipboard';
          copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(currentContent).then(() => {
              copyBtn.textContent = '✅ Copied!';
              setTimeout(() => { copyBtn.textContent = '📋 Copy to Clipboard'; }, 2000);
            });
          });
          result.appendChild(copyBtn);
        }
      }
      typeit();

      btn.innerHTML = 'Generate ✨';
      btn.disabled = false;
    }, 800);
  });
})();

/* ═══════════════════════════════════════════════════
   11. CLASSIFIER DEMO
   ═══════════════════════════════════════════════════ */
(function initClassifier() {
  const categoryKeywords = {
    support:   ['help','issue','problem','broken','not working','fix','error','bug','crash','fail'],
    billing:   ['invoice','refund','charge','payment','bill','money','price','cost','subscription','cancel'],
    sales:     ['buy','purchase','demo','trial','price','plan','upgrade','discount','offer','interested'],
    technical: ['install','setup','configure','api','integration','code','developer','sdk','token','server'],
    feedback:  ['suggest','feedback','love','hate','improve','feature','wish','idea','amazing','terrible'],
  };

  $('#classifyText').addEventListener('click', () => {
    const text = $('#classifierInput').value.trim();
    if (!text) {
      $('#classifierResult').innerHTML = '<p style="color:var(--accent)">Please enter some text to classify.</p>';
      return;
    }

    const enabled = $$('.classifier-categories input:checked').map(c => c.value);
    if (!enabled.length) {
      $('#classifierResult').innerHTML = '<p style="color:var(--accent)">Please select at least one category.</p>';
      return;
    }

    const btn = $('#classifyText');
    btn.innerHTML = '<span class="spinner"></span> Classifying…';
    btn.disabled = true;

    setTimeout(() => {
      const lower = text.toLowerCase();
      const scores = {};

      enabled.forEach(cat => {
        const kws = categoryKeywords[cat] || [];
        let score = 10; // base
        kws.forEach(kw => { if (lower.includes(kw)) score += 25; });
        // Add noise
        score += Math.floor(Math.random() * 15);
        scores[cat] = Math.min(score, 99);
      });

      // Normalize so highest = 95%+
      const max = Math.max(...Object.values(scores));
      const winner = Object.entries(scores).find(([, v]) => v === max)[0];

      const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);

      $('#classifierResult').innerHTML = `
        <p style="font-size:0.9rem;margin-bottom:12px">
          <strong style="color:#fff">Best match:</strong> 
          <span class="cls-winner">${winner.charAt(0).toUpperCase() + winner.slice(1)}</span>
          — this message would be routed to the <strong style="color:#fff">${winner}</strong> team.
        </p>
        <div class="classifier-bars" id="clsBars"></div>`;

      const barsContainer = $('#clsBars');
      sorted.forEach(([cat, score], i) => {
        const pct = Math.round((score / max) * 90 + 10);
        const item = document.createElement('div');
        item.className = 'cls-item';
        item.innerHTML = `
          <div class="cls-label">
            <span>${cat.charAt(0).toUpperCase() + cat.slice(1)}</span>
            <span>${pct}%</span>
          </div>
          <div class="cls-bar"><div class="cls-fill" id="clf-${i}" style="width:0"></div></div>`;
        barsContainer.appendChild(item);
        setTimeout(() => {
          $(`#clf-${i}`).style.width = `${pct}%`;
        }, i * 100 + 50);
      });

      btn.innerHTML = 'Classify';
      btn.disabled = false;
    }, 700);
  });

  $('#classifierInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('#classifyText').click();
  });
})();

/* ═══════════════════════════════════════════════════
   12. TESTIMONIALS CAROUSEL
   ═══════════════════════════════════════════════════ */
(function initTestimonials() {
  const track = $('#testimonialTrack');
  const cards = $$('.testimonial-card', track);
  const dotsContainer = $('#tDots');
  let current = 0;
  let autoId;

  // Create dots
  cards.forEach((_, i) => {
    const dot = document.createElement('div');
    dot.className = `t-dot${i === 0 ? ' active' : ''}`;
    dot.addEventListener('click', () => goTo(i));
    dotsContainer.appendChild(dot);
  });

  function goTo(idx) {
    current = (idx + cards.length) % cards.length;
    // Simple transform-based scroll
    const cardW = cards[0].offsetWidth + 24;
    track.style.transform = `translateX(-${current * cardW}px)`;
    track.style.transition = 'transform 0.5s cubic-bezier(0.4,0,0.2,1)';
    $$('.t-dot', dotsContainer).forEach((d, i) => d.classList.toggle('active', i === current));
    resetAuto();
  }

  function resetAuto() {
    clearInterval(autoId);
    autoId = setInterval(() => goTo(current + 1), 4000);
  }

  $('#tPrev').addEventListener('click', () => goTo(current - 1));
  $('#tNext').addEventListener('click', () => goTo(current + 1));
  resetAuto();

  // Pause on hover
  track.addEventListener('mouseenter', () => clearInterval(autoId));
  track.addEventListener('mouseleave', resetAuto);
})();

/* ═══════════════════════════════════════════════════
   13. CONTACT FORM
   ═══════════════════════════════════════════════════ */
(function initContactForm() {
  $('#contactForm').addEventListener('submit', async e => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    btn.innerHTML = '<span class="spinner"></span> Sending…';
    btn.disabled = true;
    await sleep(1500);
    btn.innerHTML = 'Send Message 🚀';
    btn.disabled = false;
    e.target.reset();
    const success = $('#formSuccess');
    success.style.display = 'block';
    setTimeout(() => { success.style.display = 'none'; }, 6000);
  });
})();

/* ═══════════════════════════════════════════════════
   14. SMOOTH ACTIVE NAV HIGHLIGHT
   ═══════════════════════════════════════════════════ */
(function initNavHighlight() {
  const sections = $$('section[id]');
  const links = $$('.nav-links a[href^="#"]');

  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const link = links.find(l => l.getAttribute('href') === `#${entry.target.id}`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => io.observe(s));
})();

/* ═══════════════════════════════════════════════════
   15. TILT EFFECT ON SERVICE CARDS
   ═══════════════════════════════════════════════════ */
(function initTilt() {
  $$('[data-tilt]').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width  - 0.5;
      const y = (e.clientY - rect.top)  / rect.height - 0.5;
      card.style.transform = `translateY(-6px) rotateX(${-y * 8}deg) rotateY(${x * 8}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
})();

console.log('%c🍜 Noodles Media', 'color:#9d78ff;font-size:22px;font-weight:900');
console.log('%cBuilding intelligence for everyone.', 'color:#8888aa;font-size:13px');

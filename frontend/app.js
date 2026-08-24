// Scaler Learner Support AI - Enhanced Modern Web UI Interaction Engine

document.addEventListener("DOMContentLoaded", () => {
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const chatMessages = document.getElementById("chat-messages");
  const inspectorBody = document.getElementById("inspector-body");
  const inspectorStatus = document.getElementById("inspector-status");
  const sendBtn = document.getElementById("send-btn");
  const reindexBtn = document.getElementById("reindex-btn");
  const viewTracesBtn = document.getElementById("view-traces-btn");
  const onlineEvalsBtn = document.getElementById("online-evals-btn");
  const tracesModal = document.getElementById("traces-modal");
  const evalsModal = document.getElementById("evals-modal");
  const closeModal = document.getElementById("close-modal");
  const closeEvalsModal = document.getElementById("close-evals-modal");
  const modalTracesList = document.getElementById("modal-traces-list");
  const modalEvalsContent = document.getElementById("modal-evals-content");

  // Handle Chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query");
      queryForm.dispatchEvent(new Event("submit"));
    });
  });

  // Handle Query Submission
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // 1. Add User Message
    appendUserMessage(query);
    queryInput.value = "";
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // 2. Add Assistant Loading Placeholder
    const loadingCard = appendLoadingMessage();
    inspectorStatus.textContent = "Processing...";
    inspectorStatus.style.color = "#f59e0b";

    try {
      const resp = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query, top_k: 4 })
      });

      if (!resp.ok) {
        throw new Error(`Server returned status ${resp.status}`);
      }

      const data = await resp.json();

      // 3. Update Assistant Message
      updateAssistantMessage(loadingCard, data);

      // 4. Update Inspector Panel
      updateInspector(data);

      inspectorStatus.textContent = `${data.latency_ms || 0}ms`;
      inspectorStatus.style.color = "#10b981";

    } catch (err) {
      console.error(err);
      loadingCard.innerHTML = `
        <div class="msg-avatar">⚠️</div>
        <div class="msg-content">
          <div class="msg-header">
            <div class="msg-header-left"><strong>System Alert</strong></div>
          </div>
          <p style="color: #ef4444;">Failed to process query: ${err.message}. Please check if the server is running.</p>
        </div>
      `;
      inspectorStatus.textContent = "Error";
      inspectorStatus.style.color = "#ef4444";
    } finally {
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  });

  function appendUserMessage(text) {
    const card = document.createElement("div");
    card.className = "message-card user-msg";
    card.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-content">
        <div class="msg-header">
          <div class="msg-header-left"><strong>You</strong></div>
          <span style="font-size: 11px; color: #94a3b8;">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        </div>
        <div class="msg-body">${escapeHtml(text)}</div>
      </div>
    `;
    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendLoadingMessage() {
    const card = document.createElement("div");
    card.className = "message-card assistant-msg";
    card.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="msg-header">
          <div class="msg-header-left">
            <strong>Scaler Support AI</strong>
            <span class="badge verified-badge">🔍 Searching Corpus...</span>
          </div>
        </div>
        <div class="msg-body" style="color: #94a3b8; display:flex; align-items:center; gap:8px;">
          <span>Searching policy documents and generating grounded answer...</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return card;
  }

  function parseMarkdownToHtml(markdownText) {
    if (!markdownText) return "";

    let text = escapeHtml(markdownText);

    // Normalize citations: [Chunk 1], 【Chunk 1】, (Chunk 1) -> Citation Badge
    text = text.replace(/(?:\[|\(|【)(?:Source:?\s*)?Chunk[\s\u202f\u00a0]*(\d+)(?:\]|\)|】)/gi, (match, chunkNum) => {
      return `<button class="citation-badge" data-chunk-target="chunk-${chunkNum}" title="Click to highlight source in Inspector">📄 Chunk ${chunkNum}</button>`;
    });

    // Headers (### Header, ## Header, # Header)
    text = text.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    text = text.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^#\s+(.+)$/gm, '<h2>$1</h2>');

    // Bold text (**bold** or __bold__)
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic (*italic* or _italic_)
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code (`code`)
    text = text.replace(/`(.+?)`/g, '<code style="background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:4px; font-family:var(--font-mono); font-size:12px;">$1</code>');

    // Bullet lists (- item, * item, • item)
    const lines = text.split('\n');
    let inList = false;
    let htmlOutput = [];

    for (let line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
        if (!inList) {
          htmlOutput.push('<ul>');
          inList = true;
        }
        htmlOutput.push(`<li>${trimmed.substring(2)}</li>`);
      } else {
        if (inList) {
          htmlOutput.push('</ul>');
          inList = false;
        }
        if (trimmed.length > 0) {
          if (!trimmed.startsWith('<h2') && !trimmed.startsWith('<h3')) {
            htmlOutput.push(`<p>${line}</p>`);
          } else {
            htmlOutput.push(line);
          }
        }
      }
    }
    if (inList) {
      htmlOutput.push('</ul>');
    }

    return htmlOutput.join('\n');
  }

  function updateAssistantMessage(card, data) {
    const traceId = (data.trace && data.trace.trace_id) || "";
    const rawAnswer = data.answer || "";
    const formattedHtml = parseMarkdownToHtml(rawAnswer);

    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = `
        <div class="msg-sources-container">
          <div class="msg-sources-title">
            <span>📑 Attributed Grounding Sources (${data.sources.length})</span>
          </div>
          <div class="msg-sources-grid">
            ${data.sources.map((s, idx) => `
              <button class="source-item-chip" data-chunk-target="chunk-${idx + 1}" title="Click to view section in Inspector">
                <strong style="color:#a5b4fc;">${s.chunk_tag}</strong>
                <span>${escapeHtml(s.doc_title)}</span>
                <span style="font-family:var(--font-mono); font-size:10px; color:#38bdf8;">(${s.similarity_score})</span>
              </button>
            `).join("")}
          </div>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="msg-header">
          <div class="msg-header-left">
            <strong>Scaler Support AI</strong>
            <span class="badge verified-badge">⚡ Grounded • ${data.latency_ms || 0}ms</span>
          </div>
          <span style="font-size: 11px; color: #94a3b8;">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        </div>
        <div class="msg-body">${formattedHtml}</div>
        ${sourcesHtml}
        <div class="msg-actions-row">
          <div class="msg-actions-left">
            <button class="action-btn copy-btn" title="Copy answer text">📋 Copy</button>
          </div>
          <div class="feedback-buttons" data-trace-id="${traceId}">
            <button class="feedback-btn thumbs-up-btn" title="Helpful answer">👍 Helpful</button>
            <button class="feedback-btn thumbs-down-btn" title="Unhelpful or incorrect">👎 Inaccurate</button>
          </div>
        </div>
      </div>
    `;

    // Attach Copy Button Listener
    const copyBtn = card.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(rawAnswer).then(() => {
          copyBtn.innerHTML = "✅ Copied!";
          setTimeout(() => { copyBtn.innerHTML = "📋 Copy"; }, 2000);
        });
      });
    }

    // Attach Citation Click Handlers to highlight inspector chunks
    card.querySelectorAll("[data-chunk-target]").forEach(btn => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-chunk-target");
        highlightInspectorChunk(targetId);
      });
    });

    // Attach feedback listeners
    const feedbackBox = card.querySelector(".feedback-buttons");
    if (feedbackBox && traceId) {
      const upBtn = feedbackBox.querySelector(".thumbs-up-btn");
      const downBtn = feedbackBox.querySelector(".thumbs-down-btn");

      upBtn.addEventListener("click", () => handleFeedback(traceId, 1, upBtn, downBtn));
      downBtn.addEventListener("click", () => handleFeedback(traceId, -1, downBtn, upBtn));
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function highlightInspectorChunk(chunkTag) {
    const chunkNum = chunkTag.replace("chunk-", "");
    const cards = document.querySelectorAll(".chunk-card");
    cards.forEach((c, idx) => {
      if (idx + 1 === parseInt(chunkNum)) {
        c.classList.add("highlighted-chunk");
        c.scrollIntoView({ behavior: "smooth", block: "center" });
        setTimeout(() => {
          c.classList.remove("highlighted-chunk");
        }, 2500);
      }
    });
  }

  async function handleFeedback(traceId, rating, activeBtn, otherBtn) {
    try {
      activeBtn.classList.add(rating === 1 ? "active-positive" : "active-negative");
      activeBtn.disabled = true;
      otherBtn.disabled = true;
      otherBtn.style.opacity = "0.4";

      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, rating: rating })
      });
    } catch (e) {
      console.error("Failed to submit feedback:", e);
    }
  }

  function updateInspector(data) {
    const trace = data.trace || {};
    const sources = data.sources || [];
    const onlineEval = trace.online_eval || {};

    inspectorBody.innerHTML = `
      <div class="metric-grid">
        <div class="metric-box">
          <div class="metric-val">${trace.total_latency_ms || 0} ms</div>
          <div class="metric-lbl">Total Latency</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">${trace.retrieval_latency_ms || 0} ms</div>
          <div class="metric-lbl">Retrieval</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">${trace.generation_latency_ms || 0} ms</div>
          <div class="metric-lbl">Generation</div>
        </div>
      </div>

      <div class="inspector-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
          <strong style="color: #f8fafc; font-size: 12px;">ONLINE EVAL (REAL-TIME)</strong>
          <span style="font-family: var(--font-mono); font-size: 11px; color:#10b981;">${onlineEval.faithfulness || '100%'} Grounded</span>
        </div>
        <div style="font-size: 11.5px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px;">
          <div><strong>Context Relevance:</strong> ${onlineEval.context_relevance || 0.0}</div>
          <div><strong>Citation Density:</strong> ${onlineEval.citation_density || '100%'}</div>
          <div><strong>Flagged for Review:</strong> ${onlineEval.flagged ? '<span style="color:#ef4444;">YES (Needs Audit)</span>' : '<span style="color:#10b981;">NO (High Confidence)</span>'}</div>
        </div>
      </div>

      <div class="inspector-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
          <strong style="color: #f8fafc; font-size: 12px;">TRACE METADATA</strong>
          <span style="font-family: var(--font-mono); font-size: 11px; color:#94a3b8;">${trace.trace_id ? trace.trace_id.substring(0, 8) : 'N/A'}</span>
        </div>
        <div style="font-size: 11.5px; color: #94a3b8;">
          <div><strong>Provider:</strong> ${trace.provider_used || 'Local Engine'}</div>
          <div><strong>Prompt Tokens:</strong> ${trace.prompt_tokens || 0}</div>
          <div><strong>Status:</strong> <span style="color:#10b981;">${trace.status || 'SUCCESS'}</span></div>
        </div>
      </div>

      <div class="inspector-card">
        <strong style="color: #f8fafc; font-size: 12px; display:block; margin-bottom:6px;">TOP RETRIEVED CHUNKS (${sources.length})</strong>
        ${sources.length === 0 ? '<p style="color:#94a3b8; font-size:12px;">No chunks met relevance threshold.</p>' : ''}
        ${sources.map((s, idx) => `
          <div class="chunk-card" id="inspector-chunk-${idx + 1}">
            <div class="chunk-title-row">
              <span>${s.chunk_tag} ${escapeHtml(s.doc_title)}</span>
              <span class="chunk-score">Score: ${s.similarity_score}</span>
            </div>
            <div style="font-size: 11px; color: #cbd5e1; margin-bottom:4px;">${escapeHtml(s.section_path)}</div>
            <div class="chunk-snippet">${escapeHtml(s.snippet.substring(0, 160))}...</div>
          </div>
        `).join("")}
      </div>
    `;
  }

  // Reindex Button
  reindexBtn.addEventListener("click", async () => {
    reindexBtn.disabled = true;
    reindexBtn.innerHTML = "<span>⏳ Indexing...</span>";
    try {
      const resp = await fetch("/ingest", { method: "POST" });
      const res = await resp.json();
      alert(`✅ Re-indexing complete! Successfully indexed ${res.chunks_indexed} chunks.`);
    } catch (e) {
      alert("❌ Re-indexing failed: " + e.message);
    } finally {
      reindexBtn.disabled = false;
      reindexBtn.innerHTML = '<span class="icon">🔄</span> Reindex';
    }
  });

  // Online Evals Modal
  onlineEvalsBtn.addEventListener("click", async () => {
    evalsModal.classList.add("active");
    modalEvalsContent.innerHTML = "<p>Fetching live online evaluation telemetry...</p>";
    try {
      const resp = await fetch("/evals/online");
      const ev = await resp.json();

      modalEvalsContent.innerHTML = `
        <div class="eval-dashboard-grid">
          <div class="eval-metric-card">
            <div class="eval-metric-number">${ev.total_live_queries || 0}</div>
            <div class="eval-metric-label">Total Live Queries</div>
          </div>
          <div class="eval-metric-card">
            <div class="eval-metric-number" style="color:#10b981;">${ev.avg_faithfulness || 0}%</div>
            <div class="eval-metric-label">Faithfulness Score</div>
          </div>
          <div class="eval-metric-card">
            <div class="eval-metric-number" style="color:#38bdf8;">${ev.user_satisfaction_pct || 100}%</div>
            <div class="eval-metric-label">Learner Satisfaction (👍 ${ev.total_thumbs_up} / 👎 ${ev.total_thumbs_down})</div>
          </div>
          <div class="eval-metric-card">
            <div class="eval-metric-number" style="color:#f59e0b;">${ev.latency_p50_ms || 0} ms</div>
            <div class="eval-metric-label">P50 Latency (P95: ${ev.latency_p95_ms || 0}ms)</div>
          </div>
        </div>

        <h3 style="color:#f8fafc; font-size:14px; margin-bottom:12px;">🚨 Flagged Queries for Review (${ev.flagged_for_review_count || 0})</h3>
        ${(!ev.recent_flagged_queries || ev.recent_flagged_queries.length === 0) ? '<p style="color:#94a3b8;">No anomalies or low-confidence queries flagged.</p>' : `
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
            <thead>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #94a3b8;">
                <th style="padding: 8px;">Time</th>
                <th style="padding: 8px;">Query</th>
                <th style="padding: 8px;">Reason</th>
                <th style="padding: 8px;">User Rating</th>
              </tr>
            </thead>
            <tbody>
              ${ev.recent_flagged_queries.map(q => `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                  <td style="padding: 8px; color: #64748b;">${new Date(q.timestamp).toLocaleTimeString()}</td>
                  <td style="padding: 8px; color: #f8fafc;">${escapeHtml(q.query)}</td>
                  <td style="padding: 8px; color: #f43f5e;">${q.review_reason}</td>
                  <td style="padding: 8px;">${q.user_rating === 1 ? '👍' : q.user_rating === -1 ? '👎' : 'None'}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `}
      `;
    } catch (e) {
      modalEvalsContent.innerHTML = `<p style="color:red;">Error loading online evaluations: ${e.message}</p>`;
    }
  });

  closeEvalsModal.addEventListener("click", () => {
    evalsModal.classList.remove("active");
  });

  evalsModal.addEventListener("click", (e) => {
    if (e.target === evalsModal) evalsModal.classList.remove("active");
  });

  // Traces Modal
  viewTracesBtn.addEventListener("click", async () => {
    tracesModal.classList.add("active");
    modalTracesList.innerHTML = "<p>Fetching traces from SQLite...</p>";
    try {
      const resp = await fetch("/traces?limit=15");
      const data = await resp.json();
      const traces = data.traces || [];

      if (traces.length === 0) {
        modalTracesList.innerHTML = "<p>No query traces recorded yet.</p>";
        return;
      }

      modalTracesList.innerHTML = `
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
          <thead>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #94a3b8;">
              <th style="padding: 8px;">Time</th>
              <th style="padding: 8px;">Query</th>
              <th style="padding: 8px;">Latency</th>
              <th style="padding: 8px;">Status</th>
              <th style="padding: 8px;">Chunks</th>
            </tr>
          </thead>
          <tbody>
            ${traces.map(t => `
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 8px; color: #64748b;">${new Date(t.timestamp).toLocaleTimeString()}</td>
                <td style="padding: 8px; color: #f8fafc; font-weight: 500;">${escapeHtml(t.query)}</td>
                <td style="padding: 8px; color: #10b981;">${t.total_latency_ms} ms</td>
                <td style="padding: 8px;"><span style="color: ${t.status === 'SUCCESS' ? '#10b981' : '#f59e0b'}">${t.status}</span></td>
                <td style="padding: 8px; color: #38bdf8;">${t.retrieved_chunks ? t.retrieved_chunks.length : 0}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    } catch (e) {
      modalTracesList.innerHTML = `<p style="color:red;">Error loading traces: ${e.message}</p>`;
    }
  });

  closeModal.addEventListener("click", () => {
    tracesModal.classList.remove("active");
  });

  tracesModal.addEventListener("click", (e) => {
    if (e.target === tracesModal) tracesModal.classList.remove("active");
  });

  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

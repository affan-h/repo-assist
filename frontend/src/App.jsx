import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

const PIPELINE_STAGES = [
  { key: 'QUEUED', label: '1. Queued in Pipeline', desc: 'Request received and scheduled in BackgroundTasks' },
  { key: 'CLONED', label: '2. Repository Cloned', desc: 'Full repository history cloned to local workspace' },
  { key: 'PARSED', label: '3. AST Symbols & Imports Parsed', desc: 'Tree-sitter extracted files, classes, methods, and imports' },
  { key: 'GRAPH_BUILT', label: '4. Call Graph & PageRank Built', desc: 'Resolved CALLS/EXTENDS edges and calculated symbol PageRank' },
  { key: 'HISTORY_ATTACHED', label: '5. History & Docs Attached', desc: 'Mined PyDriller commit provenance and chunked docs' },
  { key: 'RISK_SCORED', label: '6. Churn, Complexity & Centrality', desc: 'Multi-factor file risk scores computed across repository' },
  { key: 'READY', label: '7. Code Graph Ready', desc: 'Ingestion complete; repository unlocked for grounded querying' },
];

const STAGE_ORDER = ['QUEUED', 'CLONED', 'PARSED', 'GRAPH_BUILT', 'HISTORY_ATTACHED', 'RISK_SCORED', 'READY'];

const SAMPLE_QUESTIONS = {
  httpx: [
    "What does the Client class do?",
    "Where is the AsyncClient class defined?",
    "Trace the call chain from request() to send() in httpx",
  ],
  got: [
    "Why does got default to 2 retries?",
    "Where is the create() function defined?",
  ],
  requests: [
    "What does the Session class do?",
    "Where is the Response class defined?",
  ],
  itsdangerous: [
    "What does the Signer class do?",
    "Where is URLSafeSerializer defined?",
  ],
};

export default function App() {
  const [view, setView] = useState('input'); // 'input' | 'progress' | 'chat'
  const [reposList, setReposList] = useState([]);
  const [inputUrl, setInputUrl] = useState('');
  const [activeRepoId, setActiveRepoId] = useState('');
  const [repoStatus, setRepoStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [questionInput, setQuestionInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const messagesEndRef = useRef(null);

  // Fetch repos list on initial load and view changes
  const fetchReposList = async () => {
    try {
      const res = await fetch(`${API_BASE}/repos`);
      if (res.ok) {
        const data = await res.json();
        setReposList(data);
      }
    } catch (err) {
      console.error('Failed to fetch repositories list:', err);
    }
  };

  useEffect(() => {
    fetchReposList();
  }, [view]);

  // Scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isQuerying]);

  // Polling for repository status in progress view
  useEffect(() => {
    if (view !== 'progress' || !activeRepoId) return;

    let isMounted = true;

    const pollStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/repos/${activeRepoId}/status`);
        if (!res.ok) {
          throw new Error(`Status check failed: HTTP ${res.status}`);
        }
        const data = await res.json();
        if (isMounted) {
          setRepoStatus(data);
          if (data.status === 'READY') {
            fetchReposList();
          }
        }
      } catch (err) {
        if (isMounted) {
          console.error('Polling error:', err);
        }
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 2000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [view, activeRepoId]);

  // Handler: Ingest a new repository
  const handleIngest = async (e) => {
    e?.preventDefault();
    const url = inputUrl.trim();
    if (!url) return;

    setIsSubmitting(true);
    setErrorMsg(null);

    try {
      const res = await fetch(`${API_BASE}/repos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Ingestion request failed: ${res.statusText}`);
      }

      const data = await res.json();
      setActiveRepoId(data.repo_id);
      setRepoStatus(data);
      setMessages([]);
      setView('progress');
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handler: Select an already ingested repository
  const handleSelectExistingRepo = (repo) => {
    setActiveRepoId(repo.repo_id);
    setRepoStatus(repo);
    setMessages([]);
    if (repo.status === 'READY') {
      setView('chat');
    } else {
      setView('progress');
    }
  };

  // Handler: Submit Question in Chat
  const handleAskQuestion = async (qText) => {
    const question = (qText || questionInput).trim();
    if (!question || isQuerying || !activeRepoId) return;

    const userMessageId = Date.now();
    setQuestionInput('');
    setIsQuerying(true);
    setErrorMsg(null);

    // Append optimistic user question
    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        question,
        answer: null,
        loading: true,
      },
    ]);

    try {
      const res = await fetch(`${API_BASE}/repos/${activeRepoId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Query failed: HTTP ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === userMessageId
            ? {
                ...msg,
                loading: false,
                answer: data.answer,
                citation_source_id: data.citation_source_id,
                abstained: data.abstained,
                abstain_reason: data.abstain_reason,
                category: data.category,
                model_used: data.model_used,
              }
            : msg
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === userMessageId
            ? {
                ...msg,
                loading: false,
                error: err.message,
              }
            : msg
        )
      );
    } finally {
      setIsQuerying(false);
    }
  };

  // Helpers for Pipeline Stepper
  const currentStageIndex = repoStatus
    ? STAGE_ORDER.indexOf(repoStatus.status)
    : -1;
  const isFailed = repoStatus?.status === 'FAILED';
  const isReady = repoStatus?.status === 'READY';

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <h1>repo-assist</h1>
          </div>
          <span className="brand-badge">Grounded Code Intelligence</span>
        </div>
        <div className="nav-status">
          <div className="status-dot"></div>
          <span>Backend API Connected (127.0.0.1:8000)</span>
        </div>
      </header>

      {/* VIEW 1: REPO INPUT & PICKER */}
      {view === 'input' && (
        <div className="view-card">
          <h2>Ingest a Repository</h2>
          <p className="subtitle">
            Provide any public GitHub repository URL to build structural graphs, mine commit provenance, and calculate multi-factor risk scores.
          </p>

          <form onSubmit={handleIngest} className="ingest-form">
            <input
              type="text"
              className="input-field"
              placeholder="e.g. https://github.com/pallets/click or owner/repo"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              disabled={isSubmitting}
            />
            <button type="submit" className="btn-primary" disabled={isSubmitting || !inputUrl.trim()}>
              {isSubmitting ? (
                <>
                  <span className="spinner"></span> Ingesting...
                </>
              ) : (
                'Ingest Repository'
              )}
            </button>
          </form>

          {errorMsg && (
            <div className="alert-box alert-danger">
              <span>⚠️</span>
              <div>{errorMsg}</div>
            </div>
          )}

          {/* Available Repositories List */}
          <div className="repos-section">
            <h3>Or Query an Ingested Repository</h3>
            {reposList.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                No ingested repositories found. Ingest your first repository above!
              </p>
            ) : (
              <div className="repos-grid">
                {reposList.map((r) => (
                  <div
                    key={r.repo_id}
                    className="repo-card"
                    onClick={() => handleSelectExistingRepo(r)}
                  >
                    <div>
                      <div className="repo-card-header">
                        <span className="repo-name">{r.repo_id}</span>
                        <span
                          className={`badge ${
                            r.status === 'READY'
                              ? 'badge-ready'
                              : r.status === 'FAILED'
                              ? 'badge-failed'
                              : 'badge-pending'
                          }`}
                        >
                          {r.status}
                        </span>
                      </div>
                      <div className="repo-card-url">{r.url || `local/${r.repo_id}`}</div>
                    </div>
                    <div style={{ marginTop: '0.75rem', display: 'flex', justifyContent: 'flex-end' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: 500 }}>
                        {r.status === 'READY' ? 'Open Chat →' : 'View Progress →'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* VIEW 2: INGESTION PIPELINE PROGRESS */}
      {view === 'progress' && (
        <div className="view-card">
          <div className="stepper-header">
            <div>
              <h2>Ingesting: {activeRepoId}</h2>
              <p className="subtitle" style={{ marginBottom: 0 }}>
                {repoStatus?.url || activeRepoId}
              </p>
            </div>
            <button className="btn-secondary" onClick={() => setView('input')}>
              ← Back to Repos
            </button>
          </div>

          {/* Stepper Steps */}
          <div className="stepper-container">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isCompleted = isReady || (!isFailed && currentStageIndex > idx);
              const isActive = !isFailed && !isReady && currentStageIndex === idx;
              const isPending = !isCompleted && !isActive;

              let itemClass = 'step-item ';
              if (isCompleted) itemClass += 'completed';
              else if (isActive) itemClass += 'active';
              else itemClass += 'pending';

              return (
                <div key={stage.key} className={itemClass}>
                  <div className="step-icon-wrapper">
                    {isCompleted ? '✓' : idx + 1}
                  </div>
                  <div className="step-details">
                    <h4>{stage.label}</h4>
                    <p>{stage.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Error message */}
          {isFailed && (
            <div className="alert-box alert-danger">
              <span style={{ fontSize: '1.2rem' }}>⚠️</span>
              <div>
                <strong>Ingestion Failed:</strong>
                <p style={{ marginTop: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                  {repoStatus?.error_message || 'An unknown error occurred.'}
                </p>
              </div>
            </div>
          )}

          {/* Ready Banner */}
          {isReady && (
            <div className="alert-box alert-success" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>Repository Ingestion Complete!</strong>
                <p style={{ fontSize: '0.85rem' }}>All phases finished. The code graph and history are ready.</p>
              </div>
              <button className="btn-primary" onClick={() => setView('chat')}>
                Start Asking Questions →
              </button>
            </div>
          )}
        </div>
      )}

      {/* VIEW 3: CHAT & QUERY VIEW */}
      {view === 'chat' && (
        <div className="view-card">
          <div className="chat-header">
            <div className="active-repo-badge">
              <h2>{activeRepoId}</h2>
              <span className="badge badge-ready">READY</span>
            </div>
            <button className="btn-secondary" onClick={() => setView('input')}>
              Switch Repository
            </button>
          </div>

          {/* Messages Stream */}
          <div className="messages-list">
            {messages.length === 0 ? (
              <div className="empty-chat">
                <div className="empty-icon">💬</div>
                <h3>Ask anything about {activeRepoId}</h3>
                <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                  Every answer is strictly grounded with code symbol, commit, or doc citations.
                </p>

                {SAMPLE_QUESTIONS[activeRepoId] && (
                  <div className="suggested-questions">
                    {SAMPLE_QUESTIONS[activeRepoId].map((sq, i) => (
                      <button
                        key={i}
                        className="sample-chip"
                        onClick={() => handleAskQuestion(sq)}
                      >
                        {sq}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              messages.map((m) => (
                <div key={m.id} className="message-pair">
                  <div className="user-question">{m.question}</div>

                  <div className={`assistant-answer-card ${m.abstained ? 'abstain-card' : ''}`}>
                    {m.loading ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                        <span className="spinner"></span> Routing evidence and synthesizing answer...
                      </div>
                    ) : m.error ? (
                      <div style={{ color: 'var(--error-text)' }}>
                        <strong>Error:</strong> {m.error}
                      </div>
                    ) : (
                      <>
                        {/* Distinct Abstention Banner */}
                        {m.abstained && (
                          <div className="abstain-banner">
                            <span>🛡️</span>
                            <span>Abstained from Hallucination: {m.abstain_reason || 'Evidence insufficient to support a confident answer.'}</span>
                          </div>
                        )}

                        <div className="answer-body">{m.answer}</div>

                        {/* Prominent Citation and Metadata */}
                        <div className="citation-container">
                          {m.citation_source_id && (
                            <span className="citation-pill">
                              📌 Source: {m.citation_source_id}
                            </span>
                          )}
                          {m.category && (
                            <span className="meta-pill">Category: {m.category}</span>
                          )}
                          {m.model_used && (
                            <span className="meta-pill">Model: {m.model_used}</span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Question Input Form */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleAskQuestion();
            }}
            className="query-form"
          >
            <input
              type="text"
              className="query-input"
              placeholder={`Ask a question about ${activeRepoId}...`}
              value={questionInput}
              onChange={(e) => setQuestionInput(e.target.value)}
              disabled={isQuerying}
            />
            <button
              type="submit"
              className="btn-primary"
              disabled={isQuerying || !questionInput.trim()}
            >
              {isQuerying ? (
                <>
                  <span className="spinner"></span> Asking...
                </>
              ) : (
                'Ask'
              )}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

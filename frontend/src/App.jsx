import { useCallback, useEffect, useRef, useState } from 'react';
import { askStream, ingestRepo, listRepos } from './api.js';
import Sidebar from './components/Sidebar.jsx';
import Turn from './components/Turn.jsx';

const SAMPLE_QUESTIONS = {
  httpx: [
    'What does Client._transport_for_url return when no mount pattern matches?',
    'Trace the call chain from Client.get() to the point where httpcore is called.',
    'Why was sync support dropped in httpx 0.8?',
    'Why did the authors choose 5 seconds as the default timeout?',
  ],
  got: [
    'How does got decide how long to wait before retrying a request?',
    'Where does got merge hooks when got.extend() is called?',
    'Why did got v10 change the json option?',
    'What hooks does got expose and when does each run?',
  ],
  default: [
    'What are the main entry points of this library?',
    'Where is configuration loaded and validated?',
    'Why was the most recent major version released?',
  ],
};

export default function App() {
  const [repos, setRepos] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [mode, setMode] = useState('agent');
  const [turnsByRepo, setTurnsByRepo] = useState({});
  const [question, setQuestion] = useState('');
  const [loadError, setLoadError] = useState(null);
  const bottomRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listRepos();
      setRepos(data);
      setLoadError(null);
      setSelectedId((current) => current ?? data.find((r) => Object.keys(r.stats).length)?.id ?? null);
    } catch (err) {
      setLoadError(`Cannot reach the API: ${err.message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while any repository is being indexed.
  const indexing = repos.some((r) => r.status === 'queued' || r.status === 'ingesting');
  useEffect(() => {
    if (!indexing) return undefined;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [indexing, refresh]);

  const repo = repos.find((r) => r.id === selectedId);
  const turns = turnsByRepo[selectedId] ?? [];
  const busy = turns.some((t) => t.status === 'running');

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns.length]);

  function updateTurn(repoId, turnId, update) {
    setTurnsByRepo((all) => ({
      ...all,
      [repoId]: (all[repoId] ?? []).map((t) => (t.id === turnId ? update(t) : t)),
    }));
  }

  async function ask(text) {
    const q = text.trim();
    if (!q || !repo || busy) return;
    const repoId = repo.id;
    const turnId = crypto.randomUUID();
    setQuestion('');
    setTurnsByRepo((all) => ({
      ...all,
      [repoId]: [...(all[repoId] ?? []), { id: turnId, question: q, mode, status: 'running', steps: [] }],
    }));

    const onEvent = (event) => {
      if (event.type === 'tool_call') {
        updateTurn(repoId, turnId, (t) => ({ ...t, steps: [...t.steps, { ...event, done: false }] }));
      } else if (event.type === 'tool_result') {
        updateTurn(repoId, turnId, (t) => ({
          ...t,
          steps: t.steps.map((s) => (s.id === event.id ? { ...s, done: true, summary: event.summary } : s)),
        }));
      } else if (event.type === 'answer') {
        updateTurn(repoId, turnId, (t) => ({ ...t, status: 'done', result: event.result }));
      } else if (event.type === 'error') {
        updateTurn(repoId, turnId, (t) => ({ ...t, status: 'error', error: event.message }));
      }
    };

    try {
      await askStream(repoId, q, mode, onEvent);
    } catch (err) {
      updateTurn(repoId, turnId, (t) => ({ ...t, status: 'error', error: err.message }));
    }
  }

  async function handleIngest(url, ref) {
    const created = await ingestRepo(url, ref);
    await refresh();
    return created;
  }

  const samples = SAMPLE_QUESTIONS[repo?.name] ?? SAMPLE_QUESTIONS.default;

  return (
    <div className="layout">
      <Sidebar repos={repos} selectedId={selectedId} onSelect={setSelectedId} onIngest={handleIngest} />

      <main className="main">
        {loadError && <div className="banner error-text">{loadError}</div>}

        {!repo ? (
          <div className="empty">
            <h1>Ask grounded questions about a codebase</h1>
            <p className="muted">
              Index a GitHub repository, then ask what the code does, where something lives, how it works, or why it
              was designed that way. Every answer cites the exact lines, commits, issues or PRs it is based on.
            </p>
          </div>
        ) : (
          <>
            <header className="main-header">
              <div>
                <h1>
                  {repo.owner}/{repo.name}
                </h1>
                <a
                  className="muted small"
                  href={`${repo.url}/tree/${repo.commit_sha}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  indexed at {repo.commit_sha?.slice(0, 10)}
                </a>
              </div>
              <div className="mode-toggle" role="radiogroup" aria-label="Answering mode">
                {[
                  ['agent', 'Agent', 'Plans its own searches, reads code and history, then answers'],
                  ['single_shot', 'Single-shot', 'Classic RAG: one retrieval, one LLM call'],
                ].map(([value, label, hint]) => (
                  <button
                    key={value}
                    role="radio"
                    aria-checked={mode === value}
                    className={mode === value ? 'active' : ''}
                    onClick={() => setMode(value)}
                    title={hint}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </header>

            <section className="conversation">
              {turns.length === 0 && (
                <div className="samples">
                  <p className="muted">Try one of these, or ask your own. Each question is answered independently.</p>
                  {samples.map((s) => (
                    <button key={s} className="sample" onClick={() => ask(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {turns.map((t) => (
                <Turn key={t.id} turn={t} />
              ))}
              <div ref={bottomRef} />
            </section>

            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                ask(question);
              }}
            >
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    ask(question);
                  }
                }}
                placeholder={`Ask about ${repo.name}… (Enter to send)`}
                rows={2}
                aria-label="Question"
              />
              <button type="submit" disabled={busy || question.trim().length < 3}>
                {busy ? 'Working…' : 'Ask'}
              </button>
            </form>
          </>
        )}
      </main>
    </div>
  );
}

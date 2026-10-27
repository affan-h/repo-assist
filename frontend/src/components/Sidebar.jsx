import { useState } from 'react';

const STATUS_LABEL = { ready: 'Ready', ingesting: 'Indexing', queued: 'Queued', failed: 'Failed' };

function RepoItem({ repo, selected, onSelect }) {
  const { stats } = repo;
  const history = ['commit', 'issue', 'pr', 'discussion', 'release'].reduce((n, k) => n + (stats[k] || 0), 0);
  const usable = Object.keys(stats).length > 0;
  return (
    <li>
      <button
        className={`repo-item ${selected ? 'selected' : ''}`}
        onClick={() => onSelect(repo.id)}
        disabled={!usable}
        title={usable ? undefined : 'Not indexed yet'}
      >
        <span className="repo-item-top">
          <span className="repo-name">{repo.owner}/{repo.name}</span>
          <span className={`status status-${repo.status}`}>{STATUS_LABEL[repo.status] ?? repo.status}</span>
        </span>
        {repo.status === 'ingesting' || repo.status === 'queued' ? (
          <span className="repo-meta">{repo.stage}</span>
        ) : repo.status === 'failed' ? (
          <span className="repo-meta error-text">{repo.error}</span>
        ) : (
          <span className="repo-meta">
            <code>{repo.commit_sha?.slice(0, 7)}</code> · {(stats.code || 0) + (stats.doc || 0)} code/doc chunks ·{' '}
            {history} history items
          </span>
        )}
        {stats.warnings?.length > 0 && (
          <span className="repo-meta warn-text" title={stats.warnings.join('\n')}>
            ⚠ {stats.warnings.length === 1 ? 'Partial index (hover for details)' : `${stats.warnings.length} warnings`}
          </span>
        )}
      </button>
    </li>
  );
}

export default function Sidebar({ repos, selectedId, onSelect, onIngest }) {
  const [url, setUrl] = useState('');
  const [ref, setRef] = useState('');
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await onIngest(url.trim(), ref.trim());
      setUrl('');
      setRef('');
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden>{'</>'}</span>
        <div>
          <div className="brand-name">repo-assist</div>
          <div className="brand-tag">Cited answers about any GitHub repo</div>
        </div>
      </div>

      <h2 className="section-label">Repositories</h2>
      {repos.length === 0 ? (
        <p className="muted small">Nothing indexed yet. Add a repository below.</p>
      ) : (
        <ul className="repo-list">
          {repos.map((r) => (
            <RepoItem key={r.id} repo={r} selected={r.id === selectedId} onSelect={onSelect} />
          ))}
        </ul>
      )}

      <form className="ingest-form" onSubmit={submit}>
        <h2 className="section-label">Add a repository</h2>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="owner/name or GitHub URL"
          aria-label="Repository"
          required
        />
        <input
          value={ref}
          onChange={(e) => setRef(e.target.value)}
          placeholder="commit or tag (optional)"
          aria-label="Git ref"
        />
        <button type="submit" disabled={submitting || !url.trim()}>
          {submitting ? 'Queuing…' : 'Index repository'}
        </button>
        {error && <p className="error-text small">{error}</p>}
        <p className="muted small">
          Indexes code, docs, commits, issues, PRs, discussions and releases. Small repos take a few minutes on CPU.
        </p>
      </form>
    </aside>
  );
}

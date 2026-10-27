import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const TOOL_LABEL = {
  search_code: 'Search code & docs',
  search_history: 'Search history',
  read_thread: 'Read thread',
  find_symbol: 'Find symbol',
  read_file: 'Read file',
  grep: 'Grep',
  list_files: 'List files',
  search_commits: 'Search commits',
  code_history: 'Trace code history',
};

function describeArgs(tool, args = {}) {
  switch (tool) {
    case 'read_file':
      return `${args.path}:${args.start_line ?? 1}-${args.end_line ?? ''}`;
    case 'grep':
      return `/${args.pattern}/${args.path_glob ? ` in ${args.path_glob}` : ''}`;
    case 'search_commits':
      return `/${args.pattern}/`;
    case 'code_history':
      return `${args.path}:${args.start_line}-${args.end_line}`;
    case 'list_files':
      return args.path_glob;
    case 'find_symbol':
      return args.name;
    case 'read_thread':
      return args.offset ? `${args.ref} (from char ${args.offset})` : args.ref;
    default:
      return args.query;
  }
}

function Trace({ steps, running }) {
  if (steps.length === 0 && !running) return null;
  return (
    <details className="trace" open={running}>
      <summary>
        {running ? 'Investigating…' : `Investigation: ${steps.length} tool call${steps.length === 1 ? '' : 's'}`}
      </summary>
      <ol>
        {steps.map((s) => (
          <li key={s.id} className={s.done ? 'done' : 'pending'}>
            <span className="tool">{TOOL_LABEL[s.tool] ?? s.tool}</span>
            <code className="tool-args">{describeArgs(s.tool, s.args)}</code>
            <span className="tool-result">{s.done ? s.summary : '…'}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

/** Turn citation markers into links: both "[2]" and "[httpx/_client.py:10-20]" forms. */
function linkCitations(text, citations) {
  const index = new Map(citations.map((c, i) => [c.ref.toLowerCase(), i + 1]));
  return text.replace(/\[([^\]\n]+)\](?!\()/g, (match, inner) => {
    const parts = inner.split(/,\s*/);
    const numbers = parts.map((p) => {
      if (/^\d+$/.test(p) && Number(p) >= 1 && Number(p) <= citations.length) return Number(p);
      return index.get(p.trim().toLowerCase().replace(/^`|`$/g, ''));
    });
    if (numbers.some((n) => n === undefined)) return match;
    return numbers.map((n) => `[${n}](#cite-${n})`).join('');
  });
}

function Sources({ citations, highlighted }) {
  if (!citations.length) return null;
  return (
    <div className="sources">
      <h3>Sources</h3>
      <ol>
        {citations.map((c, i) => (
          <li key={c.ref} id={`source-${i + 1}`} className={highlighted === i + 1 ? 'highlight' : ''}>
            <details>
              <summary>
                <span className="source-num">{i + 1}</span>
                <span className={`kind kind-${c.kind}`}>{c.kind}</span>
                <code className="source-ref">{c.ref}</code>
                {c.title && <span className="source-title">{c.title}</span>}
                <a href={c.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                  GitHub ↗
                </a>
              </summary>
              <pre className={`snippet ${c.kind === 'code' ? '' : 'prose'}`}>{c.snippet}</pre>
            </details>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function Turn({ turn }) {
  const [highlighted, setHighlighted] = useState(null);
  const { result } = turn;
  const running = turn.status === 'running';

  const markdownComponents = {
    a: ({ href, children }) => {
      if (href?.startsWith('#cite-')) {
        const n = Number(href.slice(6));
        return (
          <button
            className="cite"
            onClick={() => {
              setHighlighted(n);
              document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }}
          >
            {children}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <article className="turn">
      <div className="question">
        <span className="mode-tag">{turn.mode === 'agent' ? 'Agent' : 'Single-shot'}</span>
        {turn.question}
      </div>

      <Trace steps={turn.steps} running={running} />

      {turn.status === 'error' && <div className="answer error-text">{turn.error}</div>}

      {result && (
        <div className="answer">
          {result.abstained && <div className="abstained">No grounded answer in the indexed evidence</div>}
          <ReactMarkdown components={markdownComponents}>
            {linkCitations(result.answer, result.citations)}
          </ReactMarkdown>
          {result.quotes?.length > 0 && (
            <div className="quotes">
              <h3>Stated rationale (verified verbatim)</h3>
              {result.quotes.map((q) => {
                const n = result.citations.findIndex((c) => c.ref.toLowerCase() === q.ref.toLowerCase()) + 1;
                return (
                  <blockquote key={q.ref + q.quote}>
                    “{q.quote}”
                    <span className="quote-ref">
                      {' '}
                      — {n > 0 ? <a href={result.citations[n - 1].url} target="_blank" rel="noreferrer">{q.ref}</a> : q.ref}
                    </span>
                  </blockquote>
                );
              })}
            </div>
          )}
          <Sources citations={result.citations} highlighted={highlighted} />
          <div className="metrics">
            <span>{result.latency_s.toFixed(1)}s</span>
            <span>{result.usage.requests ?? 0} LLM calls</span>
            <span>{result.trace.length} tool calls</span>
            <span>{((result.usage.input_tokens ?? 0) + (result.usage.output_tokens ?? 0)).toLocaleString()} tokens</span>
            {result.error && <span className="error-text">{result.error}</span>}
          </div>
        </div>
      )}
    </article>
  );
}

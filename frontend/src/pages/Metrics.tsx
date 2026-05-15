import { useState, useEffect } from 'react';
import { api } from '../api';
import type { SystemMetrics, MetricHistory } from '../types';

export default function Metrics() {
  const [data, setData] = useState<SystemMetrics | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getMetrics()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /><div>Cargando métricas...</div></div>;
  if (error) return <div className="error-msg">Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="metrics-page">
      <h2 className="page-title">Model Metrics</h2>
      <p className="page-subtitle">Real-time performance tracking and prediction audit log.</p>

      <div className="mini-stats" style={{ marginBottom: '2rem' }}>
        <div className="card mini-stat">
          <div className="mini-stat-label">Total Predictions</div>
          <div className="mini-stat-val">{data.metrics.total_predictions}</div>
        </div>
        <div className="card mini-stat">
          <div className="mini-stat-label">Resolved Games</div>
          <div className="mini-stat-val">{data.metrics.resolved}</div>
        </div>
        <div className="card mini-stat">
          <div className="mini-stat-label">Hit Rate Core</div>
          <div className="mini-stat-val pos">{data.metrics.hit_rate.toFixed(1)}%</div>
        </div>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <h3 className="card-title" style={{ padding: '1.5rem 1.5rem 0' }}>Recent Audits</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Matchup</th>
                <th>Pick</th>
                <th className="num">Prob</th>
                <th>Result</th>
                <th className="num">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.history.map((row: MetricHistory) => (
                <tr key={row.id}>
                  <td>{new Date(row.date).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</td>
                  <td>{row.team1} vs {row.team2}</td>
                  <td>{row.predicted}</td>
                  <td className="num">{(row.prob * 100).toFixed(1)}%</td>
                  <td>{row.actual_winner || 'Pending'}</td>
                  <td className="num">
                    {row.correct === null ? (
                      <span className="badge" style={{ background: 'rgba(255,255,255,0.1)', color: '#aaa' }}>WAITING</span>
                    ) : row.correct ? (
                      <span className="badge" style={{ background: 'rgba(34,197,94,0.15)', color: '#4ade80' }}>WIN</span>
                    ) : (
                      <span className="badge" style={{ background: 'rgba(232,70,42,0.15)', color: 'var(--accent2)' }}>LOSS</span>
                    )}
                  </td>
                </tr>
              ))}
              {data.history.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted)' }}>
                    No prediction history available yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

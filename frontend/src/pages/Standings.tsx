import { useState, useEffect } from 'react';
import { api } from '../api';
import type { Team } from '../types';

type SortKey = 'net_expected' | 'expectedScore' | 'win_streak_5' | 'fg_pct';

export default function Standings() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);
  const [conf, setConf] = useState('all');
  const [sort, setSort] = useState<SortKey>('net_expected');

  useEffect(() => {
    setLoading(true);
    api.getStandings(conf)
      .then(d => { setTeams(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [conf]);

  const sorted = [...teams].sort((a, b) => (b[sort] as number) - (a[sort] as number));

  const sortLabel: Record<SortKey, string> = {
    net_expected: 'Net Score',
    expectedScore: 'Pts Esperados',
    win_streak_5: 'Win Streak',
    fg_pct: 'FG%',
  };

  return (
    <div>
      <h1 className="page-title">STANDINGS</h1>
      <p className="page-subtitle">Clasificación basada en estadísticas rolling de los últimos 5 partidos.</p>

      <div className="tabs">
        {([['all', 'Todos'], ['West', 'Oeste'], ['East', 'Este']] as [string, string][]).map(([v, l]) => (
          <button key={v} className={`tab${conf === v ? ' active' : ''}`} onClick={() => setConf(v)}>{l}</button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '1rem' }}>
        <span style={{ fontSize: 12, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Ordenar:</span>
        {(Object.keys(sortLabel) as SortKey[]).map(k => (
          <div key={k} className={`radio-pill${sort === k ? ' selected' : ''}`} style={{ flex: 'none', padding: '4px 12px', fontSize: 12 }} onClick={() => setSort(k)}>
            {sortLabel[k]}
          </div>
        ))}
      </div>

      {loading ? (
        <div className="loading"><div className="spinner" /><br />Cargando…</div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Equipo</th>
                  <th>Conf</th>
                  <th className="num" onClick={() => setSort('expectedScore')}>Pts Esp.</th>
                  <th className="num" onClick={() => setSort('net_expected')}>Net</th>
                  <th className="num" onClick={() => setSort('win_streak_5')}>Streak</th>
                  <th className="num" onClick={() => setSort('fg_pct')}>FG%</th>
                  <th className="num">Handicap</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((t, i) => (
                  <tr key={t.team_id}>
                    <td>
                      <span className={`rank-badge${i === 0 ? ' gold' : i === 1 ? ' silver' : i === 2 ? ' bronze' : ''}`}>
                        {i + 1}
                      </span>
                    </td>
                    <td><strong>{t.name}</strong></td>
                    <td><span className={`badge badge-${t.conf.toLowerCase()}`}>{t.conf}</span></td>
                    <td className="num">{t.expectedScore}</td>
                    <td className="num">
                      <span className={t.net_expected >= 0 ? 'pos' : 'neg'}>
                        {t.net_expected >= 0 ? '+' : ''}{t.net_expected}
                      </span>
                    </td>
                    <td className="num">{t.win_streak_5}</td>
                    <td className="num">{t.fg_pct}%</td>
                    <td className="num">
                      <span className={t.RealHandicap >= 0 ? 'pos' : 'neg'}>
                        {t.RealHandicap >= 0 ? '+' : ''}{t.RealHandicap}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

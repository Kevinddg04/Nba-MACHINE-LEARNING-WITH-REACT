import { useState, useEffect } from 'react';

interface Player {
  name: string;
  team: string;
  ppg: number;
  rpg: number;
  apg: number;
  spg: number;
  fg_pct: number;
  fg3_pct: number;
}

const MOCK_PLAYERS: Player[] = [
  { name: 'LeBron James',     team: 'LAL', ppg: 25.7, rpg: 7.3, apg: 8.3, spg: 1.3, fg_pct: 54.0, fg3_pct: 41.0 },
  { name: 'Stephen Curry',    team: 'GSW', ppg: 26.4, rpg: 4.5, apg: 5.1, spg: 0.9, fg_pct: 45.2, fg3_pct: 40.8 },
  { name: 'Jayson Tatum',     team: 'BOS', ppg: 26.9, rpg: 8.1, apg: 4.9, spg: 1.1, fg_pct: 47.1, fg3_pct: 37.2 },
  { name: 'Nikola Jokic',     team: 'DEN', ppg: 26.4, rpg: 12.4, apg: 9.0, spg: 1.4, fg_pct: 58.3, fg3_pct: 35.8 },
  { name: 'Giannis',          team: 'MIL', ppg: 30.4, rpg: 11.5, apg: 6.5, spg: 1.2, fg_pct: 61.1, fg3_pct: 27.5 },
  { name: 'Shai Gilgeous',    team: 'OKC', ppg: 30.1, rpg: 5.5, apg: 6.2, spg: 2.0, fg_pct: 53.5, fg3_pct: 35.5 },
  { name: 'Donovan Mitchell', team: 'CLE', ppg: 26.6, rpg: 4.9, apg: 4.9, spg: 1.5, fg_pct: 45.5, fg3_pct: 38.8 },
  { name: 'Jimmy Butler',     team: 'MIA', ppg: 20.8, rpg: 5.3, apg: 5.0, spg: 1.3, fg_pct: 49.9, fg3_pct: 24.4 },
];

type SortKey = 'ppg' | 'rpg' | 'apg' | 'spg';

export default function Players() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('ppg');

  useEffect(() => {
    setPlayers(MOCK_PLAYERS);
  }, []);

  const filtered = players
    .filter(p => p.name.toLowerCase().includes(filter.toLowerCase()) || p.team.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => b[sortBy] - a[sortBy]);

  return (
    <div>
      <h1 className="page-title">JUGADORES</h1>
      <p className="page-subtitle">Estadísticas de los mejores jugadores de la temporada.</p>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <input
          className="search-input"
          placeholder="Buscar jugador o equipo…"
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {(['ppg', 'rpg', 'apg', 'spg'] as SortKey[]).map(k => (
            <div key={k} className={`radio-pill${sortBy === k ? ' selected' : ''}`} style={{ flex: 'none', fontSize: 13 }} onClick={() => setSortBy(k)}>
              {k.toUpperCase()}
            </div>
          ))}
        </div>
      </div>

      <div className="players-grid">
        {filtered.map(p => (
          <div key={p.name} className="player-card">
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 2 }}>{p.name}</div>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: '1rem' }}>
              {p.team} · {p.fg_pct.toFixed(1)}% FG · {p.fg3_pct.toFixed(1)}% 3P
            </div>
            <div className="mini-stats">
              {([['PPG', p.ppg], ['RPG', p.rpg], ['APG', p.apg]] as [string, number][]).map(([l, v]) => (
                <div key={l} className="mini-stat">
                  <div className="mini-stat-val">{v}</div>
                  <div className="mini-stat-label">{l}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
        {filtered.length === 0 && <p style={{ color: 'var(--muted)' }}>No se encontraron jugadores.</p>}
      </div>
    </div>
  );
}

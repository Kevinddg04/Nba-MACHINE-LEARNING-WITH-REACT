import { useState } from 'react';
import { api } from '../api';
import type { Team, H2HResult } from '../types';

interface Props { teams: Team[]; }

export default function HeadToHead({ teams }: Props) {
  const [team1, setTeam1] = useState<number>(0);
  const [team2, setTeam2] = useState<number>(0);
  const [result, setResult] = useState<H2HResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetch = async () => {
    if (!team1 || !team2) { setError('Selecciona ambos equipos'); return; }
    if (team1 === team2) { setError('Elige equipos diferentes'); return; }
    setError(''); setLoading(true); setResult(null);
    try {
      const data = await api.headToHead(team1, team2);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const teamOptions = teams.map(t => (
    <option key={t.team_id} value={t.team_id}>{t.name}</option>
  ));

  return (
    <div>
      <h1 className="page-title">HEAD TO HEAD</h1>
      <p className="page-subtitle">Historial de enfrentamientos directos entre dos equipos.</p>

      <div className="card" style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 180 }}>
          <label className="select-label">Equipo 1</label>
          <select value={team1} onChange={e => { setTeam1(Number(e.target.value)); setResult(null); }}>
            <option value={0}>— Selecciona —</option>
            {teamOptions}
          </select>
        </div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--muted)', paddingBottom: '0.75rem' }}>VS</div>
        <div style={{ flex: 1, minWidth: 180 }}>
          <label className="select-label">Equipo 2</label>
          <select value={team2} onChange={e => { setTeam2(Number(e.target.value)); setResult(null); }}>
            <option value={0}>— Selecciona —</option>
            {teamOptions}
          </select>
        </div>
        <button className="btn-secondary" style={{ marginBottom: '0.75rem' }} onClick={fetch} disabled={loading}>
          {loading ? '…' : 'VER H2H'}
        </button>
      </div>

      {error && <div className="error-msg" style={{ marginTop: '1rem' }}>{error}</div>}

      {result && (
        <div className="result-box">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1rem', textAlign: 'center', marginBottom: '1.5rem' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', letterSpacing: '1px' }}>{result.team1_name}</div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '3rem', color: result.team1_wins >= result.team2_wins ? 'var(--green)' : 'var(--muted)' }}>
                {result.team1_wins}
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>victorias</div>
            </div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', color: 'var(--muted)', alignSelf: 'center' }}>VS</div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', letterSpacing: '1px' }}>{result.team2_name}</div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '3rem', color: result.team2_wins >= result.team1_wins ? 'var(--green)' : 'var(--muted)' }}>
                {result.team2_wins}
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>victorias</div>
            </div>
          </div>

          <div className="card-title">Últimos 5 enfrentamientos</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {result.games.map(g => (
              <div key={g.game} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <div style={{ width: 70, color: 'var(--muted)', fontSize: 12 }}>Partido {g.game}</div>
                <div style={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg3)', borderRadius: 8, padding: '8px 14px' }}>
                  <span style={{ fontWeight: 700, color: g.winner_id === result.team1_id ? 'var(--green)' : 'var(--text)' }}>
                    {result.team1_name} {g.t1_score}
                  </span>
                  <span style={{ color: 'var(--muted)', fontSize: 12 }}>–</span>
                  <span style={{ fontWeight: 700, color: g.winner_id === result.team2_id ? 'var(--green)' : 'var(--text)' }}>
                    {g.t2_score} {result.team2_name}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

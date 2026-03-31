import { useState, useCallback } from 'react';
import { api } from '../api';
import type { Team, PredictionResult } from '../types';
import CustomSelect from '../components/CustomSelect';

interface Props { teams: Team[]; }

export default function Predictor({ teams }: Props) {
  const [team1, setTeam1] = useState<number>(0);
  const [team2, setTeam2] = useState<number>(0);
  const [home, setHome] = useState<string>('team1');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [animProb, setAnimProb] = useState(0);

  const predict = useCallback(async () => {
    if (!team1 || !team2) { setError('Selecciona ambos equipos'); return; }
    if (team1 === team2) { setError('Elige equipos diferentes'); return; }
    setError(''); setResult(null); setLoading(true);
    try {
      const data = await api.predict(team1, team2, home);
      setResult(data);
      setTimeout(() => setAnimProb(data.team1_win_prob), 100);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  }, [team1, team2, home]);

  const teamOptions = teams.map(t => ({ value: t.team_id, label: t.name }));

  const t1Name = teams.find(t => t.team_id === team1)?.name ?? 'Equipo 1';
  const t2Name = teams.find(t => t.team_id === team2)?.name ?? 'Equipo 2';

  return (
    <div>
      <div style={{ marginBottom: '2.5rem' }}>
        <div className="hero-tag">Machine Learning</div>
        <h1 className="page-title">PREDICTOR<br />DE PARTIDOS</h1>
        <p className="page-subtitle">Selecciona dos equipos y predice quién gana con nuestro modelo CatBoost.</p>
      </div>

      <div className="card">
        <div className="predictor-grid">
          <div>
            <label className="select-label">Equipo 1</label>
            <CustomSelect
              options={teamOptions}
              value={team1}
              onChange={(v) => { setTeam1(v); setResult(null); }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', paddingTop: '1.8rem' }}>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '2rem', color: 'var(--muted)', letterSpacing: '2px' }}>VS</span>
          </div>
          <div>
            <label className="select-label">Equipo 2</label>
            <CustomSelect
              options={teamOptions}
              value={team2}
              onChange={(v) => { setTeam2(v); setResult(null); }}
            />
          </div>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <div className="card-title">Cancha</div>
          <div className="radio-group">
            {([['team1', `Local: ${t1Name}`], ['neutral', 'Neutral'], ['team2', `Local: ${t2Name}`]] as [string, string][]).map(([v, l]) => (
              <div key={v} className={`radio-pill${home === v ? ' selected' : ''}`} onClick={() => { setHome(v); setResult(null); }}>
                {l}
              </div>
            ))}
          </div>
        </div>

        {error && <div className="error-msg" style={{ marginTop: '1rem' }}>{error}</div>}

        <button className="btn-primary" onClick={predict} disabled={loading || !team1 || !team2 || team1 === team2}>
          {loading ? 'ANALIZANDO…' : 'PREDECIR RESULTADO'}
        </button>
      </div>

      {result && (
        <div className="result-box">
          <div className="card-title">Resultado del modelo CatBoost</div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1rem', textAlign: 'center', marginBottom: '1.5rem' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', letterSpacing: '1px', lineHeight: 1.1 }}>{result.team1_name}</div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '3.5rem', letterSpacing: '2px', color: result.team1_win_prob >= 50 ? 'var(--green)' : 'var(--muted)' }}>
                {result.team1_win_prob}%
              </div>
              <div style={{ fontSize: 14, color: 'var(--muted)', marginTop: 4 }}>Proj: {result.team1_projected_score} pts</div>
            </div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--muted)', alignSelf: 'center' }}>VS</div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', letterSpacing: '1px', lineHeight: 1.1 }}>{result.team2_name}</div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '3.5rem', letterSpacing: '2px', color: result.team2_win_prob >= 50 ? 'var(--green)' : 'var(--muted)' }}>
                {result.team2_win_prob}%
              </div>
              <div style={{ fontSize: 14, color: 'var(--muted)', marginTop: 4 }}>Proj: {result.team2_projected_score} pts</div>
            </div>
          </div>

          <div className="prob-bar-wrap">
            <div className="prob-bar-fill" style={{ width: `${animProb}%` }} />
          </div>

          {result.top_features?.length > 0 && (
            <>
              <div className="card-title" style={{ marginTop: '1.25rem' }}>Top features del modelo</div>
              <div className="factors-grid">
                {result.top_features.slice(0, 3).map(f => (
                  <div key={f.feature} className="factor-card">
                    <div className="factor-val" style={{ color: 'var(--accent2)', fontSize: 16 }}>{f.importance.toFixed(1)}%</div>
                    <div className="factor-label">{f.feature}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          <div style={{ marginTop: '1rem', fontSize: 12, color: 'var(--muted)', textAlign: 'center' }}>
            Modelo: {result.model} · Los resultados son estimaciones estadísticas
          </div>
        </div>
      )}
    </div>
  );
}


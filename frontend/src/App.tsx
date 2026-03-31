import { useState, useEffect } from 'react';
import './index.css';
import { api } from './api';
import type { Team } from './types';
import Predictor from './pages/Predictor';
import Standings from './pages/Standings';
import Players from './pages/Players';
import HeadToHead from './pages/HeadToHead';

type Page = 'Predictor' | 'Standings' | 'Jugadores' | 'Head-to-Head';
const PAGES: Page[] = ['Predictor', 'Standings', 'Jugadores', 'Head-to-Head'];

export default function App() {
  const [page, setPage] = useState<Page>('Predictor');
  const [teams, setTeams] = useState<Team[]>([]);

  useEffect(() => {
    api.getTeams().then(setTeams).catch(() => {});
  }, []);

  return (
    <>
      <header className="header">
        <div className="logo">
          <div className="ball-icon" />
          NBA<span>ML</span>
        </div>
        <nav>
          {PAGES.map(p => (
            <button
              key={p}
              className={`nav-btn${page === p ? ' active' : ''}`}
              onClick={() => setPage(p)}
            >
              {p}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {page === 'Predictor'    && <Predictor teams={teams} />}
        {page === 'Standings'    && <Standings />}
        {page === 'Jugadores'    && <Players />}
        {page === 'Head-to-Head' && <HeadToHead teams={teams} />}
      </main>

      <footer>
        NBA ML Predictor · Proyecto Desarrollo Web 2025 · Datos ilustrativos
      </footer>
    </>
  );
}

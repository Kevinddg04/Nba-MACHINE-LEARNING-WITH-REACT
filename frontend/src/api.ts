import type { Team, PredictionResult, H2HResult, ModelInfo, SystemMetrics } from './types';

const rawBase = import.meta.env.VITE_API_URL || '';
const hostBase = rawBase.endsWith('/') ? rawBase.slice(0, -1) : rawBase;
// Prevenir que el frontend intente buscar datos sin la ruta /api obligatoria del servidor
const BASE = hostBase.endsWith('/api') ? hostBase : (hostBase + '/api');

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Error en la API');
  return data as T;
}

export const api = {
  getTeams: () =>
    request<Team[]>('/teams'),

  predict: (team1: number, team2: number, homeTeam: string) =>
    request<PredictionResult>('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team1, team2, home_team: homeTeam }),
    }),

  getStandings: (conf: string = 'all') =>
    request<Team[]>(`/standings?conf=${conf}`),

  headToHead: (team1: number, team2: number) =>
    request<H2HResult>('/head-to-head', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team1, team2 }),
    }),

  modelInfo: () =>
    request<ModelInfo>('/model/info'),

  getMetrics: () =>
    request<SystemMetrics>('/metrics'),
};

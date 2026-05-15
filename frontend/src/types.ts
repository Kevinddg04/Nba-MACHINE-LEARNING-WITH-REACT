export interface Team {
  team_id: number;
  name: string;
  conf: string;
  expectedScore: number;
  expectedOpponentScore: number;
  net_expected: number;
  win_streak_5: number;
  RealHandicap: number;
  fg_pct: number;
}

export interface PredictionResult {
  prediction: string;
  win_probability: number;
  team1: { name: string; probability: number };
  team2: { name: string; probability: number };
  model_info: string;
  details: {
    t1_streak: number;
    t2_streak: number;
    home_court: string;
  };
}

export interface H2HGame {
  game: number;
  t1_score: number;
  t2_score: number;
  winner_id: number;
}

export interface H2HResult {
  team1_id: number;
  team2_id: number;
  team1_name: string;
  team2_name: string;
  team1_wins: number;
  team2_wins: number;
  games: H2HGame[];
}

export interface TopFeature {
  feature: string;
  importance: number;
}

export interface ModelInfo {
  model_type: string;
  iterations: number;
  num_features: number;
  top_10_features: TopFeature[];
  teams_in_snapshot: number;
}

export interface MetricHistory {
  id: number;
  team1: string;
  team2: string;
  home: string;
  predicted: string;
  prob: number;
  actual_winner: string | null;
  correct: boolean | null;
  date: string;
}

export interface SystemMetrics {
  metrics: {
    total_predictions: number;
    resolved: number;
    correct: number;
    hit_rate: number;
  };
  history: MetricHistory[];
}

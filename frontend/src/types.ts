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

export interface Player {
  name: string;
  team: string;
  ppg: number;
  rpg: number;
  apg: number;
  spg: number;
  bpg: number;
  fg_pct: number;
  fg3_pct: number;
}

export interface TopFeature {
  feature: string;
  importance: number;
}

export interface TeamRecent {
  win_streak_5: number;
  expectedTeamScore: number;
  RealHandicap: number;
}

export interface PredictionResult {
  team1_id: number;
  team2_id: number;
  team1_name: string;
  team2_name: string;
  team1_win_prob: number;
  team2_win_prob: number;
  team1_projected_score: number;
  team2_projected_score: number;
  home_team: string;
  model: string;
  top_features: TopFeature[];
  team1_recent: TeamRecent;
  team2_recent: TeamRecent;
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

export interface ModelInfo {
  model_type: string;
  iterations: number;
  num_features: number;
  top_10_features: TopFeature[];
  teams_in_snapshot: number;
}

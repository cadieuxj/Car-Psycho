import { clsx, type ClassValue } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatDate(date: string | Date): string {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function formatDateTime(date: string | Date): string {
  return new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatPercentage(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

export const OCEAN_COLORS = {
  openness: '#8b5cf6',
  conscientiousness: '#3b82f6',
  extraversion: '#f59e0b',
  agreeableness: '#10b981',
  neuroticism: '#ef4444',
} as const;

export const OCEAN_LABELS = {
  openness: 'Openness',
  conscientiousness: 'Conscientiousness',
  extraversion: 'Extraversion',
  agreeableness: 'Agreeableness',
  neuroticism: 'Neuroticism',
} as const;

export const OCEAN_DESCRIPTIONS = {
  openness: 'Appreciation for art, emotion, adventure, unusual ideas, imagination, and curiosity',
  conscientiousness: 'Tendency to be organized, dependable, self-disciplined, and aim for achievement',
  extraversion: 'Tendency to seek stimulation in the company of others, talkativeness, assertiveness',
  agreeableness: 'Tendency to be compassionate and cooperative rather than suspicious and antagonistic',
  neuroticism: 'Tendency to experience unpleasant emotions easily, such as anxiety and irritability',
} as const;

export function getTraitLevel(score: number): 'low' | 'moderate' | 'high' {
  if (score < 0.33) return 'low';
  if (score < 0.67) return 'moderate';
  return 'high';
}

export function getTraitDescription(trait: keyof typeof OCEAN_LABELS, score: number): string {
  const level = getTraitLevel(score);
  const descriptions: Record<keyof typeof OCEAN_LABELS, Record<'low' | 'moderate' | 'high', string>> = {
    openness: {
      low: 'Prefers routine, practical solutions, and conventional choices',
      moderate: 'Balanced between traditional and new experiences',
      high: 'Seeks innovation, creative features, and unique experiences',
    },
    conscientiousness: {
      low: 'Flexible, spontaneous decision-making style',
      moderate: 'Balanced approach to planning and flexibility',
      high: 'Values reliability, safety features, and thorough research',
    },
    extraversion: {
      low: 'Prefers understated, practical vehicles',
      moderate: 'Appreciates versatility in vehicle choice',
      high: 'Drawn to sporty, attention-getting vehicles',
    },
    agreeableness: {
      low: 'Independent decision-maker, performance-focused',
      moderate: 'Balances personal and others\' needs',
      high: 'Prioritizes family comfort and safety',
    },
    neuroticism: {
      low: 'Confident in decisions, open to trying new options',
      moderate: 'Considers safety but remains open to options',
      high: 'Prioritizes safety ratings and proven reliability',
    },
  };
  return descriptions[trait][level];
}

export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

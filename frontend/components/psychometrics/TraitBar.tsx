'use client';

import { cn, OCEAN_COLORS, OCEAN_LABELS, OCEAN_DESCRIPTIONS, getTraitLevel, getTraitDescription } from '@/lib/utils';
import { OceanScores, ConfidenceScores } from '@/lib/types';
import { Info } from 'lucide-react';

interface TraitBarProps {
  trait: keyof OceanScores;
  score: number;
  confidence?: number;
  showDescription?: boolean;
  animated?: boolean;
}

export function TraitBar({
  trait,
  score,
  confidence,
  showDescription = false,
  animated = true
}: TraitBarProps) {
  const percentage = Math.round(score * 100);
  const level = getTraitLevel(score);
  const color = OCEAN_COLORS[trait];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">{OCEAN_LABELS[trait]}</span>
          {showDescription && (
            <div className="group relative">
              <Info className="w-4 h-4 text-gray-400 cursor-help" />
              <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-gray-900 text-white text-xs rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                {OCEAN_DESCRIPTIONS[trait]}
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold" style={{ color }}>
            {percentage}%
          </span>
          {confidence !== undefined && (
            <span className="text-xs text-gray-400">
              ({Math.round(confidence * 100)}% conf)
            </span>
          )}
        </div>
      </div>

      <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full',
            animated && 'transition-all duration-500 ease-out'
          )}
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
          }}
        />
      </div>

      {showDescription && (
        <p className="text-sm text-gray-500">
          {getTraitDescription(trait, score)}
        </p>
      )}
    </div>
  );
}

interface TraitBarsProps {
  scores: OceanScores;
  confidence?: ConfidenceScores;
  showDescriptions?: boolean;
}

export function TraitBars({ scores, confidence, showDescriptions = false }: TraitBarsProps) {
  const traits: (keyof OceanScores)[] = [
    'openness',
    'conscientiousness',
    'extraversion',
    'agreeableness',
    'neuroticism',
  ];

  return (
    <div className="space-y-4">
      {traits.map((trait) => (
        <TraitBar
          key={trait}
          trait={trait}
          score={scores[trait]}
          confidence={confidence?.[trait]}
          showDescription={showDescriptions}
        />
      ))}
    </div>
  );
}

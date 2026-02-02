'use client';

import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Badge } from '@/components/ui';
import { PersonalityProfile, OceanScores } from '@/lib/types';
import { TraitBars } from './TraitBar';
import OceanRadarChart from './OceanRadarChart';
import { formatDateTime, OCEAN_LABELS } from '@/lib/utils';
import { Brain, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface PersonalityCardProps {
  profile: PersonalityProfile;
  previousProfile?: PersonalityProfile;
  showChart?: boolean;
  compact?: boolean;
}

function getTopTraits(scores: OceanScores, count: number = 2): (keyof OceanScores)[] {
  return Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .slice(0, count)
    .map(([key]) => key as keyof OceanScores);
}

function getScoreChange(current: number, previous: number): 'up' | 'down' | 'stable' {
  const diff = current - previous;
  if (Math.abs(diff) < 0.05) return 'stable';
  return diff > 0 ? 'up' : 'down';
}

export default function PersonalityCard({
  profile,
  previousProfile,
  showChart = true,
  compact = false,
}: PersonalityCardProps) {
  const topTraits = getTopTraits(profile.scores);
  const avgConfidence = Object.values(profile.confidence).reduce((a, b) => a + b, 0) / 5;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <CardTitle>Personality Profile</CardTitle>
              <CardDescription>
                Big Five (OCEAN) Analysis
              </CardDescription>
            </div>
          </div>
          <Badge variant={avgConfidence > 0.8 ? 'success' : avgConfidence > 0.6 ? 'warning' : 'info'}>
            {Math.round(avgConfidence * 100)}% confidence
          </Badge>
        </div>
      </CardHeader>

      <CardBody className="space-y-6">
        {/* Top Traits Summary */}
        <div className="flex flex-wrap gap-2">
          {topTraits.map((trait) => (
            <div
              key={trait}
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium trait-${trait}`}
            >
              {previousProfile && (
                <>
                  {getScoreChange(profile.scores[trait], previousProfile.scores[trait]) === 'up' && (
                    <TrendingUp className="w-3 h-3" />
                  )}
                  {getScoreChange(profile.scores[trait], previousProfile.scores[trait]) === 'down' && (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  {getScoreChange(profile.scores[trait], previousProfile.scores[trait]) === 'stable' && (
                    <Minus className="w-3 h-3" />
                  )}
                </>
              )}
              <span>High {OCEAN_LABELS[trait]}</span>
            </div>
          ))}
        </div>

        {/* Chart and Bars */}
        {!compact && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {showChart && (
              <div>
                <OceanRadarChart
                  scores={profile.scores}
                  previousScores={previousProfile?.scores}
                  size="md"
                />
              </div>
            )}
            <div>
              <TraitBars
                scores={profile.scores}
                confidence={profile.confidence}
                showDescriptions={!showChart}
              />
            </div>
          </div>
        )}

        {compact && (
          <TraitBars scores={profile.scores} confidence={profile.confidence} />
        )}

        {/* Timestamp */}
        <p className="text-xs text-gray-400 text-right">
          Analyzed: {formatDateTime(profile.timestamp)}
        </p>
      </CardBody>
    </Card>
  );
}

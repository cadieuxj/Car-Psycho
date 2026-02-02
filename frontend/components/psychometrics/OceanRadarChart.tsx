'use client';

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import { OceanScores } from '@/lib/types';
import { OCEAN_LABELS, OCEAN_COLORS } from '@/lib/utils';

interface OceanRadarChartProps {
  scores: OceanScores;
  previousScores?: OceanScores;
  size?: 'sm' | 'md' | 'lg';
}

export default function OceanRadarChart({
  scores,
  previousScores,
  size = 'md'
}: OceanRadarChartProps) {
  const data = Object.entries(OCEAN_LABELS).map(([key, label]) => ({
    trait: label,
    current: Math.round(scores[key as keyof OceanScores] * 100),
    previous: previousScores ? Math.round(previousScores[key as keyof OceanScores] * 100) : undefined,
    fullMark: 100,
  }));

  const heights = {
    sm: 200,
    md: 300,
    lg: 400,
  };

  return (
    <ResponsiveContainer width="100%" height={heights[size]}>
      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
        <PolarGrid stroke="#e5e7eb" />
        <PolarAngleAxis
          dataKey="trait"
          tick={{ fill: '#6b7280', fontSize: 12 }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={{ fill: '#9ca3af', fontSize: 10 }}
          tickCount={5}
        />
        {previousScores && (
          <Radar
            name="Previous"
            dataKey="previous"
            stroke="#9ca3af"
            fill="#9ca3af"
            fillOpacity={0.2}
            strokeWidth={1}
            strokeDasharray="4 4"
          />
        )}
        <Radar
          name="Current"
          dataKey="current"
          stroke="#0ea5e9"
          fill="#0ea5e9"
          fillOpacity={0.3}
          strokeWidth={2}
        />
        <Tooltip
          formatter={(value: number) => [`${value}%`, '']}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          }}
        />
        {previousScores && <Legend />}
      </RadarChart>
    </ResponsiveContainer>
  );
}

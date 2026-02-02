'use client';

import { Card, CardHeader, CardTitle, CardBody } from '@/components/ui';
import { Badge } from '@/components/ui';
import { PersonalityProfile, CarRecommendation } from '@/lib/types';
import { TraitBars } from '@/components/psychometrics';
import { Brain, Car, Lightbulb, User } from 'lucide-react';
import { OCEAN_LABELS } from '@/lib/utils';

interface ChatSidebarProps {
  profile?: PersonalityProfile;
  recommendations?: CarRecommendation[];
  customerName?: string;
}

export default function ChatSidebar({
  profile,
  recommendations = [],
  customerName = 'Customer'
}: ChatSidebarProps) {
  return (
    <div className="w-80 border-l border-gray-200 bg-white overflow-y-auto">
      {/* Customer Info */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
            <User className="w-5 h-5 text-gray-500" />
          </div>
          <div>
            <h3 className="font-medium text-gray-900">{customerName}</h3>
            <p className="text-sm text-gray-500">Active conversation</p>
          </div>
        </div>
      </div>

      {/* Personality Profile */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-4 h-4 text-purple-600" />
          <h4 className="font-medium text-gray-900">Personality Profile</h4>
        </div>

        {profile ? (
          <div className="space-y-4">
            <TraitBars scores={profile.scores} />

            {/* Top Traits */}
            <div className="flex flex-wrap gap-2 mt-3">
              {Object.entries(profile.scores)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 2)
                .map(([trait]) => (
                  <Badge
                    key={trait}
                    variant="info"
                    className={`trait-${trait}`}
                  >
                    High {OCEAN_LABELS[trait as keyof typeof OCEAN_LABELS]}
                  </Badge>
                ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-gray-400">
            <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Profile will update as conversation progresses</p>
          </div>
        )}
      </div>

      {/* Sales Tips */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-4">
          <Lightbulb className="w-4 h-4 text-amber-600" />
          <h4 className="font-medium text-gray-900">Sales Tips</h4>
        </div>

        {profile ? (
          <ul className="space-y-2 text-sm text-gray-600">
            {profile.scores.conscientiousness > 0.7 && (
              <li className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">*</span>
                <span>Emphasize reliability data and safety ratings</span>
              </li>
            )}
            {profile.scores.openness > 0.7 && (
              <li className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">*</span>
                <span>Highlight innovative features and technology</span>
              </li>
            )}
            {profile.scores.agreeableness > 0.7 && (
              <li className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">*</span>
                <span>Focus on family comfort and practicality</span>
              </li>
            )}
            {profile.scores.extraversion > 0.7 && (
              <li className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">*</span>
                <span>Discuss sporty features and social appeal</span>
              </li>
            )}
            {profile.scores.neuroticism > 0.6 && (
              <li className="flex items-start gap-2">
                <span className="text-amber-500 mt-0.5">*</span>
                <span>Provide reassurance with warranties and reviews</span>
              </li>
            )}
          </ul>
        ) : (
          <p className="text-sm text-gray-400">Tips will appear based on profile analysis</p>
        )}
      </div>

      {/* Car Recommendations */}
      <div className="p-4">
        <div className="flex items-center gap-2 mb-4">
          <Car className="w-4 h-4 text-emerald-600" />
          <h4 className="font-medium text-gray-900">Recommendations</h4>
        </div>

        {recommendations.length > 0 ? (
          <div className="space-y-3">
            {recommendations.slice(0, 3).map((car) => (
              <div
                key={car.id}
                className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-gray-900 text-sm">
                    {car.year} {car.make} {car.model}
                  </span>
                  <Badge variant="success">{car.match_score}%</Badge>
                </div>
                <p className="text-xs text-gray-500">
                  ${car.price.toLocaleString()}
                </p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {car.match_reasons.slice(0, 2).map((reason, i) => (
                    <span key={i} className="text-xs text-emerald-600">
                      {reason}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-6 text-gray-400">
            <Car className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Recommendations will appear based on preferences</p>
          </div>
        )}
      </div>
    </div>
  );
}

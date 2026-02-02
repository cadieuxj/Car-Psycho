'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { PersonalityCard, OceanRadarChart, TraitBars } from '@/components/psychometrics';
import { PersonalityProfile } from '@/lib/types';
import { Brain, RefreshCw, History, Download, Filter } from 'lucide-react';

// Sample data for demonstration
const sampleProfile: PersonalityProfile = {
  scores: {
    openness: 0.72,
    conscientiousness: 0.85,
    extraversion: 0.58,
    agreeableness: 0.76,
    neuroticism: 0.32,
  },
  confidence: {
    openness: 0.88,
    conscientiousness: 0.92,
    extraversion: 0.85,
    agreeableness: 0.90,
    neuroticism: 0.87,
  },
  reasoning: {
    openness: 'Interest in electric vehicles and innovative features suggests high openness',
    conscientiousness: 'Strong emphasis on safety ratings and reliability indicates high conscientiousness',
    extraversion: 'Moderate interest in social aspects of car ownership',
    agreeableness: 'Family-oriented preferences and comfort focus suggest high agreeableness',
    neuroticism: 'Confident decision-making style with low anxiety indicators',
  },
  timestamp: new Date().toISOString(),
};

const previousProfile: PersonalityProfile = {
  scores: {
    openness: 0.65,
    conscientiousness: 0.80,
    extraversion: 0.55,
    agreeableness: 0.70,
    neuroticism: 0.38,
  },
  confidence: {
    openness: 0.75,
    conscientiousness: 0.82,
    extraversion: 0.78,
    agreeableness: 0.80,
    neuroticism: 0.76,
  },
  timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
};

const recentProfiles = [
  { id: '1', customer: 'John D.', ...sampleProfile },
  { id: '2', customer: 'Sarah M.', scores: { openness: 0.85, conscientiousness: 0.65, extraversion: 0.78, agreeableness: 0.60, neuroticism: 0.45 }, confidence: sampleProfile.confidence, timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString() },
  { id: '3', customer: 'Mike R.', scores: { openness: 0.55, conscientiousness: 0.90, extraversion: 0.42, agreeableness: 0.82, neuroticism: 0.28 }, confidence: sampleProfile.confidence, timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString() },
];

export default function PsychometricsPage() {
  const [selectedProfile, setSelectedProfile] = useState<typeof recentProfiles[0] | null>(recentProfiles[0]);
  const [showComparison, setShowComparison] = useState(false);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Psychometric Analysis</h1>
          <p className="text-gray-500 mt-1">
            View and analyze customer personality profiles using the OCEAN model
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" icon={<Filter className="w-4 h-4" />}>
            Filter
          </Button>
          <Button variant="secondary" icon={<Download className="w-4 h-4" />}>
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Profile List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Recent Profiles</CardTitle>
              <CardDescription>Select a profile to view details</CardDescription>
            </CardHeader>
            <CardBody className="p-0">
              <div className="divide-y divide-gray-100">
                {recentProfiles.map((profile) => (
                  <button
                    key={profile.id}
                    onClick={() => setSelectedProfile(profile)}
                    className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
                      selectedProfile?.id === profile.id ? 'bg-primary-50 border-l-4 border-l-primary-600' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">{profile.customer}</span>
                      <Badge variant="info">
                        {Math.round(Object.values(profile.confidence).reduce((a, b) => a + b, 0) / 5 * 100)}%
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(profile.scores)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 2)
                        .map(([trait, score]) => (
                          <span
                            key={trait}
                            className={`text-xs px-2 py-0.5 rounded-full trait-${trait}`}
                          >
                            {trait.charAt(0).toUpperCase() + trait.slice(1, 3)}: {Math.round(score * 100)}%
                          </span>
                        ))}
                    </div>
                  </button>
                ))}
              </div>
            </CardBody>
          </Card>

          {/* Analysis Settings */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Analysis Settings</CardTitle>
            </CardHeader>
            <CardBody className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Show comparison</span>
                <button
                  onClick={() => setShowComparison(!showComparison)}
                  className={`w-10 h-6 rounded-full transition-colors ${
                    showComparison ? 'bg-primary-600' : 'bg-gray-200'
                  }`}
                >
                  <div
                    className={`w-4 h-4 bg-white rounded-full shadow transform transition-transform ${
                      showComparison ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <div className="text-xs text-gray-400">
                Compare current profile with previous analysis
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Right Column - Selected Profile Details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedProfile ? (
            <>
              <PersonalityCard
                profile={selectedProfile as PersonalityProfile}
                previousProfile={showComparison ? previousProfile : undefined}
              />

              {/* Reasoning Card */}
              {sampleProfile.reasoning && (
                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Reasoning</CardTitle>
                    <CardDescription>
                      AI-generated explanations for trait assessments
                    </CardDescription>
                  </CardHeader>
                  <CardBody>
                    <div className="space-y-4">
                      {Object.entries(sampleProfile.reasoning).map(([trait, reason]) => (
                        <div key={trait} className="flex gap-3">
                          <div className={`w-1 rounded-full trait-${trait}`} />
                          <div>
                            <p className="font-medium text-gray-900 capitalize">{trait}</p>
                            <p className="text-sm text-gray-600">{reason}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardBody>
                </Card>
              )}

              {/* Car Preference Insights */}
              <Card>
                <CardHeader>
                  <CardTitle>Car Preference Insights</CardTitle>
                  <CardDescription>
                    Based on personality profile analysis
                  </CardDescription>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">Likely Preferences</h4>
                      <ul className="text-sm text-blue-700 space-y-1">
                        <li>Safety features and ratings</li>
                        <li>Reliable brands with good track record</li>
                        <li>Family-friendly vehicles</li>
                        <li>Moderate interest in EV technology</li>
                      </ul>
                    </div>
                    <div className="p-4 bg-amber-50 rounded-lg">
                      <h4 className="font-medium text-amber-900 mb-2">Sales Approach</h4>
                      <ul className="text-sm text-amber-700 space-y-1">
                        <li>Emphasize safety certifications</li>
                        <li>Provide detailed specifications</li>
                        <li>Highlight warranty and reliability</li>
                        <li>Allow time for research and comparison</li>
                      </ul>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <CardBody className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Brain className="w-12 h-12 mb-4 text-gray-300" />
                <p>Select a profile to view details</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

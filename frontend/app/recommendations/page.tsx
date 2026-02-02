'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardBody } from '@/components/ui';
import { Button } from '@/components/ui';
import { Badge } from '@/components/ui';
import { CarRecommendation, OceanScores } from '@/lib/types';
import { OCEAN_LABELS, OCEAN_COLORS } from '@/lib/utils';
import {
  Car,
  Filter,
  Search,
  Star,
  Fuel,
  Shield,
  Users,
  Zap,
  ChevronRight,
  Heart,
  Share2,
} from 'lucide-react';

// Sample car recommendations
const sampleCars: CarRecommendation[] = [
  {
    id: '1',
    make: 'Toyota',
    model: 'RAV4 Hybrid',
    year: 2024,
    price: 35400,
    image_url: '/cars/rav4.jpg',
    match_score: 94,
    match_reasons: ['Top Safety Pick+', 'Excellent fuel economy', 'Spacious interior'],
    personality_alignment: [
      { trait: 'conscientiousness', alignment: 'high', reason: 'IIHS Top Safety Pick+ rating' },
      { trait: 'agreeableness', alignment: 'high', reason: 'Family-friendly with ample cargo space' },
      { trait: 'openness', alignment: 'medium', reason: 'Hybrid technology and modern features' },
    ],
    features: ['All-Wheel Drive', 'Hybrid Engine', 'Toyota Safety Sense 2.5+', 'Apple CarPlay'],
  },
  {
    id: '2',
    make: 'Honda',
    model: 'CR-V',
    year: 2024,
    price: 33400,
    image_url: '/cars/crv.jpg',
    match_score: 91,
    match_reasons: ['Proven reliability', 'Great resale value', 'Comfortable ride'],
    personality_alignment: [
      { trait: 'conscientiousness', alignment: 'high', reason: 'Industry-leading reliability' },
      { trait: 'agreeableness', alignment: 'high', reason: 'Comfortable for all passengers' },
    ],
    features: ['All-Wheel Drive', 'Honda Sensing', 'Wireless CarPlay', 'Panoramic Roof'],
  },
  {
    id: '3',
    make: 'Tesla',
    model: 'Model Y',
    year: 2024,
    price: 44990,
    image_url: '/cars/modely.jpg',
    match_score: 87,
    match_reasons: ['Zero emissions', 'Cutting-edge technology', 'Performance'],
    personality_alignment: [
      { trait: 'openness', alignment: 'high', reason: 'Innovative EV technology' },
      { trait: 'extraversion', alignment: 'medium', reason: 'Attention-getting design' },
    ],
    features: ['All-Electric', 'Autopilot', 'Over-the-air updates', 'Supercharger network'],
  },
  {
    id: '4',
    make: 'Mazda',
    model: 'CX-5',
    year: 2024,
    price: 29300,
    image_url: '/cars/cx5.jpg',
    match_score: 85,
    match_reasons: ['Premium interior', 'Engaging drive', 'Great value'],
    personality_alignment: [
      { trait: 'openness', alignment: 'medium', reason: 'Stylish Kodo design language' },
      { trait: 'conscientiousness', alignment: 'medium', reason: 'Quality build and materials' },
    ],
    features: ['i-Activ AWD', 'i-Activsense Safety', 'Premium Bose Audio', 'Heated Seats'],
  },
  {
    id: '5',
    make: 'Subaru',
    model: 'Outback',
    year: 2024,
    price: 31995,
    image_url: '/cars/outback.jpg',
    match_score: 82,
    match_reasons: ['Standard AWD', 'Adventure-ready', 'Safety features'],
    personality_alignment: [
      { trait: 'openness', alignment: 'high', reason: 'Outdoor adventure capability' },
      { trait: 'conscientiousness', alignment: 'high', reason: 'EyeSight Driver Assist' },
    ],
    features: ['Symmetrical AWD', 'EyeSight', 'X-Mode', '8.7" Ground Clearance'],
  },
  {
    id: '6',
    make: 'BMW',
    model: 'X3',
    year: 2024,
    price: 48800,
    image_url: '/cars/x3.jpg',
    match_score: 78,
    match_reasons: ['Luxury brand', 'Sporty handling', 'Premium features'],
    personality_alignment: [
      { trait: 'extraversion', alignment: 'high', reason: 'Prestigious badge appeal' },
      { trait: 'openness', alignment: 'medium', reason: 'Advanced technology features' },
    ],
    features: ['xDrive AWD', 'iDrive 8', 'M Sport Package', 'Harman Kardon Audio'],
  },
];

interface CarCardProps {
  car: CarRecommendation;
  onSelect: () => void;
}

function CarCard({ car, onSelect }: CarCardProps) {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer" onClick={onSelect}>
      {/* Image placeholder */}
      <div className="h-48 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center relative">
        <Car className="w-20 h-20 text-gray-300" />
        <div className="absolute top-3 right-3">
          <Badge
            variant={car.match_score >= 90 ? 'success' : car.match_score >= 80 ? 'info' : 'default'}
            className="text-sm font-bold"
          >
            {car.match_score}% Match
          </Badge>
        </div>
        <button className="absolute top-3 left-3 p-2 bg-white/80 rounded-full hover:bg-white transition-colors">
          <Heart className="w-4 h-4 text-gray-400 hover:text-red-500" />
        </button>
      </div>

      <CardBody>
        <div className="flex items-start justify-between mb-2">
          <div>
            <h3 className="font-semibold text-gray-900">
              {car.year} {car.make} {car.model}
            </h3>
            <p className="text-lg font-bold text-primary-600">
              ${car.price.toLocaleString()}
            </p>
          </div>
        </div>

        {/* Match Reasons */}
        <div className="flex flex-wrap gap-1 mb-3">
          {car.match_reasons.slice(0, 2).map((reason, i) => (
            <span key={i} className="text-xs px-2 py-1 bg-green-50 text-green-700 rounded-full">
              {reason}
            </span>
          ))}
        </div>

        {/* Personality Alignment */}
        <div className="flex items-center gap-2 mb-3">
          {car.personality_alignment.slice(0, 3).map((alignment) => (
            <div
              key={alignment.trait}
              className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold"
              style={{ backgroundColor: OCEAN_COLORS[alignment.trait] }}
              title={`${OCEAN_LABELS[alignment.trait]}: ${alignment.reason}`}
            >
              {alignment.trait.charAt(0).toUpperCase()}
            </div>
          ))}
        </div>

        {/* Features */}
        <div className="flex flex-wrap gap-2 text-xs text-gray-500">
          {car.features.slice(0, 3).map((feature, i) => (
            <span key={i} className="flex items-center gap-1">
              {feature.toLowerCase().includes('awd') && <Shield className="w-3 h-3" />}
              {feature.toLowerCase().includes('hybrid') && <Fuel className="w-3 h-3" />}
              {feature.toLowerCase().includes('electric') && <Zap className="w-3 h-3" />}
              {feature}
            </span>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

export default function RecommendationsPage() {
  const [cars] = useState<CarRecommendation[]>(sampleCars);
  const [selectedCar, setSelectedCar] = useState<CarRecommendation | null>(null);
  const [filter, setFilter] = useState<'all' | 'high-match' | 'budget'>('all');

  const filteredCars = cars.filter((car) => {
    if (filter === 'high-match') return car.match_score >= 85;
    if (filter === 'budget') return car.price < 35000;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Car Recommendations</h1>
          <p className="text-gray-500 mt-1">
            AI-powered vehicle matches based on personality profiles
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search cars..."
              className="input pl-10 w-64"
            />
          </div>
          <Button variant="secondary" icon={<Filter className="w-4 h-4" />}>
            Filters
          </Button>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'all'
              ? 'bg-primary-100 text-primary-700'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          All Vehicles ({cars.length})
        </button>
        <button
          onClick={() => setFilter('high-match')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'high-match'
              ? 'bg-primary-100 text-primary-700'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          High Match (85%+)
        </button>
        <button
          onClick={() => setFilter('budget')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            filter === 'budget'
              ? 'bg-primary-100 text-primary-700'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Under $35K
        </button>
      </div>

      {/* Cars Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredCars.map((car) => (
          <CarCard key={car.id} car={car} onSelect={() => setSelectedCar(car)} />
        ))}
      </div>

      {/* Selected Car Modal/Drawer would go here */}
      {selectedCar && (
        <div
          className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedCar(null)}
        >
          <Card className="max-w-2xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>
                    {selectedCar.year} {selectedCar.make} {selectedCar.model}
                  </CardTitle>
                  <CardDescription>
                    ${selectedCar.price.toLocaleString()} | {selectedCar.match_score}% Match
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button variant="ghost" size="sm" icon={<Heart className="w-4 h-4" />}>
                    Save
                  </Button>
                  <Button variant="ghost" size="sm" icon={<Share2 className="w-4 h-4" />}>
                    Share
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardBody className="space-y-6">
              {/* Match Reasons */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Why This Car Matches</h4>
                <div className="space-y-2">
                  {selectedCar.match_reasons.map((reason, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      <Star className="w-4 h-4 text-yellow-500" />
                      <span>{reason}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Personality Alignment */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Personality Alignment</h4>
                <div className="space-y-3">
                  {selectedCar.personality_alignment.map((alignment) => (
                    <div
                      key={alignment.trait}
                      className="flex items-start gap-3 p-3 rounded-lg"
                      style={{ backgroundColor: `${OCEAN_COLORS[alignment.trait]}10` }}
                    >
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0"
                        style={{ backgroundColor: OCEAN_COLORS[alignment.trait] }}
                      >
                        {alignment.trait.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className="font-medium" style={{ color: OCEAN_COLORS[alignment.trait] }}>
                          {OCEAN_LABELS[alignment.trait]} - {alignment.alignment.charAt(0).toUpperCase() + alignment.alignment.slice(1)} Alignment
                        </p>
                        <p className="text-sm text-gray-600">{alignment.reason}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Features */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Key Features</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedCar.features.map((feature, i) => (
                    <Badge key={i} variant="default">
                      {feature}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* CTA */}
              <div className="flex gap-3 pt-4 border-t">
                <Button className="flex-1">Schedule Test Drive</Button>
                <Button variant="secondary" className="flex-1">
                  View Full Details
                </Button>
              </div>
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}

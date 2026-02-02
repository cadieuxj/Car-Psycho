import { ServiceStatus, StatsCards, RecentActivity, QuickActions } from '@/components/dashboard';

export default function Home() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 mt-1">
            Welcome to Car-Psycho, your AI-powered psychometric sales platform
          </p>
        </div>
      </div>

      {/* Stats Grid */}
      <StatsCards />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Service Status */}
        <div className="lg:col-span-1 space-y-6">
          <ServiceStatus />
          <QuickActions />
        </div>

        {/* Right Column - Activity */}
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
      </div>
    </div>
  );
}

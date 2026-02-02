'use client';

import { useState, useEffect } from 'react';
import { Bell, Search, User, Moon, Sun, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';
import { ServiceHealth } from '@/lib/types';

export default function Header() {
  const [servicesHealth, setServicesHealth] = useState<{
    manager: ServiceHealth;
    inference: ServiceHealth;
    data: ServiceHealth;
  } | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      const health = await api.getAllServicesHealth();
      setServicesHealth(health);
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const allHealthy = servicesHealth
    ? servicesHealth.manager.status === 'healthy' &&
      servicesHealth.inference.status === 'healthy' &&
      servicesHealth.data.status === 'healthy'
    : false;

  const healthyCount = servicesHealth
    ? [servicesHealth.manager, servicesHealth.inference, servicesHealth.data].filter(
        (s) => s.status === 'healthy'
      ).length
    : 0;

  return (
    <header className="flex items-center justify-between h-16 px-6 bg-white border-b border-gray-200">
      {/* Search */}
      <div className="flex-1 max-w-xl">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search customers, cars, or conversations..."
            className="input pl-10 w-full"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-4">
        {/* Service Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-100">
          {allHealthy ? (
            <Wifi className="w-4 h-4 text-green-600" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-500" />
          )}
          <span className="text-sm text-gray-600">
            {healthyCount}/3 Services
          </span>
          <span
            className={cn(
              'w-2 h-2 rounded-full',
              allHealthy ? 'bg-green-500' : healthyCount > 0 ? 'bg-yellow-500' : 'bg-red-500'
            )}
          />
        </div>

        {/* Dark Mode Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
        >
          {darkMode ? (
            <Sun className="w-5 h-5 text-gray-600" />
          ) : (
            <Moon className="w-5 h-5 text-gray-600" />
          )}
        </button>

        {/* Notifications */}
        <button className="relative p-2 rounded-lg hover:bg-gray-100 transition-colors">
          <Bell className="w-5 h-5 text-gray-600" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        {/* Profile */}
        <button className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-100 transition-colors">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <div className="text-left hidden md:block">
            <p className="text-sm font-medium text-gray-900">Sales Rep</p>
            <p className="text-xs text-gray-500">Online</p>
          </div>
        </button>
      </div>
    </header>
  );
}

'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatMessage, ChatInput, ChatSidebar } from '@/components/chat';
import { ChatMessage as ChatMessageType, PersonalityProfile, CarRecommendation } from '@/lib/types';
import { generateId } from '@/lib/utils';
import { MessageSquare, Plus, Settings, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui';

// Sample recommendations
const sampleRecommendations: CarRecommendation[] = [
  {
    id: '1',
    make: 'Toyota',
    model: 'RAV4 Hybrid',
    year: 2024,
    price: 35000,
    match_score: 92,
    match_reasons: ['High safety ratings', 'Fuel efficient', 'Family-friendly'],
    personality_alignment: [
      { trait: 'conscientiousness', alignment: 'high', reason: 'Top safety pick' },
      { trait: 'agreeableness', alignment: 'high', reason: 'Spacious for family' },
    ],
    features: ['AWD', 'Hybrid', 'Toyota Safety Sense'],
  },
  {
    id: '2',
    make: 'Honda',
    model: 'CR-V',
    year: 2024,
    price: 33000,
    match_score: 88,
    match_reasons: ['Reliable brand', 'Good resale value'],
    personality_alignment: [
      { trait: 'conscientiousness', alignment: 'high', reason: 'Proven reliability' },
    ],
    features: ['AWD', 'Honda Sensing', 'Spacious cargo'],
  },
  {
    id: '3',
    make: 'Mazda',
    model: 'CX-5',
    year: 2024,
    price: 31000,
    match_score: 85,
    match_reasons: ['Premium feel', 'Engaging drive'],
    personality_alignment: [
      { trait: 'openness', alignment: 'medium', reason: 'Stylish design' },
    ],
    features: ['AWD', 'Premium interior', 'i-Activsense'],
  },
];

// Initial messages
const initialMessages: ChatMessageType[] = [
  {
    id: '1',
    role: 'system',
    content: 'Conversation started',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    role: 'assistant',
    content: "Hello! Welcome to our dealership. I'm here to help you find the perfect car. What brings you in today? Are you looking for something specific, or would you like me to help you explore some options?",
    timestamp: new Date().toISOString(),
  },
];

export default function AssistantPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>(initialMessages);
  const [profile, setProfile] = useState<PersonalityProfile | undefined>();
  const [recommendations, setRecommendations] = useState<CarRecommendation[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (content: string) => {
    // Add user message
    const userMessage: ChatMessageType = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      // Update profile based on conversation
      const newProfile: PersonalityProfile = {
        scores: {
          openness: 0.65 + Math.random() * 0.2,
          conscientiousness: 0.75 + Math.random() * 0.15,
          extraversion: 0.50 + Math.random() * 0.2,
          agreeableness: 0.70 + Math.random() * 0.15,
          neuroticism: 0.30 + Math.random() * 0.2,
        },
        confidence: {
          openness: 0.75 + Math.random() * 0.2,
          conscientiousness: 0.80 + Math.random() * 0.15,
          extraversion: 0.70 + Math.random() * 0.2,
          agreeableness: 0.75 + Math.random() * 0.15,
          neuroticism: 0.72 + Math.random() * 0.2,
        },
        timestamp: new Date().toISOString(),
      };
      setProfile(newProfile);

      // Generate response based on content
      let response = '';
      let thoughtTrace;

      if (content.toLowerCase().includes('family') || content.toLowerCase().includes('kids')) {
        response = "Family transportation is so important! Based on what you've shared, I'd recommend looking at our SUV lineup. The Toyota RAV4 Hybrid is excellent for families - it has top safety ratings and great fuel economy. Would you like to hear more about its safety features?";
        thoughtTrace = {
          reasoning_steps: ['Customer mentioned family needs', 'Prioritizing safety and space'],
          personality_signals: ['High agreeableness indicated by family focus', 'Conscientiousness shown through practical considerations'],
        };
        setRecommendations(sampleRecommendations);
      } else if (content.toLowerCase().includes('safe') || content.toLowerCase().includes('reliable')) {
        response = "Safety and reliability are great priorities! I can see you value making informed decisions. Let me show you some vehicles with the highest safety ratings - the IIHS Top Safety Pick+ winners. The Toyota RAV4 and Honda CR-V both earned that distinction.";
        thoughtTrace = {
          reasoning_steps: ['Customer emphasizes safety', 'Showing data-driven recommendations'],
          personality_signals: ['High conscientiousness - values reliability', 'Methodical decision-making approach'],
        };
        setRecommendations(sampleRecommendations);
      } else if (content.toLowerCase().includes('electric') || content.toLowerCase().includes('ev') || content.toLowerCase().includes('tech')) {
        response = "Exciting choice! Electric and hybrid vehicles are the future. The technology has come so far - you get instant torque, lower running costs, and some amazing tech features. Are you interested in full electric, or would a hybrid be a better fit for your driving needs?";
        thoughtTrace = {
          reasoning_steps: ['Customer shows interest in innovation', 'Exploring technology preferences'],
          personality_signals: ['High openness to new experiences', 'Forward-thinking mindset'],
        };
      } else {
        response = "That's helpful to know! To give you the best recommendations, could you tell me a bit more about how you'll primarily use the vehicle? For example, is it mainly for commuting, family activities, or perhaps weekend adventures?";
        thoughtTrace = {
          reasoning_steps: ['Gathering more information', 'Building customer profile'],
          personality_signals: ['Analyzing response patterns'],
        };
      }

      const assistantMessage: ChatMessageType = {
        id: generateId(),
        role: 'assistant',
        content: response,
        timestamp: new Date().toISOString(),
        personality_update: newProfile,
        thought_trace: thoughtTrace,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="h-[calc(100vh-7rem)] flex">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white rounded-xl border border-gray-200 overflow-hidden">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <MessageSquare className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Sales Assistant</h2>
              <p className="text-sm text-gray-500">AI-powered conversation with real-time profiling</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" icon={<Plus className="w-4 h-4" />}>
              New Chat
            </Button>
            <Button variant="ghost" size="sm" icon={<Settings className="w-4 h-4" />}>
              Settings
            </Button>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors lg:hidden"
            >
              {sidebarOpen ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isTyping && (
            <div className="flex gap-3 mb-4">
              <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                <MessageSquare className="w-4 h-4 text-purple-600" />
              </div>
              <div className="px-4 py-3 bg-gray-100 rounded-2xl rounded-bl-md">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <ChatInput onSend={handleSend} disabled={isTyping} />
      </div>

      {/* Sidebar */}
      {sidebarOpen && (
        <ChatSidebar
          profile={profile}
          recommendations={recommendations}
          customerName="Current Customer"
        />
      )}
    </div>
  );
}

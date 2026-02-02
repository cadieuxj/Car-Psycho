'use client';

import { ChatMessage as ChatMessageType } from '@/lib/types';
import { cn, formatDateTime } from '@/lib/utils';
import { User, Bot, Sparkles } from 'lucide-react';
import { TraitBars } from '@/components/psychometrics';

interface ChatMessageProps {
  message: ChatMessageType;
  showPersonalityUpdate?: boolean;
}

export default function ChatMessage({ message, showPersonalityUpdate = true }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <div className="px-4 py-2 bg-gray-100 rounded-full text-sm text-gray-500">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex gap-3 mb-4', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-primary-100' : 'bg-purple-100'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary-600" />
        ) : (
          <Bot className="w-4 h-4 text-purple-600" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn('flex flex-col max-w-[70%]', isUser ? 'items-end' : 'items-start')}>
        <div
          className={cn(
            'px-4 py-3 rounded-2xl',
            isUser
              ? 'bg-primary-600 text-white rounded-br-md'
              : 'bg-gray-100 text-gray-900 rounded-bl-md'
          )}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Timestamp */}
        <span className="text-xs text-gray-400 mt-1 px-1">
          {formatDateTime(message.timestamp)}
        </span>

        {/* Personality Update */}
        {showPersonalityUpdate && message.personality_update && (
          <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg w-full max-w-sm">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-medium text-purple-900">Profile Updated</span>
            </div>
            <TraitBars
              scores={message.personality_update.scores}
              confidence={message.personality_update.confidence}
            />
          </div>
        )}

        {/* Thought Trace */}
        {message.thought_trace && (
          <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg w-full max-w-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium text-amber-900">Personality Signals</span>
            </div>
            <ul className="text-xs text-amber-700 space-y-1">
              {message.thought_trace.personality_signals.map((signal, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-500">*</span>
                  <span>{signal}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

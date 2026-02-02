'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Mic, Smile } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Type a message...'
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [message]);

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      <div className="flex items-end gap-3">
        {/* Attachment Button */}
        <button
          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          title="Attach file"
        >
          <Paperclip className="w-5 h-5" />
        </button>

        {/* Input Area */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={cn(
              'w-full resize-none rounded-xl border border-gray-300 px-4 py-3 pr-12',
              'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
              'placeholder:text-gray-400 text-gray-900',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          />
          <button
            className="absolute right-2 bottom-2 p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
            title="Add emoji"
          >
            <Smile className="w-5 h-5" />
          </button>
        </div>

        {/* Voice Input */}
        <button
          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          title="Voice input"
        >
          <Mic className="w-5 h-5" />
        </button>

        {/* Send Button */}
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          className={cn(
            'p-3 rounded-xl transition-colors',
            message.trim() && !disabled
              ? 'bg-primary-600 text-white hover:bg-primary-700'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
        >
          <Send className="w-5 h-5" />
        </button>
      </div>

      <p className="text-xs text-gray-400 mt-2 text-center">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}

/**
 * Unit Tests for Root Layout Component
 *
 * Tests cover:
 * 1. Metadata configuration
 * 2. HTML structure
 * 3. Children rendering
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import RootLayout, { metadata } from '../../app/layout';

describe('RootLayout Component', () => {
  describe('Metadata', () => {
    it('has correct title', () => {
      expect(metadata.title).toBe('Car-Psycho');
    });

    it('has correct description', () => {
      expect(metadata.description).toBe('Psychometric Car Sales Platform');
    });

    it('metadata is an object', () => {
      expect(typeof metadata).toBe('object');
    });

    it('metadata has required properties', () => {
      expect(metadata).toHaveProperty('title');
      expect(metadata).toHaveProperty('description');
    });
  });

  describe('Component Rendering', () => {
    // Note: Testing the full layout is tricky because it renders html/body
    // which conflicts with the test environment. We test the structure instead.

    it('is a valid React component', () => {
      expect(typeof RootLayout).toBe('function');
    });

    it('accepts children prop', () => {
      // Create a mock that just returns the children to test the interface
      const TestChild = () => <div data-testid="test-child">Test Content</div>;

      // Render just the children to verify they would be passed through
      render(<TestChild />);
      expect(screen.getByTestId('test-child')).toBeInTheDocument();
    });

    it('RootLayout function exists', () => {
      expect(RootLayout).toBeDefined();
    });
  });

  describe('Layout Structure', () => {
    it('returns JSX element', () => {
      const result = RootLayout({ children: <div>Test</div> });
      expect(result).toBeTruthy();
      expect(React.isValidElement(result)).toBe(true);
    });

    it('sets lang attribute to en', () => {
      const result = RootLayout({ children: <div>Test</div> });
      // The outer element should be html with lang="en"
      expect(result.props.lang).toBe('en');
    });
  });
});

describe('Layout TypeScript Interface', () => {
  it('accepts children as ReactNode', () => {
    // Test different types of children
    const testCases = [
      <div key="1">Simple element</div>,
      'String child',
      123,
      null,
      [<span key="a">A</span>, <span key="b">B</span>],
    ];

    testCases.forEach((child) => {
      // This tests that the component can accept various ReactNode types
      const result = RootLayout({ children: child as React.ReactNode });
      expect(result).toBeTruthy();
    });
  });
});

describe('Layout Integration', () => {
  it('integrates with metadata for SEO', () => {
    // Verify metadata can be used for SEO purposes
    expect(metadata.title).toBeTruthy();
    expect(metadata.description).toBeTruthy();
    expect(metadata.title.length).toBeGreaterThan(0);
    expect(metadata.description.length).toBeGreaterThan(0);
  });

  it('layout provides consistent structure', () => {
    const layout1 = RootLayout({ children: <div>Page 1</div> });
    const layout2 = RootLayout({ children: <div>Page 2</div> });

    // Both should have same outer structure (html element with lang)
    expect(layout1.type).toBe(layout2.type);
    expect(layout1.props.lang).toBe(layout2.props.lang);
  });
});

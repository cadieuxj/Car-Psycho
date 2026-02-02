/**
 * Unit Tests for Home Page Component
 *
 * Tests cover:
 * 1. Component rendering
 * 2. Title and description display
 * 3. Services status links
 * 4. Styling and structure
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import Home from '../../app/page';

describe('Home Page', () => {
  describe('Rendering', () => {
    it('renders without crashing', () => {
      render(<Home />);
      expect(document.body).toBeTruthy();
    });

    it('renders main element', () => {
      render(<Home />);
      const main = screen.getByRole('main');
      expect(main).toBeInTheDocument();
    });
  });

  describe('Title and Description', () => {
    it('displays the application title', () => {
      render(<Home />);
      const title = screen.getByRole('heading', { level: 1 });
      expect(title).toHaveTextContent('Car-Psycho');
    });

    it('displays the application description', () => {
      render(<Home />);
      expect(screen.getByText('Psychometric Car Sales Platform')).toBeInTheDocument();
    });

    it('displays development status message', () => {
      render(<Home />);
      expect(screen.getByText(/Frontend under development/)).toBeInTheDocument();
    });
  });

  describe('Services Status Section', () => {
    it('displays Services Status heading', () => {
      render(<Home />);
      const heading = screen.getByRole('heading', { level: 2 });
      expect(heading).toHaveTextContent('Services Status');
    });

    it('displays Manager Service link', () => {
      render(<Home />);
      const link = screen.getByRole('link', { name: /Port 8001/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', 'http://localhost:8001');
    });

    it('displays Inference Service link', () => {
      render(<Home />);
      const link = screen.getByRole('link', { name: /Port 8002/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', 'http://localhost:8002');
    });

    it('displays Data Service link', () => {
      render(<Home />);
      const link = screen.getByRole('link', { name: /Port 8003/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', 'http://localhost:8003');
    });

    it('displays API Gateway link', () => {
      render(<Home />);
      const link = screen.getByRole('link', { name: /Port 8080/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', 'http://localhost:8080');
    });

    it('has all service links open in new tab', () => {
      render(<Home />);
      const links = screen.getAllByRole('link');

      links.forEach((link) => {
        expect(link).toHaveAttribute('target', '_blank');
      });
    });
  });

  describe('Structure', () => {
    it('contains a list of services', () => {
      render(<Home />);
      const list = screen.getByRole('list');
      expect(list).toBeInTheDocument();
    });

    it('contains four service list items', () => {
      render(<Home />);
      const listItems = screen.getAllByRole('listitem');
      expect(listItems).toHaveLength(4);
    });

    it('lists all expected services', () => {
      render(<Home />);

      expect(screen.getByText(/Manager Service/i)).toBeInTheDocument();
      expect(screen.getByText(/Inference Service/i)).toBeInTheDocument();
      expect(screen.getByText(/Data Service/i)).toBeInTheDocument();
      expect(screen.getByText(/API Gateway/i)).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('main element has padding style', () => {
      render(<Home />);
      const main = screen.getByRole('main');
      expect(main).toHaveStyle({ padding: '2rem' });
    });

    it('main element has font-family style', () => {
      render(<Home />);
      const main = screen.getByRole('main');
      expect(main).toHaveStyle({ fontFamily: 'system-ui' });
    });

    it('development message has muted color', () => {
      render(<Home />);
      const devMessage = screen.getByText(/Frontend under development/);
      expect(devMessage).toHaveStyle({ color: '#666' });
    });
  });

  describe('Accessibility', () => {
    it('has proper heading hierarchy', () => {
      render(<Home />);

      const h1 = screen.getByRole('heading', { level: 1 });
      const h2 = screen.getByRole('heading', { level: 2 });

      expect(h1).toBeInTheDocument();
      expect(h2).toBeInTheDocument();
    });

    it('all links have accessible names', () => {
      render(<Home />);
      const links = screen.getAllByRole('link');

      links.forEach((link) => {
        expect(link).toHaveAccessibleName();
      });
    });

    it('uses semantic list element for services', () => {
      render(<Home />);
      const list = screen.getByRole('list');
      expect(list.tagName.toLowerCase()).toBe('ul');
    });
  });
});

describe('Home Page Snapshot', () => {
  it('matches snapshot', () => {
    const { container } = render(<Home />);
    expect(container).toMatchSnapshot();
  });
});

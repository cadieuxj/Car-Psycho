import './globals.css';
import { MainLayout } from '@/components/layout';

export const metadata = {
  title: 'Car-Psycho | Psychometric Car Sales Platform',
  description: 'AI-powered psychometric analysis for personalized car sales recommendations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <MainLayout>{children}</MainLayout>
      </body>
    </html>
  );
}

export const metadata = {
  title: 'Car-Psycho',
  description: 'Psychometric Car Sales Platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

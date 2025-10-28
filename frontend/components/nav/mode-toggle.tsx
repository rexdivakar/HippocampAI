'use client';

import * as React from 'react';
import { useTheme } from 'next-themes';
import { Sun, Moon } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function ModeToggle() {
  const { theme, resolvedTheme, setTheme } = useTheme();
const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => setMounted(true), []);

  // Avoid SSR/CSR mismatch
  if (!mounted) return null;

  const isDark = (theme === 'dark') || (theme === 'system' && resolvedTheme === 'dark');

  return (
    <Button
      className="cursor-pointer"
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
    >
      {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  );
}

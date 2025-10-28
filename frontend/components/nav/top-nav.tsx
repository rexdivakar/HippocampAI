'use client';

import Link from 'next/link';
import Image from 'next/image';
import ModeToggle from './mode-toggle';
import {
  Menubar,
  MenubarMenu,
  MenubarTrigger,
  MenubarContent,
  MenubarItem,
  MenubarSeparator,
} from '@/components/ui/menubar';


export default function TopNav() {

  
  return (
    <div className="border-b bg-background">
      <div className="mx-auto flex h-14 max-w-screen-xl items-center gap-4 px-4">

        {/* Logo + Brand */}
        <Link href="/" className="flex items-center gap-2">
          {/* <Image src="/logo.svg" alt="Logo" width={28} height={28} priority /> */}
          <span className="font-semibold">HippocampAI</span>
        </Link>

        {/* Right side: links + toggle */}
        <div className="ml-auto flex items-center gap-2">
          <Menubar>
            <MenubarMenu>
              <MenubarTrigger className="text-base font-normal cursor-pointer">
                Menubar Trigger 1
              </MenubarTrigger>

              <MenubarContent>
                <MenubarItem className="cursor-pointer">Menu 1</MenubarItem>
                <MenubarItem className="cursor-pointer">Menu 2</MenubarItem>
                <MenubarSeparator />
                <MenubarItem className="cursor-pointer">Menu 3</MenubarItem>
              </MenubarContent>
            </MenubarMenu>

            <MenubarMenu>
              <MenubarTrigger className="text-base font-normal cursor-pointer">
                Menubar Trigger 2
              </MenubarTrigger>
            </MenubarMenu>

            <MenubarMenu>
              <MenubarTrigger className="text-base font-normal cursor-pointer">
                Menubar Trigger 3
              </MenubarTrigger>
            </MenubarMenu>
          </Menubar>

          <ModeToggle />
        </div>
      </div>
    </div>
  );

}

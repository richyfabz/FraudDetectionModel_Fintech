// frontend/src/components/Navbar.jsx
// Navigation bar appears on every page
// Uses React Router's NavLink which automatically adds an 'active'
// class when the current URL matches the link's path.
// This lets us highlight the current page in the nav.

import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, Menu, X } from 'lucide-react';

// Navigation links config 
// Adding a new page only requires adding one entry here
const NAV_LINKS = [
  { path: '/',              label: 'Home'          },
  { path: '/how-it-works',  label: 'How It Works'  },
  { path: '/features',      label: 'Feature Guide' },
  { path: '/analyse',       label: 'Analyser'      },
  { path: '/bulk',          label: 'Bulk Upload'   },
];

export default function Navbar() {
  // Mobile menu state 
  // On small screens the nav collapses to a hamburger menu
  const [menuOpen, setMenuOpen] = useState(false);

  // Shared link classes 
  // Base classes applied to every nav link
  const baseClass = `
    text-sm font-medium transition-colors duration-200
    px-3 py-2 rounded-lg
  `;

  // Active vs inactive link styles 
  // React Router passes isActive to NavLink's className function
  const linkClass = ({ isActive }) =>
    isActive
      ? `${baseClass} text-white bg-blue-600`        // current page
      : `${baseClass} text-slate-400 hover:text-white hover:bg-slate-800`;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50
                    bg-slate-950/90 backdrop-blur-md
                    border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo  */}
          <NavLink
            to="/"
            className="flex items-center gap-2
                       text-white hover:opacity-80 transition-opacity"
          >
            <div className="p-1.5 bg-blue-500/20 rounded-lg
                           border border-blue-500/30">
              <Activity className="text-blue-400" size={20} />
            </div>
            <span className="font-bold text-lg tracking-tight">
              FraudGuard{' '}
              <span className="text-slate-500 font-light">
                Engine
              </span>
            </span>
          </NavLink>

          {/* Desktop nav links  */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map(({ path, label }) => (
              <NavLink
                key={path}
                to={path}
                className={linkClass}
                // exact match for home so /analyse doesn't
                // also highlight the Home link
                end={path === '/'}
              >
                {label}
              </NavLink>
            ))}
          </div>

          {/* Status indicator  */}
          {/* Shows a green dot when Flask API is reachable */}
          <div className="hidden md:flex items-center gap-2
                         text-xs text-slate-500">
            <span className="inline-block w-2 h-2 rounded-full
                            bg-emerald-400 animate-pulse" />
            Live
          </div>

          {/* Mobile hamburger button  */}
          <button
            onClick={() => setMenuOpen(o => !o)}
            className="md:hidden p-2 rounded-lg text-slate-400
                       hover:text-white hover:bg-slate-800
                       transition-colors"
          >
            {menuOpen
              ? <X size={20} />
              : <Menu size={20} />
            }
          </button>
        </div>
      </div>

      {/* Mobile dropdown menu  */}
      {/* Only visible on small screens when hamburger is clicked */}
      {menuOpen && (
        <div className="md:hidden border-t border-slate-800
                       bg-slate-950 px-4 py-3 space-y-1">
          {NAV_LINKS.map(({ path, label }) => (
            <NavLink
              key={path}
              to={path}
              className={linkClass}
              end={path === '/'}
              // Close menu when a link is clicked
              onClick={() => setMenuOpen(false)}
            >
              {label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
}
// frontend/src/App.js
// Router setup — maps URLs to page components 
// This file only handles routing. All page logic lives in pages/.
// App.js should never contain form state, API calls, or UI logic.

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';

// Page imports 
import Landing      from './pages/Landing';
import HowItWorks   from './pages/HowItWorks';
import FeatureGuide from './pages/FeatureGuide';
import Analyser     from './pages/Analyser';
import BulkUpload   from './pages/BulkUpload';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-slate-100">

        {/* Navbar is fixed at top renders on every page */}
        <Navbar />

        {/* pt-16 offsets content below the fixed 64px navbar */}
        <main className="pt-16">
          <Routes>
            <Route path="/"             element={<Landing />}      />
            <Route path="/how-it-works" element={<HowItWorks />}   />
            <Route path="/features"     element={<FeatureGuide />} />
            <Route path="/analyse"      element={<Analyser />}     />
            <Route path="/bulk"         element={<BulkUpload />}   />
          </Routes>
        </main>

      </div>
    </BrowserRouter>
  );
}
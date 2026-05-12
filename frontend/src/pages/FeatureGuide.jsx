// frontend/src/pages/FeatureGuide.jsx
// Feature Guide page 
// Explains what each of the 30 features means, why V1-V28 are
// anonymous PCA components, and which features matter most.
// Feature metadata is fetched live from Flask /api/features.

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import {
  BookOpen, Search, Info,
  TrendingUp, Minus, AlertTriangle
} from 'lucide-react';

// Animation helper 
const fadeUp = (delay = 0) => ({
  initial:    { opacity: 0, y: 24 },
  animate:    { opacity: 1, y: 0  },
  transition: { duration: 0.5, delay }
});

// Importance level visual config 
// Maps importance string to colors and icons for the feature table
const IMPORTANCE_CONFIG = {
  critical: {
    label:  'Critical',
    color:  'text-red-400',
    bg:     'bg-red-500/10',
    border: 'border-red-500/30',
    icon:   AlertTriangle
  },
  high: {
    label:  'High',
    color:  'text-orange-400',
    bg:     'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon:   TrendingUp
  },
  medium: {
    label:  'Medium',
    color:  'text-yellow-400',
    bg:     'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    icon:   TrendingUp
  },
  low: {
    label:  'Low',
    color:  'text-slate-500',
    bg:     'bg-slate-800',
    border: 'border-slate-700',
    icon:   Minus
  }
};

// PCA explanation cards 
const PCA_POINTS = [
  {
    title: 'What is PCA?',
    desc:  'Principal Component Analysis compresses many correlated features into fewer uncorrelated components. Each V component is a mathematical combination of dozens of original bank features — merchant category, location, device, spending history, and more.'
  },
  {
    title: 'Why are they anonymous?',
    desc:  'The Kaggle dataset authors applied PCA before releasing the data to protect cardholder privacy. The original features contained identifiable information card numbers, merchant names, locations. PCA removes this while preserving the statistical patterns that distinguish fraud from legitimate transactions.'
  },
  {
    title: 'Can I interpret V14 = -4.94?',
    desc:  'Not directly. A very negative V14 means this transaction is extreme along the 14th principal component a direction in feature space that happens to be strongly associated with fraud. The actual bank-level meaning of that direction is unknown because the original features were anonymised.'
  },
  {
    title: 'Why keep Time and Amount raw?',
    desc:  'Time and Amount were not anonymised in the original dataset. Isolation Forest ranked Time as the 2nd most important feature for anomaly detection. Dropping them caused the model to miss 4 out of 10 fraud cases in testing. Both are kept in their original form no transformation applied.'
  }
];

export default function FeatureGuide() {
  // State 
  const [features, setFeatures] = useState([]);
  const [loading, setLoading]   = useState(true);
  const [search, setSearch]     = useState('');
  const [filter, setFilter]     = useState('all');

  // Fetch feature metadata from Flask 
  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/features')
      .then(res => {
        setFeatures(res.data.features || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Filter and search logic 
  // Filters by importance level and searches by feature name
  const filtered = features.filter(f => {
    const matchSearch = f.name.toLowerCase()
      .includes(search.toLowerCase());
    const matchFilter = filter === 'all'
      || f.importance === filter;
    return matchSearch && matchFilter;
  });

  // Count features by importance for filter badges 
  const counts = features.reduce((acc, f) => {
    acc[f.importance] = (acc[f.importance] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="max-w-5xl mx-auto
                    px-4 sm:px-6 lg:px-8 py-16">

      {/* Page header  */}
      <motion.div
        className="text-center mb-16"
        {...fadeUp(0)}
      >
        <div className="inline-flex items-center gap-2
                       px-4 py-1.5 bg-purple-500/10
                       border border-purple-500/30 rounded-full
                       text-purple-400 text-sm font-medium mb-6">
          <BookOpen size={14} />
          Feature Reference Guide
        </div>
        <h1 className="text-4xl sm:text-5xl font-black mb-4">
          Understanding the Features
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          All 30 input features explained what V1–V28 represent,
          why Time and Amount are kept raw, and which features the
          model relies on most heavily.
        </p>
      </motion.div>

      {/* PCA explanation section  */}
      <motion.div className="mb-16" {...fadeUp(0.1)}>
        <h2 className="text-2xl font-bold mb-6">
          What Are V1–V28?
        </h2>

        {/* Info banner */}
        <div className="bg-blue-500/5 border border-blue-500/20
                       rounded-xl p-5 mb-6 flex gap-4">
          <Info className="text-blue-400 flex-shrink-0 mt-0.5"
                size={20} />
          <div>
            <p className="text-sm text-slate-300 font-medium mb-1">
              Why are the features anonymous?
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              V1–V28 are PCA-transformed components of the original
              transaction data. The bank anonymised all features
              before releasing the dataset publicly to protect
              cardholder privacy. Each V value is a mathematical
              combination of dozens of original bank features
              merchant details, location, device, spending history.
            </p>
          </div>
        </div>

        {/* PCA explanation cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {PCA_POINTS.map(({ title, desc }) => (
            <div
              key={title}
              className="bg-slate-900 border border-slate-800
                         rounded-xl p-5"
            >
              <h3 className="font-semibold text-white mb-2
                            text-sm">
                {title}
              </h3>
              <p className="text-xs text-slate-400
                           leading-relaxed">
                {desc}
              </p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Feature importance summary  */}
      <motion.div className="mb-10" {...fadeUp(0.2)}>
        <h2 className="text-2xl font-bold mb-6">
          Feature Importance Rankings
        </h2>

        {/* Top 4 critical features highlighted */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          {[
            { name: 'V14', rank: '#1', note: 'Strongest single fraud predictor. Very low values are the clearest fraud signal.' },
            { name: 'V17', rank: '#2', note: 'Second strongest. Captures transaction network anomaly patterns.' },
            { name: 'V4',  rank: '#3', note: 'High values strongly correlate with fraudulent transactions.' },
            { name: 'V12', rank: '#4', note: 'Unusually low values are a reliable fraud indicator.' },
          ].map(({ name, rank, note }) => (
            <div
              key={name}
              className="bg-red-500/5 border border-red-500/20
                         rounded-xl p-4 text-center"
            >
              <span className="text-xs text-red-400 font-semibold
                             uppercase tracking-wider">
                {rank}
              </span>
              <p className="text-3xl font-black text-red-400
                           font-mono my-1">
                {name}
              </p>
              <p className="text-xs text-slate-500
                           leading-tight">
                {note}
              </p>
            </div>
          ))}
        </div>

        {/* Importance determined by message */}
        <p className="text-xs text-slate-500 text-center">
          Importance determined by AutoGluon permutation importance
          measures accuracy drop when each feature is shuffled
          across 5 shuffle sets on 5,000 test rows.
        </p>
      </motion.div>

      {/* Feature table  */}
      <motion.div {...fadeUp(0.3)}>
        <div className="flex flex-col sm:flex-row gap-4
                       mb-6 items-start sm:items-center
                       justify-between">
          <h2 className="text-2xl font-bold">
            All 30 Features
          </h2>

          {/* Search and filter controls */}
          <div className="flex gap-3 w-full sm:w-auto">

            {/* Search box */}
            <div className="relative flex-1 sm:w-48">
              <Search
                size={14}
                className="absolute left-3 top-1/2
                           -translate-y-1/2 text-slate-500"
              />
              <input
                type="text"
                placeholder="Search features..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="w-full bg-slate-800 border
                           border-slate-700 rounded-lg
                           pl-8 pr-3 py-2 text-sm
                           text-slate-100
                           placeholder-slate-600
                           focus:outline-none
                           focus:border-blue-500
                           transition-colors"
              />
            </div>

            {/* Importance filter */}
            <select
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="bg-slate-800 border border-slate-700
                         rounded-lg px-3 py-2 text-sm
                         text-slate-300 focus:outline-none
                         focus:border-blue-500 transition-colors"
            >
              <option value="all">
                All ({features.length})
              </option>
              <option value="critical">
                Critical ({counts.critical || 0})
              </option>
              <option value="high">
                High ({counts.high || 0})
              </option>
              <option value="medium">
                Medium ({counts.medium || 0})
              </option>
              <option value="low">
                Low ({counts.low || 0})
              </option>
            </select>
          </div>
        </div>

        {/* Feature list */}
        <div className="bg-slate-900 border border-slate-800
                       rounded-xl overflow-hidden">

          {/* Table header */}
          <div className="grid grid-cols-12 px-6 py-3
                         bg-slate-800/50 border-b
                         border-slate-800 text-xs font-semibold
                         text-slate-400 uppercase tracking-wider">
            <span className="col-span-2">Feature</span>
            <span className="col-span-2">Type</span>
            <span className="col-span-2">Importance</span>
            <span className="col-span-6">Description</span>
          </div>

          {/* Rows */}
          {loading ? (
            <div className="px-6 py-12 text-center
                           text-slate-500 text-sm">
              Loading feature data from Flask API...
            </div>
          ) : filtered.length === 0 ? (
            <div className="px-6 py-12 text-center
                           text-slate-500 text-sm">
              No features match your search.
            </div>
          ) : (
            filtered.map((feature, i) => {
              const imp = IMPORTANCE_CONFIG[feature.importance]
                || IMPORTANCE_CONFIG.low;
              const ImpIcon = imp.icon;

              return (
                <div
                  key={feature.name}
                  className={`grid grid-cols-12 px-6 py-4
                             items-start
                             ${i < filtered.length - 1
                               ? 'border-b border-slate-800/50'
                               : ''
                             }
                             ${feature.importance === 'critical'
                               ? 'bg-red-500/3'
                               : ''
                             }`}
                >
                  {/* Feature name */}
                  <div className="col-span-2">
                    <span className="font-mono font-bold
                                    text-white text-sm">
                      {feature.name}
                    </span>
                    {feature.kept_raw && (
                      <span className="block text-xs
                                      text-blue-400 mt-0.5">
                        Raw
                      </span>
                    )}
                  </div>

                  {/* Type badge */}
                  <div className="col-span-2">
                    <span className={`text-xs px-2 py-0.5
                      rounded-full font-medium
                      ${feature.type === 'pca'
                        ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30'
                        : 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                      }`}>
                      {feature.type === 'pca' ? 'PCA' : 'Original'}
                    </span>
                  </div>

                  {/* Importance badge */}
                  <div className="col-span-2">
                    <span className={`inline-flex items-center
                      gap-1 text-xs px-2 py-0.5 rounded-full
                      font-medium border
                      ${imp.color} ${imp.bg} ${imp.border}`}>
                      <ImpIcon size={10} />
                      {imp.label}
                    </span>
                  </div>

                  {/* Description */}
                  <div className="col-span-6">
                    <p className="text-xs text-slate-400
                                 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer count */}
        <p className="text-xs text-slate-600 mt-3 text-right">
          Showing {filtered.length} of {features.length} features
        </p>
      </motion.div>

    </div>
  );
}
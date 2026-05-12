// frontend/src/pages/Landing.jsx
// Landing page hero section + architecture overview 
// This is the first page users see. It explains what FraudGuard is,
// shows the key performance metrics, and links to other pages.

import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ShieldCheck, Activity, Brain, Upload,
  BookOpen, ArrowRight, TrendingUp,
  AlertTriangle, CheckCircle, Zap
} from 'lucide-react';

// Animation helper 
// fadeUp animates each section in from below as it enters the screen
// delay staggers the animations so they don't all fire at once
const fadeUp = (delay = 0) => ({
  initial:   { opacity: 0, y: 30 },
  animate:   { opacity: 1, y: 0  },
  transition:{ duration: 0.6, delay }
});

// Performance metrics to display in the stats bar 
const STATS = [
  { label: 'AUC-ROC Score',      value: '0.9610', color: 'text-blue-400'    },
  { label: 'F1-Score',           value: '0.8939', color: 'text-emerald-400' },
  { label: 'Fraud Precision',    value: '97%',    color: 'text-purple-400'  },
  { label: 'Fraud Recall',       value: '82%',    color: 'text-orange-400'  },
  { label: 'Test Transactions',  value: '85,118', color: 'text-slate-300'   },
  { label: 'Fraud Cases Caught', value: '116/142',color: 'text-emerald-400' },
];

// Navigation cards linking to each section 
const CARDS = [
  {
    icon:        Brain,
    title:       'How It Works',
    description: 'Understand the hybrid Isolation Forest + AutoGluon pipeline, model architecture, and how fraud decisions are made.',
    path:        '/how-it-works',
    color:       'text-blue-400',
    border:      'hover:border-blue-500/50',
    bg:          'hover:bg-blue-500/5',
  },
  {
    icon:        BookOpen,
    title:       'Feature Guide',
    description: 'Learn what V1–V28 mean, why Time and Amount were kept raw, and which features matter most for fraud detection.',
    path:        '/features',
    color:       'text-purple-400',
    border:      'hover:border-purple-500/50',
    bg:          'hover:bg-purple-500/5',
  },
  {
    icon:        Activity,
    title:       'Transaction Analyser',
    description: 'Score a single transaction step by step. Load real fraud and legitimate examples from the test set.',
    path:        '/analyse',
    color:       'text-emerald-400',
    border:      'hover:border-emerald-500/50',
    bg:          'hover:bg-emerald-500/5',
  },
  {
    icon:        Upload,
    title:       'Bulk CSV Upload',
    description: 'Upload a CSV file with multiple transactions. Get batch predictions, summary statistics, and downloadable results.',
    path:        '/bulk',
    color:       'text-orange-400',
    border:      'hover:border-orange-500/50',
    bg:          'hover:bg-orange-500/5',
  },
];

// Pipeline steps shown in the architecture section 
const PIPELINE = [
  {
    step:  '01',
    title: 'Raw Transaction',
    desc:  'Time, Amount, V1–V28 features enter the pipeline',
    icon:  Zap,
    color: 'text-slate-400'
  },
  {
    step:  '02',
    title: 'StandardScaler',
    desc:  'All 30 features normalised to the same magnitude',
    icon:  Activity,
    color: 'text-blue-400'
  },
  {
    step:  '03',
    title: 'Isolation Forest',
    desc:  'Unsupervised anomaly score computed from 226k normal transactions',
    icon:  AlertTriangle,
    color: 'text-orange-400'
  },
  {
    step:  '04',
    title: 'AutoGluon Ensemble',
    desc:  'Weighted ensemble of RF, LightGBM, CatBoost scores the 31 features',
    icon:  Brain,
    color: 'text-purple-400'
  },
  {
    step:  '05',
    title: 'Threshold Decision',
    desc:  'Fraud probability compared against tuned threshold of 39.7%',
    icon:  TrendingUp,
    color: 'text-yellow-400'
  },
  {
    step:  '06',
    title: 'Final Decision',
    desc:  'FRAUD or LEGITIMATE with confidence tier and anomaly score',
    icon:  CheckCircle,
    color: 'text-emerald-400'
  },
];

export default function Landing() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

      {/* Hero section  */}
      <motion.div
        className="text-center mb-20"
        {...fadeUp(0)}
      >
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5
                       bg-blue-500/10 border border-blue-500/30
                       rounded-full text-blue-400 text-sm
                       font-medium mb-6">
          <ShieldCheck size={14} />
          Production-Grade Fraud Detection System
        </div>

        {/* Title */}
        <h1 className="text-5xl sm:text-7xl font-black
                       tracking-tight mb-6">
          <span className="text-white">Richy FraudGuard</span>
          <br />
          <span className="bg-gradient-to-r from-blue-400
                          to-purple-400 bg-clip-text
                          text-transparent">
            Engine
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl text-slate-400 max-w-3xl mx-auto mb-10
                     leading-relaxed">
          A hybrid unsupervised + supervised ML system combining
          Isolation Forest anomaly detection with AutoGluon's
          automated ensemble learning to detect credit card fraud
          with 96.1% AUC-ROC across 85,118 test transactions.
        </p>

        {/* CTA buttons */}
        <div className="flex flex-col sm:flex-row gap-4
                       justify-center">
          <Link
            to="/analyse"
            className="flex items-center justify-center gap-2
                       px-8 py-4 bg-blue-600 hover:bg-blue-500
                       rounded-xl font-bold text-white
                       transition-all text-lg"
          >
            <Activity size={20} />
            Analyse a Transaction
          </Link>
          <Link
            to="/how-it-works"
            className="flex items-center justify-center gap-2
                       px-8 py-4 bg-slate-800 hover:bg-slate-700
                       border border-slate-600 rounded-xl
                       font-semibold text-slate-300
                       hover:text-white transition-all text-lg"
          >
            How It Works
            <ArrowRight size={20} />
          </Link>
        </div>
      </motion.div>

      {/* Stats bar  */}
      <motion.div
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6
                   gap-4 mb-20"
        {...fadeUp(0.2)}
      >
        {STATS.map(({ label, value, color }) => (
          <div
            key={label}
            className="bg-slate-900 border border-slate-800
                       rounded-xl p-4 text-center"
          >
            <p className={`text-2xl font-black font-mono
                          mb-1 ${color}`}>
              {value}
            </p>
            <p className="text-xs text-slate-500 leading-tight">
              {label}
            </p>
          </div>
        ))}
      </motion.div>

      {/* Pipeline overview  */}
      <motion.div className="mb-20" {...fadeUp(0.3)}>
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold mb-3">
            How Fraud Gets Detected
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Every transaction passes through a 6-stage hybrid pipeline
            combining unsupervised anomaly detection with supervised
            ensemble classification.
          </p>
        </div>

        {/* Pipeline steps grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2
                       lg:grid-cols-3 gap-4">
          {PIPELINE.map(({ step, title, desc, icon: Icon, color }) => (
            <div
              key={step}
              className="bg-slate-900 border border-slate-800
                         rounded-xl p-6 relative overflow-hidden"
            >
              {/* Large step number in background */}
              <span className="absolute top-3 right-4 text-6xl
                              font-black text-slate-800 select-none">
                {step}
              </span>

              {/* Icon and title */}
              <div className={`mb-3 ${color}`}>
                <Icon size={24} />
              </div>
              <h3 className="font-bold text-white mb-2">{title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                {desc}
              </p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Navigation cards  */}
      <motion.div {...fadeUp(0.4)}>
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold mb-3">
            Explore the System
          </h2>
          <p className="text-slate-400">
            Four sections covering everything from model architecture
            to live transaction scoring.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {CARDS.map(({
            icon: Icon, title, description, path, color, border, bg
          }) => (
            <Link
              key={path}
              to={path}
              className={`block bg-slate-900 border border-slate-800
                         rounded-2xl p-6 transition-all duration-200
                         ${border} ${bg} group`}
            >
              {/* Icon */}
              <div className={`inline-flex p-3 rounded-xl
                              bg-slate-800 mb-4 ${color}
                              group-hover:scale-110 transition-transform`}>
                <Icon size={24} />
              </div>

              {/* Content */}
              <h3 className="text-lg font-bold text-white mb-2">
                {title}
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">
                {description}
              </p>

              {/* Arrow */}
              <div className={`flex items-center gap-1 text-sm
                              font-medium ${color}`}>
                Explore
                <ArrowRight
                  size={14}
                  className="group-hover:translate-x-1
                             transition-transform"
                />
              </div>
            </Link>
          ))}
        </div>
      </motion.div>

      {/* Footer note */}
      <motion.p
        className="text-center text-xs text-slate-700 mt-16"
        {...fadeUp(0.5)}
      >
        Built on 284,807 real European credit card transactions ·
        Kaggle Credit Card Fraud Detection Dataset ·
        AutoGluon 1.5.0 · Isolation Forest · Python 3.10
      </motion.p>

    </div>
  );
}
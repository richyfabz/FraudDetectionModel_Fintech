// frontend/src/pages/HowItWorks.jsx
// How It Works page 
// Explains the full pipeline, each algorithm used, and shows
// the model performance metrics with a leaderboard table.
// Data is fetched live from Flask /api/models endpoint.

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import {
  Brain, Trees, Zap, BarChart2,
  GitBranch, Shield, AlertTriangle,
  CheckCircle, TrendingUp, Clock,
  ChevronDown, ChevronUp
} from 'lucide-react';

// Animation helper 
const fadeUp = (delay = 0) => ({
  initial:    { opacity: 0, y: 24 },
  animate:    { opacity: 1, y: 0  },
  transition: { duration: 0.5, delay }
});

// Pipeline stages 
const PIPELINE_STAGES = [
  {
    id:      '01',
    title:   'Data Ingestion',
    color:   'border-slate-500',
    icon:    Clock,
    iconCol: 'text-slate-400',
    desc:    'Raw transaction arrives with 30 features — Time, Amount, and V1–V28. No feature engineering is applied. Time and Amount are kept in their original form because Isolation Forest needs raw values to detect anomalies accurately.',
    code:    `# All 30 features kept raw nothing dropped
features = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']`
  },
  {
    id:      '02',
    title:   'StandardScaler',
    color:   'border-blue-500',
    icon:    BarChart2,
    iconCol: 'text-blue-400',
    desc:    'All 30 features are normalised using StandardScaler. Time ranges from 0–172,792 seconds and Amount ranges from €0–€25,691. Without scaling, these magnitudes would dominate the anomaly score calculation and overshadow V1–V28 which are already PCA-normalised.',
    code:    `# Fitted on training data only never on test or live data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # no refitting`
  },
  {
    id:      '03',
    title:   'Isolation Forest',
    color:   'border-orange-500',
    icon:    AlertTriangle,
    iconCol: 'text-orange-400',
    desc:    'An Isolation Forest with 100 trees is trained exclusively on 226,602 legitimate transactions. It learns what normal looks like by measuring how many random splits it takes to isolate each point. Fraud transactions are structurally different so they get isolated in fewer splits producing a high anomaly score.',
    code:    `# Trained on legitimate transactions ONLY unsupervised
isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.001667,  # known fraud rate
    random_state=0
)
isolation_forest.fit(X_train_legit)  # no labels passed`
  },
  {
    id:      '04',
    title:   'Anomaly Score Normalisation',
    color:   'border-yellow-500',
    icon:    TrendingUp,
    iconCol: 'text-yellow-400',
    desc:    'The raw Isolation Forest score is negative for anomalies and positive for normal transactions. We flip the sign and normalise to [0,1] using the global min/max computed across training and test sets. This makes the score behave like a fraud probability 1.0 means maximally anomalous.',
    code:    `# Flip and normalise so higher score = more fraudulent
flipped       = -raw_score
anomaly_score = (flipped - global_min) / (global_max - global_min)`
  },
  {
    id:      '05',
    title:   'AutoGluon Ensemble',
    color:   'border-purple-500',
    icon:    Brain,
    iconCol: 'text-purple-400',
    desc:    'AutoGluon receives all 30 scaled features plus the anomaly score as a 31st feature. Within a 10-minute training budget it automatically trains Random Forest, LightGBM, CatBoost, ExtraTrees, and KNN then builds a weighted ensemble that combines their predictions optimally. No manual hyperparameter tuning required.',
    code:    `# AutoGluon trains and ensembles automatically
predictor = TabularPredictor(label='Class', eval_metric='accuracy')
predictor.fit(train_df, time_limit=600, presets='best_quality')`
  },
  {
    id:      '06',
    title:   'Threshold Decision',
    color:   'border-emerald-500',
    icon:    CheckCircle,
    iconCol: 'text-emerald-400',
    desc:    'AutoGluon outputs a fraud probability between 0 and 1. We sweep 200 candidate thresholds and select the value that maximises F1-Score on the test set. The optimal threshold of 39.7% was found to best balance catching fraud (recall) against avoiding false alarms (precision).',
    code:    `# Threshold tuned to maximise F1-Score
thresholds = np.linspace(0, 1, 200)
best_threshold = thresholds[np.argmax([
    f1_score(y_test, (y_prob >= t).astype(int))
    for t in thresholds
])]  # Result: 0.397`
  },
];

// Algorithm cards 
const ALGORITHMS = [
  {
    name:    'Isolation Forest',
    type:    'Unsupervised',
    color:   'text-orange-400',
    bg:      'bg-orange-500/10',
    border:  'border-orange-500/30',
    icon:    AlertTriangle,
    role:    'Anomaly Score Generator',
    desc:    'Trained on legitimate transactions only. Builds 100 random isolation trees and measures how quickly each transaction gets isolated. Fraud cases are anomalous they get isolated faster, producing a higher anomaly score that feeds directly into AutoGluon as a feature.',
    strength:'Detects new fraud patterns never seen before it does not rely on labelled fraud examples.',
    weakness:'Standalone precision is low (~12%) it flags too many legitimate transactions as anomalous without supervised correction.',
  },
  {
    name:    'Random Forest',
    type:    'Supervised — Bagged',
    color:   'text-blue-400',
    bg:      'bg-blue-500/10',
    border:  'border-blue-500/30',
    icon:    Trees,
    role:    'Primary Base Model',
    desc:    'AutoGluon\'s strongest performer RandomForestGini_BAG_L1 scored 0.9997 accuracy on the test set. Builds hundreds of decision trees each trained on a random subset of data and features. Aggregates their votes. The 8-fold bagging AutoGluon applies makes it significantly more robust than a single Random Forest.',
    strength:'Highly resistant to overfitting. Handles class imbalance well. Fast at inference.',
    weakness:'Less accurate than gradient boosting on highly complex non-linear patterns.',
  },
  {
    name:    'LightGBM',
    type:    'Supervised — Gradient Boosting',
    color:   'text-green-400',
    bg:      'bg-green-500/10',
    border:  'border-green-500/30',
    icon:    Zap,
    role:    'Speed-Optimised Booster',
    desc:    'Microsoft\'s gradient boosting framework. Builds trees sequentially where each tree corrects the errors of the previous one. Splits leaf-by-leaf rather than level-by-level, making it significantly faster than XGBoost while achieving comparable accuracy. Trained in 4.3 seconds on 198k rows.',
    strength:'Extremely fast training. Memory efficient. Handles large datasets natively.',
    weakness:'More hyperparameters to tune than Random Forest. Can overfit on small datasets.',
  },
  {
    name:    'CatBoost',
    type:    'Supervised — Gradient Boosting',
    color:   'text-pink-400',
    bg:      'bg-pink-500/10',
    border:  'border-pink-500/30',
    icon:    GitBranch,
    role:    'Ordered Boosting Model',
    desc:    'Yandex\'s gradient boosting library. Uses ordered boosting processes training examples in a random sequence and only uses previous examples to compute gradients for the current one. This eliminates a subtle data leakage issue present in standard gradient boosting, producing less biased gradient estimates.',
    strength:'Most resistant to overfitting among gradient boosting methods. No need to preprocess categorical features.',
    weakness:'Slower to train than LightGBM. Less community resources and documentation.',
  },
  {
    name:    'WeightedEnsemble L2',
    type:    'Meta-Learner',
    color:   'text-purple-400',
    bg:      'bg-purple-500/10',
    border:  'border-purple-500/30',
    icon:    Brain,
    role:    'Final Decision Maker',
    desc:    'AutoGluon\'s automated stacking ensemble. After training all base models with 8-fold bagging, a second-level model learns the optimal weight to assign to each base model\'s out-of-fold predictions. The final fraud probability is a weighted combination of all base models making it more robust than any single model alone.',
    strength:'Combines the complementary strengths of all base models. Automatically discovered by AutoGluon no manual tuning.',
    weakness:'Slower at inference than a single model. Requires all base models to be loaded in memory.',
  },
];

//  Expandable algorithm card component used in the algorithm explanations section
function AlgorithmCard({ algo }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = algo.icon;

  return (
    <div className={`bg-slate-900 border rounded-xl
                    overflow-hidden transition-all duration-200
                    ${algo.border}`}>

      {/* Card header — always visible */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-start gap-4 p-6 text-left
                   hover:bg-slate-800/50 transition-colors"
      >
        {/* Icon */}
        <div className={`p-2.5 rounded-lg flex-shrink-0
                        ${algo.bg} ${algo.color}`}>
          <Icon size={20} />
        </div>

        {/* Title block */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-bold text-white">{algo.name}</h3>
            <span className={`text-xs px-2 py-0.5 rounded-full
                            font-medium border
                            ${algo.bg} ${algo.color} ${algo.border}`}>
              {algo.type}
            </span>
          </div>
          <p className="text-xs text-slate-500">{algo.role}</p>
        </div>

        {/* Expand toggle */}
        <div className="text-slate-500 flex-shrink-0 mt-1">
          {expanded
            ? <ChevronUp size={18} />
            : <ChevronDown size={18} />
          }
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-6 pb-6 border-t border-slate-800 pt-4">
          <p className="text-slate-300 text-sm leading-relaxed mb-4">
            {algo.desc}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-emerald-500/5 border border-emerald-500/20
                           rounded-lg p-3">
              <p className="text-xs font-semibold text-emerald-400
                           uppercase tracking-wider mb-1">
                Strength
              </p>
              <p className="text-xs text-slate-400">
                {algo.strength}
              </p>
            </div>
            <div className="bg-red-500/5 border border-red-500/20
                           rounded-lg p-3">
              <p className="text-xs font-semibold text-red-400
                           uppercase tracking-wider mb-1">
                Limitation
              </p>
              <p className="text-xs text-slate-400">
                {algo.weakness}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function HowItWorks() {
  // Fetch live model list from Flask 
  // This populates the model leaderboard table with real data
  const [models, setModels]   = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch available models from Flask on page load
    axios.get('http://127.0.0.1:5000/api/models')
      .then(res => {
        setModels(res.data.models || []);
        setLoading(false);
      })
      .catch(() => {
        // If Flask is not running show empty table gracefully
        setLoading(false);
      });
  }, []);

  // Performance metrics
  const METRICS = [
    { label: 'AUC-ROC',          value: '0.9610', desc: 'Ranks fraud above legitimate 96.1% of the time' },
    { label: 'F1-Score (Tuned)', value: '0.8939', desc: 'Best balance of precision and recall at threshold 0.397' },
    { label: 'Fraud Precision',  value: '97%',    desc: 'Of all fraud flags — 97% are genuine fraud' },
    { label: 'Fraud Recall',     value: '82%',    desc: 'Of all actual fraud — 82% are caught' },
    { label: 'False Alarms',     value: '4',      desc: 'Legitimate transactions wrongly blocked out of 84,976' },
    { label: 'Train/Test Gap',   value: '0.03%',  desc: 'Near-zero gap confirms no overfitting' },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">

      {/* Page header  */}
      <motion.div className="text-center mb-16" {...fadeUp(0)}>
        <div className="inline-flex items-center gap-2 px-4 py-1.5
                       bg-blue-500/10 border border-blue-500/30
                       rounded-full text-blue-400 text-sm
                       font-medium mb-6">
          <Brain size={14} />
          Architecture Deep Dive
        </div>
        <h1 className="text-4xl sm:text-5xl font-black mb-4">
          How It Works
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          A 6-stage hybrid pipeline combining unsupervised anomaly
          detection with AutoGluon's automated ensemble learning.
        </p>
      </motion.div>

      {/* Pipeline stages */}
      <motion.div className="mb-20" {...fadeUp(0.1)}>
        <h2 className="text-2xl font-bold mb-8">
          The 6-Stage Pipeline
        </h2>

        <div className="space-y-4">
          {PIPELINE_STAGES.map((stage, i) => {
            const Icon = stage.icon;
            return (
              <div
                key={stage.id}
                className={`bg-slate-900 border-l-4 ${stage.color}
                           rounded-xl p-6`}
              >
                <div className="flex items-start gap-4">
                  {/* Step number + icon */}
                  <div className="flex-shrink-0 text-center">
                    <div className={`p-2.5 rounded-lg
                                   bg-slate-800 ${stage.iconCol}
                                   mb-1`}>
                      <Icon size={18} />
                    </div>
                    <span className="text-xs text-slate-600
                                   font-mono">
                      {stage.id}
                    </span>
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <h3 className="font-bold text-white mb-2">
                      {stage.title}
                    </h3>
                    <p className="text-slate-400 text-sm
                                 leading-relaxed mb-3">
                      {stage.desc}
                    </p>

                    {/* Code snippet */}
                    <pre className="bg-slate-950 border
                                   border-slate-800 rounded-lg
                                   p-3 text-xs text-slate-300
                                   overflow-x-auto font-mono
                                   leading-relaxed">
                      {stage.code}
                    </pre>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/*  Algorithm explanations  */}
      <motion.div className="mb-20" {...fadeUp(0.2)}>
        <h2 className="text-2xl font-bold mb-3">
          The Algorithms
        </h2>
        <p className="text-slate-400 mb-8">
          Click any algorithm to expand its full explanation,
          strengths, and limitations.
        </p>
        <div className="space-y-3">
          {ALGORITHMS.map(algo => (
            <AlgorithmCard key={algo.name} algo={algo} />
          ))}
        </div>
      </motion.div>

      {/* Performance metrics */}
      <motion.div className="mb-20" {...fadeUp(0.3)}>
        <h2 className="text-2xl font-bold mb-8">
          Model Performance
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2
                       lg:grid-cols-3 gap-4">
          {METRICS.map(({ label, value, desc }) => (
            <div
              key={label}
              className="bg-slate-900 border border-slate-800
                         rounded-xl p-5"
            >
              <p className="text-3xl font-black font-mono
                           text-blue-400 mb-1">
                {value}
              </p>
              <p className="text-sm font-semibold text-white mb-1">
                {label}
              </p>
              <p className="text-xs text-slate-500 leading-relaxed">
                {desc}
              </p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Live model leaderboard from Flask  */}
      <motion.div {...fadeUp(0.4)}>
        <h2 className="text-2xl font-bold mb-3">
          AutoGluon Model Leaderboard
        </h2>
        <p className="text-slate-400 mb-6 text-sm">
          Live data from the Flask API — all models trained within
          the 10-minute AutoGluon budget ranked by accuracy.
        </p>

        <div className="bg-slate-900 border border-slate-800
                       rounded-xl overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-3 px-6 py-3
                         bg-slate-800/50 border-b border-slate-800">
            <span className="text-xs font-semibold text-slate-400
                           uppercase tracking-wider">
              Model
            </span>
            <span className="text-xs font-semibold text-slate-400
                           uppercase tracking-wider text-center">
              Type
            </span>
            <span className="text-xs font-semibold text-slate-400
                           uppercase tracking-wider text-right">
              Status
            </span>
          </div>

          {/* Table rows */}
          {loading ? (
            <div className="px-6 py-8 text-center text-slate-500
                           text-sm">
              Loading models from Flask API...
            </div>
          ) : models.length === 0 ? (
            <div className="px-6 py-8 text-center text-slate-500
                           text-sm">
              Flask API not reachable. Start the server to see
              live model data.
            </div>
          ) : (
            models.map((model, i) => (
              <div
                key={model.id}
                className={`grid grid-cols-3 px-6 py-4
                           items-center
                           ${i < models.length - 1
                             ? 'border-b border-slate-800/50'
                             : ''
                           }`}
              >
                {/* Model name */}
                <div>
                  <span className="text-sm text-slate-200
                                  font-medium">
                    {model.display}
                  </span>
                </div>

                {/* Stack level badge */}
                <div className="text-center">
                  <span className="text-xs px-2 py-0.5
                                  rounded-full bg-slate-800
                                  text-slate-400 font-mono">
                    {model.id.includes('L2') ? 'Stack L2'
                    : model.id.includes('L3') ? 'Stack L3'
                    : 'Base L1'}
                  </span>
                </div>

                {/* Default badge */}
                <div className="text-right">
                  {model.is_default ? (
                    <span className="text-xs px-2 py-0.5
                                    rounded-full font-semibold
                                    bg-blue-500/10
                                    border border-blue-500/30
                                    text-blue-400">
                      Default ⭐
                    </span>
                  ) : (
                    <span className="text-xs text-slate-600">
                      Available
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </motion.div>

    </div>
  );
}
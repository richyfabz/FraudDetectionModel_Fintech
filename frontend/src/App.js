import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import {
  ShieldCheck, AlertOctagon, ArrowRight,
  ArrowLeft, Loader2, RefreshCw, Activity
} from 'lucide-react';

// Feature groups 
// We split the 30 features into logical steps so the form
// doesn't overwhelm the user with all fields at once
const STEP_CONFIG = [
  {
    title:    'Core Transaction Details',
    subtitle: 'Enter the primary transaction information',
    fields:   ['Time', 'Amount']
  },
  {
    title:    'PCA Components — V1 to V7',
    subtitle: 'Principal component features from the original transaction data',
    fields:   ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
  },
  {
    title:    'PCA Components — V8 to V14',
    subtitle: 'V14 is the strongest fraud predictor in this model',
    fields:   ['V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14']
  },
  {
    title:    'PCA Components — V15 to V28',
    subtitle: 'Final set of principal components',
    fields:   [
      'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
      'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
    ]
  }
];

// Known fraud transaction for demo 
const FRAUD_EXAMPLE = {
  Time: 406.0, Amount: 229.15,
  V1: -3.043541, V2: -3.157307, V3:  1.088463,
  V4:  2.288282, V5:  1.359805, V6: -1.064823,
  V7: -3.216816, V8:  0.963958, V9: -4.498295,
  V10: -1.903324, V11: 1.453888, V12: -2.833819,
  V13: -0.764650, V14: -4.941888, V15: 0.392831,
  V16: -1.140788, V17: -2.459499, V18: -1.637940,
  V19: 0.774543,  V20: 0.034249,  V21: 0.641673,
  V22: 0.339485,  V23: -0.182899, V24: 0.551707,
  V25: 0.239334,  V26: 0.406271,  V27: 0.222938,
  V28: 0.027510
};

// Default empty form state
const DEFAULT_FORM = {
  Time: '', Amount: '',
  ...Object.fromEntries(
    Array.from({ length: 28 }, (_, i) => [`V${i + 1}`, ''])
  )
};

// Animation variants 
const slideVariants = {
  hidden:  { opacity: 0, x: 60 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.35 } },
  exit:    { opacity: 0, x: -60, transition: { duration: 0.25 } }
};

const resultVariants = {
  hidden:  { opacity: 0, scale: 0.92 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.4 } }
};

// Confidence tier config 
const TIER_CONFIG = {
  HIGH:   { color: 'text-red-400',    bg: 'bg-red-500/10',    border: 'border-red-500/30'    },
  MEDIUM: { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30' },
  REVIEW: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30' },
  CLEAR:  { color: 'text-emerald-400',bg: 'bg-emerald-500/10',border: 'border-emerald-500/30'}
};


export default function App() {
  const [step, setStep]         = useState(0);
  const [formData, setFormData] = useState(DEFAULT_FORM);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);

  const totalSteps = STEP_CONFIG.length;
  const isLastStep = step === totalSteps - 1;

  // Input handler
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Load fraud example 
  const loadExample = () => {
    const stringified = Object.fromEntries(
      Object.entries(FRAUD_EXAMPLE).map(([k, v]) => [k, String(v)])
    );
    setFormData(stringified);
  };

  // Navigate steps
  const next = () => {
    if (isLastStep) {
      submit();
    } else {
      setStep(s => s + 1);
    }
  };
  const back = () => setStep(s => s - 1);

  // Reset everything to start a new prediction 
  const reset = () => {
    setStep(0);
    setFormData(DEFAULT_FORM);
    setResult(null);
    setError(null);
  };

  // Submit to Flask API and get prediction 
  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const numericData = Object.fromEntries(
        Object.entries(formData).map(([k, v]) => [k, parseFloat(v) || 0])
      );
      const response = await axios.post(
        'http://127.0.0.1:5000/api/predict',
        numericData
      );
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.error ||
        'Could not connect to the prediction engine. Is Flask running?'
      );
    } finally {
      setLoading(false);
    }
  };

  //  Progress bar width 
  const progress = ((step + 1) / totalSteps) * 100;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100
                    flex flex-col items-center justify-center p-4">

      {/* Header */}
      <header className="mb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="p-2 bg-blue-500/20 rounded-xl border border-blue-500/30">
            <Activity className="text-blue-400" size={28} />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            FraudGuard <span className="text-slate-500 font-light">Engine</span>
          </h1>
        </div>
        <p className="text-slate-400 text-sm">
          Hybrid Isolation Forest + AutoGluon Architecture
        </p>
      </header>

      {/* Main card */}
      <div className="w-full max-w-2xl bg-slate-900 border border-slate-700/50
                      rounded-2xl shadow-2xl overflow-hidden">

        {/* Progress bar */}
        {!result && (
          <div className="h-1 bg-slate-800">
            <motion.div
              className="h-full bg-blue-500"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}

        <div className="p-8">
          <AnimatePresence mode="wait">

            {/* Result view  */}
            {result ? (
              <motion.div
                key="result"
                variants={resultVariants}
                initial="hidden"
                animate="visible"
                className="text-center"
              >
                {/* Decision icon */}
                <div className={`inline-flex p-5 rounded-full mb-6
                  ${result.decision === 'FRAUD'
                    ? 'bg-red-500/10 border border-red-500/30'
                    : 'bg-emerald-500/10 border border-emerald-500/30'
                  }`}>
                  {result.decision === 'FRAUD'
                    ? <AlertOctagon className="text-red-400" size={52} />
                    : <ShieldCheck  className="text-emerald-400" size={52} />
                  }
                </div>

                {/* Decision label */}
                <h2 className={`text-5xl font-black tracking-tight mb-2
                  ${result.decision === 'FRAUD'
                    ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                  {result.decision}
                </h2>

                {/* Confidence tier badge */}
                {(() => {
                  const tier = TIER_CONFIG[result.confidence_tier];
                  return (
                    <span className={`inline-block px-4 py-1 rounded-full
                      text-sm font-semibold border mb-8
                      ${tier.color} ${tier.bg} ${tier.border}`}>
                      {result.confidence_tier} CONFIDENCE
                    </span>
                  );
                })()}

                {/* Metrics grid */}
                <div className="grid grid-cols-3 gap-4 mb-8">
                  {[
                    {
                      label: 'Fraud Probability',
                      value: `${result.fraud_probability}%`,
                      color: result.decision === 'FRAUD'
                        ? 'text-red-400' : 'text-emerald-400'
                    },
                    {
                      label: 'Anomaly Score',
                      value: `${result.anomaly_score}%`,
                      color: 'text-blue-400'
                    },
                    {
                      label: 'Threshold',
                      value: `${result.threshold_used}%`,
                      color: 'text-slate-400'
                    }
                  ].map(({ label, value, color }) => (
                    <div key={label}
                      className="bg-slate-800/60 border border-slate-700/50
                                 rounded-xl p-4">
                      <p className="text-xs text-slate-500 uppercase
                                   tracking-wider mb-1">{label}</p>
                      <p className={`text-2xl font-mono font-bold ${color}`}>
                        {value}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Key feature values */}
                <div className="bg-slate-800/40 border border-slate-700/50
                                rounded-xl p-4 mb-8 text-left">
                  <p className="text-xs text-slate-500 uppercase
                               tracking-wider mb-3">
                    Key Feature Values
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {['V14', 'V17', 'V4', 'V12'].map(f => (
                      <div key={f}
                        className="flex justify-between items-center
                                   bg-slate-900/60 rounded-lg px-3 py-2">
                        <span className="text-slate-400 text-sm
                                        font-mono">{f}</span>
                        <span className="text-slate-200 text-sm font-mono">
                          {parseFloat(result.raw_input[f]).toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Reset button */}
                <button
                  onClick={reset}
                  className="flex items-center gap-2 mx-auto px-6 py-3
                             bg-slate-800 hover:bg-slate-700 border
                             border-slate-600 rounded-xl transition-all
                             text-slate-300 hover:text-white"
                >
                  <RefreshCw size={16} />
                  New Prediction
                </button>
              </motion.div>

            ) : (
              /* Form steps */
              <motion.div
                key={`step-${step}`}
                variants={slideVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                {/* Step header */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-blue-400 font-semibold
                                    uppercase tracking-wider">
                      Step {step + 1} of {totalSteps}
                    </span>
                    {step === 0 && (
                      <button
                        onClick={loadExample}
                        className="text-xs text-slate-400 hover:text-blue-400
                                   transition-colors underline underline-offset-2"
                      >
                        Load fraud example
                      </button>
                    )}
                  </div>
                  <h2 className="text-xl font-bold">
                    {STEP_CONFIG[step].title}
                  </h2>
                  <p className="text-slate-400 text-sm mt-1">
                    {STEP_CONFIG[step].subtitle}
                  </p>
                </div>

                {/* Input fields */}
                <div className="grid grid-cols-2 gap-3 mb-8">
                  {STEP_CONFIG[step].fields.map(field => (
                    <div key={field}>
                      <label className="block text-xs text-slate-400
                                       mb-1 font-medium">
                        {field === 'Time'
                          ? 'Time (seconds)'
                          : field === 'Amount'
                          ? 'Amount (€)'
                          : field}
                      </label>
                      <input
                        type="number"
                        name={field}
                        value={formData[field]}
                        onChange={handleChange}
                        step="any"
                        placeholder="0.000000"
                        className="w-full bg-slate-800 border border-slate-700
                                   rounded-lg px-3 py-2.5 text-sm font-mono
                                   text-slate-100 placeholder-slate-600
                                   focus:outline-none focus:border-blue-500
                                   focus:ring-1 focus:ring-blue-500/50
                                   transition-colors"
                      />
                    </div>
                  ))}
                </div>

                {/* Error display */}
                {error && (
                  <div className="mb-4 px-4 py-3 bg-red-500/10
                                 border border-red-500/30 rounded-xl
                                 text-red-400 text-sm">
                    {error}
                  </div>
                )}

                {/* Navigation buttons */}
                <div className="flex gap-3">
                  {step > 0 && (
                    <button
                      onClick={back}
                      className="flex items-center gap-2 px-5 py-3
                                 bg-slate-800 hover:bg-slate-700 border
                                 border-slate-600 rounded-xl transition-all
                                 text-slate-300"
                    >
                      <ArrowLeft size={16} />
                      Back
                    </button>
                  )}

                  <button
                    onClick={next}
                    disabled={loading}
                    className="flex-1 flex items-center justify-center
                               gap-2 py-3 bg-blue-600 hover:bg-blue-500
                               disabled:opacity-50 disabled:cursor-not-allowed
                               rounded-xl font-semibold transition-all"
                  >
                    {loading ? (
                      <>
                        <Loader2 size={18} className="animate-spin" />
                        Analysing...
                      </>
                    ) : isLastStep ? (
                      <>
                        <ShieldCheck size={18} />
                        Analyse Transaction
                      </>
                    ) : (
                      <>
                        Continue
                        <ArrowRight size={18} />
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>
      </div>

      {/* Footer  */}
      <p className="mt-6 text-xs text-slate-600">
        AutoGluon WeightedEnsemble · Isolation Forest · AUC-ROC 0.9610
      </p>
    </div>
  );
}
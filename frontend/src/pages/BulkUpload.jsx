// frontend/src/pages/BulkUpload.jsx
// Bulk CSV Upload page
// Accepts any CSV file, sends it to Flask /api/predict/batch,
// displays colour-coded results table with summary statistics,
// and allows downloading results as a new CSV file.

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import {
  Upload, FileText, CheckCircle,
  AlertOctagon, Download, RefreshCw,
  Loader2, Info, ChevronDown
} from 'lucide-react';

// Animation helper
const fadeUp = (delay = 0) => ({
  initial:    { opacity: 0, y: 20 },
  animate:    { opacity: 1, y: 0  },
  transition: { duration: 0.4, delay }
});

export default function BulkUpload() {
  // State 
  const [file, setFile]         = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [results, setResults]   = useState(null);
  const [summary, setSummary]   = useState(null);
  const [error, setError]       = useState(null);
  const [page, setPage]         = useState(1);

  // File input ref triggers hidden input on button click 
  const fileInputRef = useRef(null);

  // Rows per page for results table pagination 
  const ROWS_PER_PAGE = 20;

  // Drag and drop handlers 
  const handleDragOver = e => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => setDragging(false);

  const handleDrop = e => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped?.name.endsWith('.csv')) {
      setFile(dropped);
      setError(null);
    } else {
      setError('Please upload a CSV file.');
    }
  };

  const handleFileSelect = e => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
    }
  };

  // Upload and score the CSV file 
  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setSummary(null);

    try {
      // FormData is required for file uploads
      // Flask reads it via request.files['file']
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(
        'http://127.0.0.1:5000/api/predict/batch',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      setResults(response.data.results);
      setSummary(response.data.summary);
      setPage(1);
    } catch (err) {
      setError(
        err.response?.data?.error ||
        'Upload failed. Is Flask running?'
      );
    } finally {
      setLoading(false);
    }
  };

  // Reset everything to upload a new file 
  const reset = () => {
    setFile(null);
    setResults(null);
    setSummary(null);
    setError(null);
    setPage(1);
  };

  // Download results as CSV 
  // Converts the results array to a CSV string and triggers
  // a browser download without any server involvement
  const downloadCSV = () => {
    if (!results) return;

    // Build CSV header row
    const headers = [
      'Row', 'Decision', 'Fraud_Probability_%',
      'Anomaly_Score_%', 'Confidence_Tier',
      'Threshold_%', 'True_Label', 'Correct'
    ];

    // Build data rows
    const rows = results.map(r => [
      r.row_number,
      r.decision,
      r.fraud_probability,
      r.anomaly_score,
      r.confidence_tier,
      r.threshold_used,
      r.true_label ?? 'Unknown',
      r.correct === null ? 'N/A'
        : r.correct ? 'Yes' : 'No'
    ]);

    // Join into CSV string
    const csv = [headers, ...rows]
      .map(row => row.join(','))
      .join('\n');

    // Create a temporary download link and click it
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'fraudguard_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Paginated results slice 
  const totalPages = results
    ? Math.ceil(results.length / ROWS_PER_PAGE)
    : 0;
  const pageResults = results
    ? results.slice(
        (page - 1) * ROWS_PER_PAGE,
        page * ROWS_PER_PAGE
      )
    : [];

  return (
    <div className="max-w-5xl mx-auto
                    px-4 sm:px-6 lg:px-8 py-16">

      {/* Page header  */}
      <motion.div
        className="text-center mb-12"
        {...fadeUp(0)}
      >
        <div className="inline-flex items-center gap-2
                       px-4 py-1.5 bg-orange-500/10
                       border border-orange-500/30 rounded-full
                       text-orange-400 text-sm font-medium mb-6">
          <Upload size={14} />
          Batch Transaction Scoring
        </div>
        <h1 className="text-4xl sm:text-5xl font-black mb-4">
          Bulk CSV Upload
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Upload any CSV file with transaction data. The system
          accepts any column format and fills missing features
          with neutral values automatically.
        </p>
      </motion.div>

      {/* CSV format info  */}
      <motion.div className="mb-8" {...fadeUp(0.1)}>
        <div className="bg-blue-500/5 border border-blue-500/20
                       rounded-xl p-5 flex gap-4">
          <Info className="text-blue-400 flex-shrink-0 mt-0.5"
                size={18} />
          <div>
            <p className="text-sm font-medium text-slate-300 mb-1">
              Accepted CSV formats
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              The system accepts any CSV. Columns matching
              Time, Amount, V1–V28 are used directly.
              Missing columns are filled with 0 automatically.
              If your CSV has a Class column (0=legitimate,
              1=fraud), predictions are checked against it and
              accuracy is shown. Maximum 500 rows per upload.
            </p>
          </div>
        </div>
      </motion.div>

      {!results ? (
        /* Upload zone  */
        <motion.div {...fadeUp(0.2)}>

          {/* Drag and drop area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl
                       p-16 text-center cursor-pointer
                       transition-all duration-200
                       ${dragging
                         ? 'border-blue-500 bg-blue-500/10'
                         : file
                         ? 'border-emerald-500 bg-emerald-500/5'
                         : 'border-slate-700 bg-slate-900 hover:border-slate-500 hover:bg-slate-800/50'
                       }`}
          >
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
            />

            {file ? (
              /* File selected state */
              <>
                <FileText
                  className="text-emerald-400 mx-auto mb-4"
                  size={48}
                />
                <p className="text-lg font-semibold text-white mb-1">
                  {file.name}
                </p>
                <p className="text-slate-400 text-sm">
                  {(file.size / 1024).toFixed(1)} KB —
                  click to change
                </p>
              </>
            ) : (
              /* Empty state */
              <>
                <Upload
                  className={`mx-auto mb-4 transition-colors
                    ${dragging
                      ? 'text-blue-400' : 'text-slate-500'
                    }`}
                  size={48}
                />
                <p className="text-lg font-semibold text-slate-300
                             mb-2">
                  Drop your CSV here
                </p>
                <p className="text-slate-500 text-sm">
                  or click to browse files
                </p>
              </>
            )}
          </div>

          {/* Error message */}
          {error && (
            <div className="mt-4 px-4 py-3 bg-red-500/10
                           border border-red-500/30 rounded-xl
                           text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Upload button */}
          <button
            onClick={upload}
            disabled={!file || loading}
            className="mt-6 w-full flex items-center
                       justify-center gap-2 py-4
                       bg-orange-600 hover:bg-orange-500
                       disabled:opacity-40
                       disabled:cursor-not-allowed
                       rounded-xl font-bold text-white
                       text-lg transition-all"
          >
            {loading ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                Analysing transactions...
              </>
            ) : (
              <>
                <Upload size={20} />
                Analyse CSV
              </>
            )}
          </button>
        </motion.div>

      ) : (
        /* Results view  */
        <motion.div
          initial={{ opacity:0 }}
          animate={{ opacity:1 }}
          transition={{ duration:0.4 }}
        >
          {/* Summary stats */}
          {summary && (
            <div className="grid grid-cols-2 sm:grid-cols-4
                           gap-4 mb-8">
              {[
                {
                  label: 'Total Analysed',
                  value: summary.total,
                  color: 'text-slate-200'
                },
                {
                  label: 'Fraud Detected',
                  value: summary.fraud,
                  color: 'text-red-400'
                },
                {
                  label: 'Fraud Rate',
                  value: `${summary.fraud_rate}%`,
                  color: 'text-orange-400'
                },
                {
                  label: summary.accuracy !== null
                    ? 'Accuracy'
                    : 'Avg Fraud Prob',
                  value: summary.accuracy !== null
                    ? `${summary.accuracy}%`
                    : `${summary.avg_prob}%`,
                  color: 'text-emerald-400'
                }
              ].map(({ label, value, color }) => (
                <div key={label}
                  className="bg-slate-900 border
                             border-slate-800 rounded-xl p-5
                             text-center">
                  <p className={`text-3xl font-black
                               font-mono mb-1 ${color}`}>
                    {value}
                  </p>
                  <p className="text-xs text-slate-500">
                    {label}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Truncation warning */}
          {summary?.truncated && (
            <div className="mb-6 px-4 py-3 bg-yellow-500/10
                           border border-yellow-500/30
                           rounded-xl text-yellow-400 text-sm">
              File was truncated to 500 rows for performance.
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-3 mb-6">
            <button
              onClick={downloadCSV}
              className="flex items-center gap-2 px-5 py-2.5
                         bg-slate-800 hover:bg-slate-700
                         border border-slate-600 rounded-xl
                         text-slate-300 hover:text-white
                         transition-all text-sm font-medium"
            >
              <Download size={16} />
              Download Results CSV
            </button>
            <button
              onClick={reset}
              className="flex items-center gap-2 px-5 py-2.5
                         bg-slate-800 hover:bg-slate-700
                         border border-slate-600 rounded-xl
                         text-slate-300 hover:text-white
                         transition-all text-sm font-medium"
            >
              <RefreshCw size={16} />
              Upload New File
            </button>
          </div>

          {/* Results table */}
          <div className="bg-slate-900 border border-slate-800
                         rounded-xl overflow-hidden">

            {/* Table header */}
            <div className="grid grid-cols-12 px-6 py-3
                           bg-slate-800/50 border-b
                           border-slate-800 text-xs font-semibold
                           text-slate-400 uppercase tracking-wider">
              <span className="col-span-1">#</span>
              <span className="col-span-3">Decision</span>
              <span className="col-span-2">Fraud Prob</span>
              <span className="col-span-2">Anomaly</span>
              <span className="col-span-2">Tier</span>
              <span className="col-span-2">Correct</span>
            </div>

            {/* Table rows */}
            {pageResults.map((r, i) => (
              <div
                key={r.row_number}
                className={`grid grid-cols-12 px-6 py-3
                           items-center text-sm
                           ${i < pageResults.length - 1
                             ? 'border-b border-slate-800/50'
                             : ''
                           }
                           ${r.decision === 'FRAUD'
                             ? 'bg-red-500/3' : ''
                           }`}
              >
                {/* Row number */}
                <span className="col-span-1 text-slate-500
                                text-xs font-mono">
                  {r.row_number}
                </span>

                {/* Decision */}
                <div className="col-span-3 flex items-center gap-2">
                  {r.decision === 'FRAUD'
                    ? <AlertOctagon
                        size={14}
                        className="text-red-400 flex-shrink-0"
                      />
                    : <CheckCircle
                        size={14}
                        className="text-emerald-400 flex-shrink-0"
                      />
                  }
                  <span className={`font-semibold text-xs
                    ${r.decision === 'FRAUD'
                      ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                    {r.decision}
                  </span>
                </div>

                {/* Fraud probability */}
                <span className="col-span-2 font-mono text-xs
                                text-slate-300">
                  {r.fraud_probability}%
                </span>

                {/* Anomaly score */}
                <span className="col-span-2 font-mono text-xs
                                text-blue-400">
                  {r.anomaly_score}%
                </span>

                {/* Confidence tier */}
                <span className={`col-span-2 text-xs font-medium
                  ${r.confidence_tier === 'HIGH'   ? 'text-red-400'
                  : r.confidence_tier === 'MEDIUM' ? 'text-orange-400'
                  : r.confidence_tier === 'REVIEW' ? 'text-yellow-400'
                  : 'text-emerald-400'}`}>
                  {r.confidence_tier}
                </span>

                {/* Correct indicator */}
                <span className={`col-span-2 text-xs
                  ${r.correct === true  ? 'text-emerald-400'
                  : r.correct === false ? 'text-red-400'
                  : 'text-slate-600'}`}>
                  {r.correct === null ? '—'
                  : r.correct ? '✓ Yes' : '✗ No'}
                </span>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between
                           mt-4">
              <p className="text-xs text-slate-500">
                Page {page} of {totalPages} —
                {' '}{results.length} total rows
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p-1))}
                  disabled={page === 1}
                  className="px-3 py-1.5 bg-slate-800
                             border border-slate-700 rounded-lg
                             text-xs text-slate-300
                             disabled:opacity-40
                             hover:bg-slate-700 transition-colors"
                >
                  Previous
                </button>
                <button
                  onClick={() =>
                    setPage(p => Math.min(totalPages, p+1))
                  }
                  disabled={page === totalPages}
                  className="px-3 py-1.5 bg-slate-800
                             border border-slate-700 rounded-lg
                             text-xs text-slate-300
                             disabled:opacity-40
                             hover:bg-slate-700 transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
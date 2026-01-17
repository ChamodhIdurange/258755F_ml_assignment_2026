import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const FIELD_GROUPS = [
  {
    title: 'Role & Workload',
    description: 'Work context that often correlates with churn.',
    fields: ['Department', 'Overtime']
  },
  {
    title: 'Career & Satisfaction',
    description: 'Signals of growth, motivation, and engagement.',
    fields: ['Promotion_Gap', 'Job_Satisfaction']
  },
  {
    title: 'Security & Market',
    description: 'External pressure and perceived stability.',
    fields: ['AI_Automation_Risk', 'Recent_Layoffs', 'Job_Security', 'Market_Demand']
  }
];

const FIELD_DEFS = {
  Department: {
    label: 'Department',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: [
      'IT / Software Engineering',
      'HR, Sales & Marketing',
      'Finance',
      'Other'
    ]
  },
  Overtime: {
    label: 'Average monthly overtime',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['0 hours', '1-10 hours', '11-20 hours', '20+ hours'],
    helper: 'Includes after-hours work and weekend work.'
  },
  Promotion_Gap: {
    label: 'Years since last promotion/title change',
    type: 'number',
    required: true,
    min: 0,
    max: 50,
    step: 0.1,
    placeholder: 'e.g., 2.5',
    helper: 'Numeric values only (decimals allowed).'
  },
  Job_Satisfaction: {
    label: 'Job satisfaction',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
  },
  AI_Automation_Risk: {
    label: 'AI/automation risk',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    helper: 'How likely your role could be impacted by AI/automation.'
  },
  Recent_Layoffs: {
    label: 'Recent layoffs (last 6 months)',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['Yes', 'No']
  },
  Job_Security: {
    label: 'Job security',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['Very Unstable', 'Unstable', 'Medium', 'Secure', 'Very Secure']
  },
  Market_Demand: {
    label: 'Ease of finding a similar role elsewhere',
    type: 'select',
    required: true,
    placeholder: 'Select…',
    options: ['Very Easy', 'Easy', 'Neutral', 'Difficult']
  }
};

const INITIAL_FORM = Object.keys(FIELD_DEFS).reduce((acc, key) => {
  acc[key] = '';
  return acc;
}, {});

function clamp01(n) {
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function riskLabel(leaveProbability) {
  const p = clamp01(leaveProbability);
  if (p >= 0.75) return { label: 'Very High Risk', tone: 'danger' };
  if (p >= 0.55) return { label: 'High Risk', tone: 'danger' };
  if (p >= 0.40) return { label: 'Moderate Risk', tone: 'warning' };
  if (p >= 0.25) return { label: 'Low Risk', tone: 'success' };
  return { label: 'Very Low Risk', tone: 'success' };
}

function App() {
  const [formData, setFormData] = useState(INITIAL_FORM);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const filledCount = Object.keys(FIELD_DEFS).filter((k) => String(formData[k]).trim() !== '').length;
  const totalCount = Object.keys(FIELD_DEFS).length;
  const completion = totalCount ? filledCount / totalCount : 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Convert Promotion_Gap to number
      const submitData = {
        ...formData,
        Promotion_Gap: parseFloat(formData.Promotion_Gap) || 0
      };

      const response = await axios.post(`${API_URL}/api/predict`, submitData, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 10000 // 10 second timeout
      });
      setPrediction(response.data);
    } catch (err) {
      let errorMessage = 'An error occurred';
      
      if (err.code === 'ECONNREFUSED' || err.message === 'Network Error') {
        errorMessage = `Cannot connect to backend server at ${API_URL}. Please make sure the backend is running on port 5001.`;
      } else if (err.response) {
        // Server responded with error status
        errorMessage = err.response.data?.error || `Server error: ${err.response.status}`;
      } else if (err.request) {
        // Request was made but no response received
        errorMessage = `No response from server at ${API_URL}. Please check if the backend is running.`;
      } else {
        errorMessage = err.message || 'An error occurred';
      }
      
      console.error('Prediction error:', err);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM);
    setPrediction(null);
    setError(null);
  };

  const leaveProb = prediction?.probability?.leave ?? 0;
  const stayProb = prediction?.probability?.stay ?? 0;
  const confidence = prediction?.confidence ?? 0;
  const risk = riskLabel(leaveProb);

  return (
    <div className="Page">
      <div className="Bg" aria-hidden="true" />

      <div className="Shell">
        <header className="Topbar">
          <div className="Brand">
            <div className="BrandMark" aria-hidden="true" />
            <div className="BrandText">
              <div className="BrandTitle">Employee Attrition Predictor</div>
              <div className="BrandSubtitle">Machine-learning powered risk estimation</div>
            </div>
          </div>

          <div className="TopbarMeta">
            <div className="MetaStat">
              <div className="MetaLabel">Form completion</div>
              <div className="MetaValue">{filledCount}/{totalCount}</div>
              <div className="MiniBar" aria-hidden="true">
                <div className="MiniBarFill" style={{ width: `${completion * 100}%` }} />
              </div>
            </div>
          </div>
        </header>

        <main className="Grid">
          <section className="Card">
            <div className="CardHeader">
              <div>
                <h2 className="CardTitle">Employee profile</h2>
                <p className="CardHint">Fill the fields below to generate an attrition risk estimate.</p>
              </div>
              <div className="CardActions">
                <button type="button" className="Btn BtnGhost" onClick={handleReset} disabled={loading}>
                  Reset
                </button>
              </div>
            </div>

            <form className="Form" onSubmit={handleSubmit}>
              {FIELD_GROUPS.map((group) => (
                <fieldset key={group.title} className="Fieldset" disabled={loading}>
                  <legend className="Legend">
                    <div className="LegendTitle">{group.title}</div>
                    <div className="LegendSubtitle">{group.description}</div>
                  </legend>

                  <div className="FieldGrid">
                    {group.fields.map((name) => {
                      const def = FIELD_DEFS[name];
                      const id = `field-${name}`;
                      const value = formData[name];

                      return (
                        <div key={name} className="Field">
                          <label className="Label" htmlFor={id}>
                            {def.label}
                            {def.required ? <span className="Req"> *</span> : null}
                          </label>

                          {def.type === 'select' ? (
                            <div className="ControlWrap">
                              <select
                                id={id}
                                name={name}
                                className="Control"
                                value={value}
                                onChange={handleChange}
                                required={def.required}
                              >
                                <option value="">{def.placeholder || 'Select…'}</option>
                                {def.options.map((opt) => (
                                  <option key={opt} value={opt}>
                                    {opt}
                                  </option>
                                ))}
                              </select>
                              <span className="SelectChevron" aria-hidden="true">▾</span>
                            </div>
                          ) : (
                            <input
                              id={id}
                              name={name}
                              className="Control"
                              type="number"
                              inputMode="decimal"
                              value={value}
                              onChange={handleChange}
                              required={def.required}
                              min={def.min}
                              max={def.max}
                              step={def.step}
                              placeholder={def.placeholder}
                            />
                          )}

                          {def.helper ? <div className="Help">{def.helper}</div> : null}
                        </div>
                      );
                    })}
                  </div>
                </fieldset>
              ))}

              <div className="FormFooter">
                <button
                  type="submit"
                  className="Btn BtnPrimary"
                  disabled={loading || filledCount !== totalCount}
                  title={filledCount !== totalCount ? 'Please complete all required fields.' : undefined}
                >
                  {loading ? (
                    <span className="BtnInline">
                      <span className="Spinner" aria-hidden="true" />
                      Predicting…
                    </span>
                  ) : (
                    'Predict attrition risk'
                  )}
                </button>
              </div>
            </form>
          </section>

          <aside className="Card CardSticky" aria-live="polite">
            <div className="CardHeader">
              <div>
                <h2 className="CardTitle">Result</h2>
                <p className="CardHint">Your prediction will appear here once you submit the form.</p>
              </div>
            </div>

            {error && (
              <div className="Alert AlertError" role="alert">
                <div className="AlertTitle">Couldn’t fetch prediction</div>
                <div className="AlertBody">{error}</div>
              </div>
            )}

            {!error && !prediction && (
              <div className="EmptyState">
                <div className="EmptyIcon" aria-hidden="true" />
                <div className="EmptyTitle">No prediction yet</div>
                <div className="EmptyBody">Complete the form and click “Predict attrition risk”.</div>
              </div>
            )}

            {prediction && (
              <div className="Result">
                <div className="ResultTop">
                  <div className={`RiskPill RiskPill--${risk.tone}`}>
                    {risk.label}
                  </div>
                  <div className="Subtle">
                    {Math.round(confidence * 100)}% model confidence
                  </div>
                </div>

                <div className="GaugeRow">
                  <div
                    className="Gauge"
                    style={{ '--p': `${Math.round(clamp01(leaveProb) * 100)}` }}
                    aria-label={`Leave probability ${Math.round(clamp01(leaveProb) * 100)} percent`}
                  >
                    <div className="GaugeInner">
                      <div className="GaugeValue">{Math.round(clamp01(leaveProb) * 100)}%</div>
                      <div className="GaugeLabel">Leave probability</div>
                    </div>
                  </div>

                  <div className="SplitBars">
                    <div className="SplitItem">
                      <div className="SplitHeader">
                        <span>Stay</span>
                        <span className="SplitPct">{Math.round(clamp01(stayProb) * 100)}%</span>
                      </div>
                      <div className="SplitTrack" aria-hidden="true">
                        <div className="SplitFill SplitFill--stay" style={{ width: `${clamp01(stayProb) * 100}%` }} />
                      </div>
                    </div>

                    <div className="SplitItem">
                      <div className="SplitHeader">
                        <span>Leave</span>
                        <span className="SplitPct">{Math.round(clamp01(leaveProb) * 100)}%</span>
                      </div>
                      <div className="SplitTrack" aria-hidden="true">
                        <div className="SplitFill SplitFill--leave" style={{ width: `${clamp01(leaveProb) * 100}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="PredictionLine">
                  <div className="PredictionLabel">Model prediction</div>
                  <div className="PredictionValue">{prediction.prediction_label}</div>
                </div>

                {prediction.feature_importance && (
                  <div className="Insight">
                    <div className="InsightTitle">Top drivers (feature importance)</div>
                    <div className="InsightBody">
                      {(() => {
                        const entries = Object.entries(prediction.feature_importance).sort((a, b) => b[1] - a[1]);
                        const max = Math.max(...entries.map(([, v]) => v), 1);
                        return (
                          <div className="ImportanceList">
                            {entries.map(([feature, importance]) => (
                              <div key={feature} className="ImportanceItem">
                                <div className="ImportanceName">{feature.replace(/_/g, ' ')}</div>
                                <div className="ImportanceTrack" aria-hidden="true">
                                  <div
                                    className="ImportanceFill"
                                    style={{ width: `${(importance / max) * 100}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                )}

                <div className="Footnote">
                  This tool provides a statistical estimate. Use it as decision support, not as a sole source of truth.
                </div>
              </div>
            )}
          </aside>
        </main>

        <footer className="Footer">
          <div className="FooterText">
            API: <span className="Mono">{API_URL}</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;


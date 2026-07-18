import React, { useState } from 'react';

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    age_years: 45, gender: 1, height: 165, weight: 70,
    ap_hi: 120, ap_lo: 80, cholesterol: 1, gluc: 1,
    smoke: 0, alco: 0, active: 1
  });

  // Auto-detects if you are testing locally or running on a live server
  const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://cardio-risk-app-ghi0.onrender.com'; // Replace this later when you host the backend

  const handleNumChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handleCheckChange = (e) => {
    const { name, checked } = e.target;
    setFormData(prev => ({ ...prev, [name]: checked ? 1 : 0 }));
  };

  const handleInferenceRun = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Connection Failed! Make sure your backend terminal has uvicorn running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', fontFamily: 'system-ui, sans-serif' }}>
      <nav style={{ background: '#0f172a', padding: '16px 24px', borderBottom: '1px solid #1e293b' }}>
        <h1 style={{ color: '#fff', margin: 0, fontSize: '18px', fontWeight: 'bold' }}>
          🩺 Cardiovascular Decision Support Dashboard
        </h1>
      </nav>

      <div style={{ padding: '24px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '450px 1fr', gap: '24px', maxWidth: '1200px', margin: '0 auto' }}>
          
          {/* Patient Form Component */}
          <form onSubmit={handleInferenceRun} style={{ background: '#fff', padding: '24px', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <h3 style={{ marginTop: 0, marginBottom: '16px', color: '#1e293b', borderBottom: '1px solid #f1f5f9', paddingBottom: '8px' }}>Patient Vitals</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
              <label style={labelStyle}>Age (Years)
                <input type="number" name="age_years" value={formData.age_years} onChange={handleNumChange} style={inputStyle} />
              </label>
              <label style={labelStyle}>Gender
                <select name="gender" value={formData.gender} onChange={handleNumChange} style={inputStyle}>
                  <option value={1}>Female</option>
                  <option value={2}>Male</option>
                </select>
              </label>
              <label style={labelStyle}>Height (cm)
                <input type="number" name="height" value={formData.height} onChange={handleNumChange} style={inputStyle} />
              </label>
              <label style={labelStyle}>Weight (kg)
                <input type="number" name="weight" value={formData.weight} onChange={handleNumChange} style={inputStyle} />
              </label>
              <label style={labelStyle}>Systolic BP (ap_hi)
                <input type="number" name="ap_hi" value={formData.ap_hi} onChange={handleNumChange} style={inputStyle} />
              </label>
              <label style={labelStyle}>Diastolic BP (ap_lo)
                <input type="number" name="ap_lo" value={formData.ap_lo} onChange={handleNumChange} style={inputStyle} />
              </label>
              <label style={labelStyle}>Cholesterol
                <select name="cholesterol" value={formData.cholesterol} onChange={handleNumChange} style={inputStyle}>
                  <option value={1}>Normal</option>
                  <option value={2}>Elevated</option>
                  <option value={3}>Critical</option>
                </select>
              </label>
              <label style={labelStyle}>Glucose
                <select name="gluc" value={formData.gluc} onChange={handleNumChange} style={inputStyle}>
                  <option value={1}>Normal</option>
                  <option value={2}>Elevated</option>
                  <option value={3}>Critical</option>
                </select>
              </label>
            </div>

            <div style={{ display: 'flex', gap: '20px', margin: '20px 0' }}>
              <label><input type="checkbox" name="smoke" checked={formData.smoke === 1} onChange={handleCheckChange} /> Smoker</label>
              <label><input type="checkbox" name="alco" checked={formData.alco === 1} onChange={handleCheckChange} /> Alcohol Use</label>
              <label><input type="checkbox" name="active" checked={formData.active === 1} onChange={handleCheckChange} /> Active</label>
            </div>

            <button type="submit" disabled={loading} style={{ width: '100%', padding: '12px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer' }}>
              {loading ? 'Processing Model Inference...' : 'Calculate Risk Score'}
            </button>
          </form>

          {/* Analytics Output Dashboard Component */}
          <div style={{ background: '#fff', padding: '24px', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ marginTop: 0, marginBottom: '16px', color: '#1e293b', borderBottom: '1px solid #f1f5f9', paddingBottom: '8px' }}>Analysis Output</h3>
            
            {!result ? (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', border: '2px dashed #e2e8f0', borderRadius: '8px', padding: '20px', textAlign: 'center' }}>
                Awaiting patient diagnostic inputs to engage the neural model infrastructure.
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px', borderRadius: '8px', background: result.high_risk_classification ? '#fee2e2' : '#dcfce7', marginBottom: '20px' }}>
                  <div>
                    <h4 style={{ margin: 0, color: result.high_risk_classification ? '#991b1b' : '#166534', fontSize: '16px' }}>
                      {result.high_risk_classification ? 'CRITICAL RISK CLASSIFIED' : 'LOW RISK ASSESSMENT'}
                    </h4>
                    <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#475569' }}>Calculated BMI: <strong>{result.calculated_bmi}</strong></p>
                  </div>
                  <div style={{ fontSize: '42px', fontWeight: '800', color: result.high_risk_classification ? '#991b1b' : '#166534' }}>{result.risk_score}%</div>
                </div>

                <div style={{ width: '100%', height: '10px', background: '#f1f5f9', borderRadius: '10px', overflow: 'hidden', marginBottom: '24px' }}>
                  <div style={{ width: `${result.risk_score}%`, height: '100%', background: result.risk_score > 50 ? '#ef4444' : '#22c55e', transition: 'width 0.4s ease' }} />
                </div>

                <h5 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#334155' }}>Primary Risk Drivers (SHAP Mapping Analysis):</h5>
                <ul style={{ paddingLeft: '20px', margin: 0, color: '#475569', fontSize: '14px', lineHeight: '1.6' }}>
                  {result.primary_risk_drivers && result.primary_risk_drivers.map((driver, idx) => (
                    <li key={idx} style={{ color: result.high_risk_classification && driver !== 'Vitals fall within normal tolerances.' ? '#b91c1c' : '#475569' }}>
                      {driver}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

const labelStyle = { display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '14px', color: '#475569', fontWeight: '500' };
const inputStyle = { padding: '10px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '14px', boxSizing: 'border-box', width: '100%' };

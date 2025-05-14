import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const ImportFHIR = ({ accessToken }) => {
  const [importResults, setImportResults] = useState([]);
  const [patientIds, setPatientIds] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    if (!accessToken) {
      navigate('/login');
      return;
    }

    axios.get('http://127.0.0.1:8000/import_local_files_to_meldrx', {
      headers: { Authorization: `Bearer ${accessToken}` }
    })
      .then(response => {
        setImportResults(response.data.import_results || []);
        const ids = response.data.import_results
          ?.filter(r => r.status === "success")
          .map(r => r.patientId) || [];
        setPatientIds([...new Set(ids)]);
      })
      .catch(error => {
        if (error.response?.status === 401) {
          sessionStorage.removeItem('accessToken');
          navigate('/login');
        }
        console.error('Import error:', error);
      });
  }, [accessToken, navigate]);

  const viewTimeline = (patientId) => {
    navigate(`/timeline/${patientId}`);
  };

  return (
    <div>
      <h2>Import FHIR Bundles</h2>
      {importResults.length > 0 ? (
        <ul>
          {importResults.map((result, index) => (
            <li key={index}>
              {result.file}: {result.status === "success" ? `Imported (ID: ${result.patientId})` : `Failed - ${result.error || 'Unknown error'}`}
            </li>
          ))}
        </ul>
      ) : (
        <p>Importing...</p>
      )}
      {patientIds.length > 0 && (
        <div>
          <h3>View Timelines</h3>
          {patientIds.map((id, index) => (
            <button key={index} onClick={() => viewTimeline(id)}>
              View Timeline for Patient {id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImportFHIR;
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, AppBar, Toolbar, CircularProgress } from '@mui/material';
import Login from './components/Login';
import PatientsList from './components/PatientList';
import PractitionerNetwork from './components/PractitionerNetwork';
import Timeline from './components/Timeline';
import axios from 'axios';

// Callback component to handle OAuth redirection
const Callback = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');

    if (code && state) {
      setLoading(true);
      axios.get(`http://127.0.0.1:8000/callback?code=${code}&state=${state}`)
        .then(response => {
          // Extract the access_token string from the response
          const accessToken = response.data.access_token;
          if (accessToken) {
            console.log('Callback received access token:', {
              tokenSnippet: accessToken.substring(0, 20) + '...',
              tokenLength: accessToken.length,
              tokenFormat: typeof accessToken,
              isBearerToken: accessToken.startsWith('eyJhbGciOi'), // Check if JWT
              tokenParts: accessToken.split('.').length, // Verify JWT structure
              scopes: accessToken.split('.')[1] ? JSON.parse(atob(accessToken.split('.')[1]))['scope'] || 'No scopes' : 'Cannot decode'
            });
            // Store only the access_token string in sessionStorage
            sessionStorage.setItem('accessToken', accessToken);
            // Send the token to the parent window
            window.opener.postMessage({ type: 'SET_ACCESS_TOKEN', accessToken }, '*');
            // Redirect to the patients list page
            navigate('/patients', { replace: true });
          } else {
            throw new Error('No access_token received in response');
          }
          setLoading(false);
        })
        .catch(err => {
          console.error('Callback error:', {
            error: err,
            message: err.message,
            response: err.response ? {
              status: err.response.status,
              statusText: err.response.statusText,
              data: err.response.data,
              headers: err.response.headers
            } : null,
            request: err.request ? {
              method: err.request.method,
              url: err.request.url,
              headers: err.request.getAllResponseHeaders(),
              requestTime: new Date().toISOString()
            } : null,
            config: err.config
          });
          setError('Failed to authenticate: ' + (err.response?.data?.detail || err.message || err.message));
          setLoading(false);
        });
    } else {
      setError('Invalid callback parameters: code or state missing');
      setLoading(false);
    }
  }, [location, navigate]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 4, bgcolor: 'white', borderRadius: 2, boxShadow: 1, maxWidth: 600, margin: '0 auto', mt: 4 }}>
        <Typography variant="h4" color="error" gutterBottom>
          Error
        </Typography>
        <Typography>{error}</Typography>
        <Typography variant="body2" sx={{ mt: 2 }}>
          Please try logging in again or contact support if the issue persists.
        </Typography>
      </Box>
    );
  }

  return null; // This component will redirect, so no UI is needed
};

function App() {
  const [accessToken, setAccessToken] = useState(sessionStorage.getItem('accessToken') || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle message from Callback window to update access token
  const handleMessage = (event) => {
    if (event.data.type === 'SET_ACCESS_TOKEN' && event.data.accessToken) {
      const newAccessToken = event.data.accessToken;
      sessionStorage.setItem('accessToken', newAccessToken);
      setAccessToken(newAccessToken); // Update state with the new token
      console.log('Access token updated in App via message:', {
        tokenSnippet: newAccessToken.substring(0, 20) + '...',
        tokenLength: newAccessToken.length,
        tokenFormat: typeof newAccessToken,
        isBearerToken: newAccessToken.startsWith('eyJhbGciOi'),
        tokenParts: newAccessToken.split('.').length
      });
    }
  };

  // Track the access token on every render with detailed information
  useEffect(() => {
    console.log('Current access token in App:', {
      token: accessToken,
      tokenSnippet: accessToken ? accessToken.substring(0, 20) + '...' : 'null',
      storedInSession: sessionStorage.getItem('accessToken'),
      tokenFormat: typeof accessToken,
      isBearerToken: accessToken ? accessToken.startsWith('eyJhbGciOi') : false,
      tokenParts: accessToken ? accessToken.split('.').length : 0,
      scopes: accessToken ? (JSON.parse(atob(accessToken.split('.')[1]))['scope'] || 'No scopes') : 'No token'
    });

    // Add event listener for messages
    window.addEventListener('message', handleMessage);

    // Cleanup event listener on unmount
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [accessToken]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 4, bgcolor: 'white', borderRadius: 2, boxShadow: 1, maxWidth: 600, margin: '0 auto', mt: 4 }}>
        <Typography variant="h4" color="error" gutterBottom>
          Error
        </Typography>
        <Typography>{error}</Typography>
        <Typography variant="body2" sx={{ mt: 2 }}>
          Please try logging in again or contact support if the issue persists.
        </Typography>
      </Box>
    );
  }

  return (
    <Router>
      <AppBar position="static" sx={{ bgcolor: '#1976d2' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: 'white' }}>
            MeldRx Doctor Referral
          </Typography>
          {accessToken && (
            <Typography variant="body1" sx={{ color: 'white', mr: 2 }}>
              Logged in
            </Typography>
          )}
        </Toolbar>
      </AppBar>
      <Box sx={{ p: 2 }}>
        <Routes>
          <Route path="/callback" element={<Callback />} />
          <Route path="/login" element={<Login />} />
          <Route path="/patients" element={<PatientsList accessToken={accessToken} />} />
          <Route path="/timeline/:patientId" element={<Timeline accessToken={accessToken} />} />
          <Route path="/practitioner-network" element={<PractitionerNetwork accessToken={accessToken} />} />
          <Route path="/" element={<Navigate to="/login" replace />} />
        </Routes>
      </Box>
    </Router>
  );
}

export default App;
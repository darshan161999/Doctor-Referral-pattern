import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import { Box, Typography, List, ListItem, ListItemText, CircularProgress, Alert, Button } from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';

const PatientsList = ({ accessToken }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Track the access token when the component mounts or updates
  useEffect(() => {
    console.log('PatientsList received access token:', {
      token: accessToken,
      tokenSnippet: accessToken ? accessToken.substring(0, 20) + '...' : 'null',
      tokenFormat: typeof accessToken,
      isBearerToken: accessToken ? accessToken.startsWith('eyJhbGciOi') : false,
      tokenParts: accessToken ? accessToken.split('.').length : 0,
      scopes: accessToken ? (JSON.parse(atob(accessToken.split('.')[1]))['scope'] || 'No scopes') : 'No token'
    });
  }, [accessToken]);

  // Memoize the data fetching to prevent unnecessary re-fetches
  const fetchPatients = useCallback(async () => {
    if (!accessToken) {
      setError('Please login to access patient list.');
      setLoading(false);
      return;
    }

    setLoading(true);
    console.log('Fetching patients with token before request:', {
      tokenSnippet: accessToken.substring(0, 20) + '...',
      tokenLength: accessToken.length,
      tokenFormat: typeof accessToken,
      isBearerToken: accessToken.startsWith('eyJhbGciOi'),  // Check if JWT
      tokenParts: accessToken.split('.').length,  // Verify JWT structure
      scopes: accessToken.split('.')[1] ? JSON.parse(atob(accessToken.split('.')[1]))['scope'] || 'No scopes' : 'Cannot decode'
    });

    try {
      const response = await axios.get('http://127.0.0.1:8000/patients', {
        headers: { 
          'access-token': accessToken,  // Use the correct header name
          'Content-Type': 'application/json'
        },
        // Enable detailed response logging
        validateStatus: (status) => true  // Allow all status codes for debugging
      });

      console.log('Patients response after request:', {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        data: response.data,
        config: response.config,
        requestTime: new Date().toISOString()
      });

      // Handle response format more robustly
      if (!response.data) {
        throw new Error('Empty response from server');
      }
      if (typeof response.data === 'string' || response.data instanceof String) {
        throw new Error('Unexpected string response: ' + response.data);
      }
      if (!response.data.patientIds) {
        throw new Error('Invalid response format: No patientIds found in ' + JSON.stringify(response.data));
      }
      setPatients(response.data.patientIds || []);
    } catch (err) {
      console.error('Error fetching patients:', {
        error: err,
        isAxiosError: err instanceof axios.AxiosError,
        message: err.message,
        response: err.response ? {
          status: err.response.status,
          statusText: err.response.statusText,
          headers: err.response.headers,
          data: err.response.data,
          config: err.response.config
        } : null,
        request: err.request ? {
          method: err.request.method,
          url: err.request.url,
          headers: err.request.getAllResponseHeaders(),
          requestTime: new Date().toISOString()
        } : null,
        config: err.config,
        tokenSnippet: accessToken ? accessToken.substring(0, 20) + '...' : 'null'
      });
      let errorMessage = 'Failed to fetch patients';
      if (err.response) {
        errorMessage += `: ${err.response.status} ${err.response.statusText} - ${JSON.stringify(err.response.data) || '[No detail]'}`;
      } else if (err.request) {
        errorMessage += ': Network error, no response received';
      } else {
        errorMessage += `: ${err.message}`;
      }
      setError(errorMessage);
    } finally {
      console.log('Fetching patients completed with token:', {
        tokenSnippet: accessToken ? accessToken.substring(0, 20) + '...' : 'null',
        loading: loading
      });
      setLoading(false);
    }
  }, [accessToken]);

  // Fetch data on mount or when accessToken changes
  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  // Memoize the patient list to prevent unnecessary re-renders
  const patientItems = useMemo(() => {
    return patients.map(patientId => (
      <ListItem 
        key={patientId} 
        button 
        component="a" 
        href={`/timeline/${patientId}`}
        sx={{ '&:hover': { backgroundColor: '#f5f5f5' } }}
      >
        <ListItemText primary={`Patient ID: ${patientId}`} />
      </ListItem>
    ));
  }, [patients]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" icon={<WarningIcon />}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Select a Patient
      </Typography>
      <List>
        {patientItems}
      </List>
      <Box mt={2}>
        <Button variant="contained" color="primary" href="/practitioner-network">
          Practitioner Care Network
        </Button>
      </Box>
    </Box>
  );
};

export default PatientsList;
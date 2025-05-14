import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import { Box, Typography, Card, CardContent, Grid, CircularProgress, Alert } from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import InfoIcon from '@mui/icons-material/Info';

const Timeline = ({ accessToken }) => {
  const { patientId } = useParams();
  const [timelineData, setTimelineData] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!accessToken) {
      setError('Please login to access patient timeline.');
      setLoading(false);
      return;
    }

    setLoading(true);
    Promise.all([
      axios.get(`http://127.0.0.1:8000/patient_timeline_visual/${patientId}`, {
        headers: { Authorization: `Bearer ${accessToken}` }
      }),
      axios.get(`http://127.0.0.1:8000/patient_insights/${patientId}`, {
        headers: { Authorization: `Bearer ${accessToken}` }
      })
    ])
      .then(([timelineRes, insightsRes]) => {
        setTimelineData(timelineRes.data);
        setInsights(insightsRes.data.insights);
        setLoading(false);
      })
      .catch(err => {
        setError(`Failed to fetch data: ${err.response?.data?.detail || err.message}`);
        setLoading(false);
      });
  }, [accessToken, patientId]);

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

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown Date';
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit', 
      hour12: true 
    });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Patient Timeline
      </Typography>
      <Typography>Patient ID: {timelineData?.patientId || 'Unknown'}</Typography>
      <Typography>Patient Name: {timelineData?.name || 'Unnamed'}</Typography>
      <Box mt={2}>
        {timelineData?.timeline?.map((event, index) => (
          <Card key={index} sx={{ mb: 2, bgcolor: event.isMajor ? '#ffebee' : '#f5f5f5' }}>
            <CardContent>
              <Typography variant="h6">
                {event.isMajor ? <ErrorIcon color="error" /> : <CheckCircleIcon color="success" />}
                {formatDate(event.eventDate)} - {event.eventDescription}
              </Typography>
              <Typography>Class Code: {event.classCode}</Typography>
              {event.referral && (
                <Typography>Referral: {event.referral.referralDate} by {event.referral.referredBy.name}</Typography>
              )}
              <Typography>Type: {event.subsequentActivities.encounters[0]?.type || 'Unknown'}</Typography>
              <Typography>Reason: {event.subsequentActivities.encounters[0]?.reasonCode || 'No reason'}</Typography>
              <Typography>Practitioners: {event.subsequentActivities.encounters[0]?.practitioners.map(p => p.name).join(', ') || 'None'}</Typography>
              {event.subsequentActivities.conditions.length > 0 && (
                <Typography>Conditions: {event.subsequentActivities.conditions.map(c => c.code).join(', ') || 'None'}</Typography>
              )}
              {event.subsequentActivities.procedures.length > 0 && (
                <Typography>Procedures: {event.subsequentActivities.procedures.map(p => p.code).join(', ') || 'None'}</Typography>
              )}
              {event.subsequentActivities.observations.length > 0 && (
                <Typography>Observations: {event.subsequentActivities.observations.map(o => `${o.code}: ${o.value} ${o.unit || ''}`).join(', ') || 'None'}</Typography>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>
      {insights && (
        <Box mt={2}>
          <Typography variant="h5">Patient Insights</Typography>
          <Typography>{insights.text || 'No insights available.'}</Typography>
        </Box>
      )}
    </Box>
  );
};

export default Timeline;
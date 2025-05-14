import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  CircularProgress, 
  Alert, 
  Stack 
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';

const PractitionerNetwork = ({ accessToken }) => {
  const [networkData, setNetworkData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!accessToken) {
      setError('Please login to access practitioner network.');
      setLoading(false);
      return;
    }

    setLoading(true);
    axios.get('http://127.0.0.1:8000/practitioner_care_network', {
      headers: { Authorization: `Bearer ${accessToken}` }
    })
      .then(response => {
        setNetworkData(response.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch practitioner network: ' + (err.response?.data?.detail || err.message));
        setLoading(false);
      });
  }, [accessToken]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ bgcolor: '#ffebee', borderRadius: 2, boxShadow: 1 }} icon={<WarningIcon />}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={ { 
      maxWidth: 1200, 
      margin: '0 auto', 
      padding: 2, 
      bgcolor: '#f8f9fa', 
      borderRadius: 2, 
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
      transition: 'all 0.3s ease'
    }}>
      <Typography 
        variant="h4" 
        gutterBottom 
        sx={{ 
          color: '#2c3e50', 
          fontWeight: 'bold', 
          textAlign: 'center', 
          paddingBottom: 2,
          borderBottom: '2px solid #3498db'
        }}
      >
        Practitioner Care Network
      </Typography>

      {networkData && networkData.network && (
        <Grid container spacing={3}>
          {networkData.network.map(node => (
            <Grid item xs={12} sm={6} md={4} key={node.id}>
              <Card 
                sx={{ 
                  bgcolor: '#ecf0f1', 
                  borderRadius: 2, 
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', 
                  transition: 'all 0.3s ease',
                  '&:hover': { 
                    transform: 'translateY(-5px)', 
                    boxShadow: '0 6px 12px rgba(0, 0, 0, 0.2)',
                    bgcolor: '#d5e8f7'
                  }
                }}
              >
                <CardContent>
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      color: '#2c3e50', 
                      fontWeight: 'bold', 
                      mb: 1 
                    }}
                  >
                    {node.name}
                  </Typography>
                  <Stack spacing={0.5}>
                    <Typography sx={{ color: '#34495e' }}>
                      Collaborations: <span style={{ color: '#27ae60' }}>{node.collaboration_count}</span>
                    </Typography>
                    <Typography sx={{ color: '#34495e' }}>
                      Centrality: <span style={{ color: '#e74c3c' }}>{node.centrality.toFixed(2)}</span>
                    </Typography>
                    <Typography sx={{ color: '#34495e' }}>
                      Specialty: <span style={{ color: '#8e44ad' }}>{node.specialty_cluster}</span>
                    </Typography>
                    <Typography sx={{ color: '#34495e' }}>
                      Location: <span style={{ color: '#2980b9' }}>{node.location.city}, {node.location.state}</span>
                    </Typography>
                    <Typography sx={{ color: '#34495e' }}>
                      Collaborators: {node.collaborators.length > 0 ? 
                        node.collaborators.map(c => 
                          <span key={c.id} style={{ color: '#27ae60', marginRight: 5 }}>{`${c.name} (Weight: ${c.weight})`}</span>
                        ) : 'None'}
                    </Typography>
                    <Typography sx={{ color: '#34495e' }}>
                      Community: <span style={{ color: '#f1c40f' }}>{node.community === -1 ? 'None' : `Community ${node.community}`}</span>
                    </Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {networkData && networkData.insights && (
        <Box mt={3} sx={{ 
          bgcolor: '#e8f5e9', 
          borderRadius: 2, 
          p: 2, 
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease'
        }}>
          <Typography 
            variant="h5" 
            sx={{ 
              color: '#27ae60', 
              fontWeight: 'bold', 
              mb: 1, 
              display: 'flex', 
              alignItems: 'center'
            }}
          >
            <InfoIcon sx={{ mr: 1, color: '#27ae60' }} />
            Network Insights
          </Typography>
          <Typography sx={{ color: '#2c3e50', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
            {networkData.insights.text || 'No insights available.'}
          </Typography>
        </Box>
      )}

      {networkData && (
        <Box mt={3} sx={{ 
          bgcolor: '#f5f7fa', 
          borderRadius: 2, 
          p: 2, 
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease'
        }}>
          <Typography 
            variant="h5" 
            sx={{ 
              color: '#2980b9', 
              fontWeight: 'bold', 
              mb: 1 
            }}
          >
            Network Summary
          </Typography>
          <Stack spacing={1}>
            <Typography sx={{ color: '#34495e' }}>
              Total Practitioners: <span style={{ color: '#e67e22' }}>{networkData.total_practitioners}</span>
            </Typography>
            <Typography sx={{ color: '#34495e' }}>
              Total Collaborations: <span style={{ color: '#e67e22' }}>{networkData.total_collaborations}</span>
            </Typography>
            <Typography sx={{ color: '#34495e' }}>
              Network Density: <span style={{ color: '#e67e22' }}>{networkData.network_density.toFixed(2)}</span>
            </Typography>
            <Typography sx={{ color: '#34495e' }}>
              Key Influencers: <span style={{ color: '#e74c3c' }}>{networkData.key_influencers.map(i => `${i[1].toFixed(2)} - ${networkData.network.find(n => n.id === i[0])?.name || 'Unknown'}`).join(', ') || 'No influencers available'}</span>
            </Typography>
          </Stack>
        </Box>
      )}
    </Box>
  );
};

export default PractitionerNetwork;
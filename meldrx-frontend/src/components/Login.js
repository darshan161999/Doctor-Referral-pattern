import React from 'react';
import { Box, Button, Typography } from '@mui/material';

const Login = () => {
  const handleLogin = () => {
    // Fetch the authorization URL from the backend and redirect
    fetch('http://127.0.0.1:8000/login')
      .then(response => response.json())
      .then(data => {
        window.location.href = data.auth_url;
      })
      .catch(error => {
        console.error('Error initiating login:', error);
        alert('Failed to initiate login. Please try again or contact support.');
      });
  };

  return (
    <Box 
      display="flex" 
      justifyContent="center" 
      alignItems="center" 
      minHeight="100vh"
      sx={{ bgcolor: '#ffffff' }}
    >
      <Box textAlign="center">
        <Typography variant="h4" gutterBottom sx={{ color: '#2c3e50', fontWeight: 'bold' }}>
          Welcome to MeldRx Doctor Referral
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleLogin}
          sx={{ 
            mt: 2, 
            px: 4, 
            py: 1.5, 
            fontSize: '1rem', 
            borderRadius: 2,
            backgroundColor: '#3498db',
            '&:hover': { backgroundColor: '#2980b9' }
          }}
        >
          Login with MeldRx
        </Button>
      </Box>
    </Box>
  );
};

export default Login;
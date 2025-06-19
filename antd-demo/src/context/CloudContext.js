import React, { createContext, useState, useContext } from 'react';

const CloudContext = createContext();

export const CloudProvider = ({ children }) => {
  const [cloudName, setCloudName] = useState('');

  return (
    <CloudContext.Provider value={{ cloudName, setCloudName }}>
      {children}
    </CloudContext.Provider>
  );
};

export const useCloud = () => {
  const context = useContext(CloudContext);
  if (!context) {
    throw new Error('useCloud must be used within a CloudProvider');
  }
  return context;
};

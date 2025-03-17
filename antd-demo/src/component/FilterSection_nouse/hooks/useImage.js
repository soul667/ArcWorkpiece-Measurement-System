import { useState, useEffect, useCallback } from 'react';
import { message } from 'antd';
import { loadImage, fetchCoordinateRanges } from '../utils/imageUtils';

/**
 * Custom hook for managing image loading and coordinate ranges
 * @param {Object} params Image parameters
 * @returns {Object} Image state and handlers
 */
const useImage = ({
  imageRef,
  currentView,
  onCoordinateRangesUpdate = () => {},
  onImageLoad = () => {}
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load new image when view changes
  const loadCurrentImage = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Cleanup old URL if exists
      if (imageRef.current.src && imageRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(imageRef.current.src);
      }
      
      // Load new image
      const imageUrl = await loadImage(currentView);
      imageRef.current.src = imageUrl;
      
      // Fetch coordinate ranges
      const ranges = await fetchCoordinateRanges();
      onCoordinateRangesUpdate(ranges);
      
      // Call onLoad callback when image loads
      imageRef.current.onload = () => {
        onImageLoad();
        setLoading(false);
      };
    } catch (error) {
      console.error('Image loading error:', error);
      setError(error.message);
      message.error(error.message);
      setLoading(false);
    }
  }, [currentView, imageRef, onCoordinateRangesUpdate, onImageLoad]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (imageRef.current.src && imageRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(imageRef.current.src);
      }
    };
  }, []);

  // Load image when view changes
  useEffect(() => {
    loadCurrentImage();
  }, [loadCurrentImage]);

  return {
    loading,
    error,
    reloadImage: loadCurrentImage
  };
};

export default useImage;

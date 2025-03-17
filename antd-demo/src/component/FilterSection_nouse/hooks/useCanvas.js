import { useState, useEffect, useCallback } from 'react';
import { calculateCanvasDimensions, drawCanvas, getRelativeCoordinates } from '../utils/canvasUtils';

/**
 * Custom hook for managing canvas operations
 * @param {Object} params Canvas parameters
 * @returns {Object} Canvas state and handlers
 */
const useCanvas = ({
  canvasRef,
  imageRef,
  isFullscreen,
  drawMode,
  xRegions,
  yRegions,
  lines,
  onLineAdd
}) => {
  const [canvasStyle, setCanvasStyle] = useState({
    border: '1px solid #d9d9d9',
    width: '100%',
    cursor: 'crosshair',
    transition: 'all 0.3s ease'
  });

  // Handle canvas resize
  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current.complete) return;
    
    const { width, height, scale } = calculateCanvasDimensions(
      canvas.parentElement,
      imageRef.current,
      isFullscreen
    );
    
    // Set display size
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    
    // Set actual size
    canvas.width = width * scale;
    canvas.height = height * scale;
    
    // Scale context
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);

    // Update position for fullscreen
    if (isFullscreen) {
      setCanvasStyle(prev => ({
        ...prev,
        position: 'fixed',
        left: `${(window.innerWidth - width) / 2}px`,
        top: `${(window.innerHeight - height) / 2}px`
      }));
    } else {
      setCanvasStyle(prev => ({
        ...prev,
        position: 'static',
        left: 'auto',
        top: 'auto'
      }));
    }
    
    // Redraw canvas
    drawCanvas(
      ctx,
      imageRef.current,
      xRegions,
      yRegions,
      lines,
      drawMode,
      width,
      height
    );
  }, [isFullscreen, drawMode, xRegions, yRegions, lines]);

  // Handle mouse events
  const handleMouseUp = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const { x, y } = getRelativeCoordinates(e, canvas);
    const newLine = drawMode === 'x' ? { x } : { y };
    onLineAdd(newLine);
  }, [drawMode, onLineAdd]);

  // Setup resize handler
  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Setup mouse handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('mouseup', handleMouseUp);
    return () => canvas.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseUp]);

  // Trigger resize when image loads or dependencies change
  useEffect(() => {
    const image = imageRef.current;
    if (image.complete) {
      handleResize();
    } else {
      image.onload = handleResize;
    }
  }, [handleResize]);

  return {
    canvasStyle,
    redraw: handleResize
  };
};

export default useCanvas;
